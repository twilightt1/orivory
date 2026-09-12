"""LLM + fallback extraction for Orivory's personal knowledge graph.

This module intentionally has no database dependency. It accepts a Memory-like
object and returns normalized extraction objects that the graph builder can
persist into the existing `entities`, `memory_entities`, and `relations` tables.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

# Shared client seam: tests patch <module>._get_client, which rebinds
# this module attribute and is picked up by all call sites below.
from app.agents.llm_client import get_llm_client as _get_client
from app.agents.llm_client import get_sync_llm_client as _get_sync_client
from app.agents.llm_parsing import (
    coerce_float,
    coerce_string_list,
    parse_llm_json_object,
)
from app.config import settings
from app.models.entity import ENTITY_TYPES, RELATION_TYPES
from app.models.memory import Memory

log = logging.getLogger(__name__)


MAX_ENTITIES = 12
MAX_RELATIONS = 24

_DATE_RE = re.compile(r"\b(?:20\d{2}[-/][01]?\d[-/][0-3]?\d|[0-3]?\d[-/][01]?\d[-/]20\d{2})\b")
_CAPITALIZED_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][\wÀ-ỹ]+|[A-Z]{2,})(?:\s+(?:[A-Z][\wÀ-ỹ]+|[A-Z]{2,}|\d+)){0,3}\b",
    re.UNICODE,
)
_ALLOWED_ENTITY_TYPES = set(ENTITY_TYPES)
_ALLOWED_RELATION_TYPES = set(RELATION_TYPES)


@dataclass(frozen=True)
class ExtractedEntity:
    """Normalized entity extracted from a memory."""

    name: str
    entity_type: str = "other"
    aliases: list[str] = field(default_factory=list)
    salience: float = 0.5
    description: str | None = None


@dataclass(frozen=True)
class ExtractedRelation:
    """Normalized relation between two extracted entity names."""

    source: str
    target: str
    relation: str = "related_to"
    weight: float = 0.5
    reason: str | None = None


@dataclass(frozen=True)
class EntityExtractionResult:
    entities: list[ExtractedEntity]
    fallback_used: bool = False
    error: str | None = None
    raw_preview: str | None = None


@dataclass(frozen=True)
class RelationExtractionResult:
    relations: list[ExtractedRelation]
    fallback_used: bool = False
    error: str | None = None
    raw_preview: str | None = None


ENTITY_EXTRACTION_PROMPT = """You extract a personal knowledge graph from a second-brain memory.
Return ONLY valid JSON. No markdown.

Allowed entity types:
- person
- project
- topic
- concept
- organization
- place
- date
- event
- media
- other

Rules:
- Extract at most 12 important entities.
- Prefer stable names over pronouns.
- Use aliases for short names or alternate spellings.
- Salience is 0.0 to 1.0.
- Descriptions must be one short phrase.

Memory title:
{title}

Memory summary:
{summary}

Memory tags:
{tags}

Memory content:
{content}

Output JSON:
{{
  "entities": [
    {{
      "name": "Project Atlas",
      "type": "project",
      "aliases": ["Atlas"],
      "salience": 0.85,
      "description": "Project discussed in the memory"
    }}
  ]
}}
"""


RELATION_EXTRACTION_PROMPT = """You extract semantic relations between already-detected entities in one memory.
Return ONLY valid JSON. No markdown.

Allowed relation types:
- related_to
- works_on
- knows
- owns
- part_of
- mentioned_in
- references
- contradicts
- follows
- precedes
- summarizes

Rules:
- Use ONLY entity names from the provided entity list.
- Ignore uncertain/self relations.
- Weight is 0.0 to 1.0.
- Extract at most 24 relations.

Memory title:
{title}

Memory summary:
{summary}

Memory content:
{content}

Known entities:
{entities}

Output JSON:
{{
  "relations": [
    {{
      "source": "Mom",
      "target": "Project Atlas",
      "relation": "references",
      "weight": 0.75,
      "reason": "Mom discussed Project Atlas"
    }}
  ]
}}
"""


def normalize_entity_type(value: object) -> str:
    normalized = str(value or "other").strip().casefold().replace(" ", "_")
    return normalized if normalized in _ALLOWED_ENTITY_TYPES else "other"


def normalize_relation_type(value: object) -> str:
    normalized = str(value or "related_to").strip().casefold().replace(" ", "_")
    return normalized if normalized in _ALLOWED_RELATION_TYPES else "related_to"


def normalize_entity_name(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:255]


def _memory_text(memory: Memory, *, max_chars: int = 6000) -> dict[str, str]:
    content = (getattr(memory, "content", None) or "").strip()
    return {
        "title": (getattr(memory, "title", None) or "").strip(),
        "summary": (getattr(memory, "summary", None) or "").strip(),
        "tags": ", ".join(getattr(memory, "tags", None) or []),
        "content": content[:max_chars],
    }


def _dedupe_entities(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
    best: dict[tuple[str, str], ExtractedEntity] = {}
    for entity in entities:
        name = normalize_entity_name(entity.name)
        if not name:
            continue
        entity_type = normalize_entity_type(entity.entity_type)
        key = (name.casefold(), entity_type)
        clean = ExtractedEntity(
            name=name,
            entity_type=entity_type,
            aliases=list(dict.fromkeys(a.strip() for a in entity.aliases if a and a.strip()))[:8],
            salience=coerce_float(entity.salience, 0.5, minimum=0.0, maximum=1.0),
            description=(entity.description or None),
        )
        previous = best.get(key)
        if previous is None or clean.salience > previous.salience:
            best[key] = clean
    return sorted(best.values(), key=lambda e: e.salience, reverse=True)[:MAX_ENTITIES]


def _entity_from_mapping(item: dict[str, Any]) -> ExtractedEntity | None:
    name = normalize_entity_name(item.get("name"))
    if not name:
        return None
    return ExtractedEntity(
        name=name,
        entity_type=normalize_entity_type(item.get("type") or item.get("entity_type")),
        aliases=coerce_string_list(item.get("aliases"), limit=8),
        salience=coerce_float(item.get("salience"), 0.5, minimum=0.0, maximum=1.0),
        description=str(item.get("description")).strip()[:500] if item.get("description") else None,
    )


def _fallback_entities(memory: Memory) -> list[ExtractedEntity]:
    """Cheap deterministic fallback when the LLM is unavailable."""
    text_parts = _memory_text(memory)
    combined = "\n".join(text_parts.values())
    entities: list[ExtractedEntity] = []

    for tag in getattr(memory, "tags", None) or []:
        name = normalize_entity_name(tag)
        if name:
            entities.append(ExtractedEntity(name=name, entity_type="topic", salience=0.55))

    for match in _DATE_RE.findall(combined):
        entities.append(ExtractedEntity(name=match, entity_type="date", salience=0.5))

    for match in _CAPITALIZED_PHRASE_RE.findall(combined):
        name = normalize_entity_name(match)
        if len(name) >= 3 and name.casefold() not in {"the", "and", "for"}:
            inferred = "project" if name.casefold().startswith("project ") else "other"
            entities.append(ExtractedEntity(name=name, entity_type=inferred, salience=0.45))

    return _dedupe_entities(entities)


def _dedupe_relations(relations: list[ExtractedRelation]) -> list[ExtractedRelation]:
    best: dict[tuple[str, str, str], ExtractedRelation] = {}
    for relation in relations:
        source = normalize_entity_name(relation.source)
        target = normalize_entity_name(relation.target)
        if not source or not target or source.casefold() == target.casefold():
            continue
        relation_type = normalize_relation_type(relation.relation)
        key = (source.casefold(), target.casefold(), relation_type)
        clean = ExtractedRelation(
            source=source,
            target=target,
            relation=relation_type,
            weight=coerce_float(relation.weight, 0.5, minimum=0.0, maximum=1.0),
            reason=(relation.reason or None),
        )
        previous = best.get(key)
        if previous is None or clean.weight > previous.weight:
            best[key] = clean
    return sorted(best.values(), key=lambda r: r.weight, reverse=True)[:MAX_RELATIONS]


def _relation_from_mapping(item: dict[str, Any], allowed_names: set[str]) -> ExtractedRelation | None:
    source = normalize_entity_name(item.get("source"))
    target = normalize_entity_name(item.get("target"))
    if source.casefold() not in allowed_names or target.casefold() not in allowed_names:
        return None
    if source.casefold() == target.casefold():
        return None
    return ExtractedRelation(
        source=source,
        target=target,
        relation=normalize_relation_type(item.get("relation")),
        weight=coerce_float(item.get("weight"), 0.5, minimum=0.0, maximum=1.0),
        reason=str(item.get("reason")).strip()[:500] if item.get("reason") else None,
    )


def _entity_request(memory: Memory, model: str | None) -> dict[str, Any]:
    text = _memory_text(memory)
    return {
        "model": model or settings.LLM_MODEL,
        "messages": [{"role": "user", "content": ENTITY_EXTRACTION_PROMPT.format(**text)}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "extra_headers": {
            "HTTP-Referer": settings.FRONTEND_URL,
            "X-Title": "Orivory Graph Extraction",
        },
    }


def _relation_request(
    memory: Memory,
    entities: list[ExtractedEntity],
    model: str | None,
) -> dict[str, Any]:
    text = _memory_text(memory)
    entity_lines = "\n".join(f"- {e.name} ({e.entity_type})" for e in entities)
    return {
        "model": model or settings.LLM_MODEL,
        "messages": [{
            "role": "user",
            "content": RELATION_EXTRACTION_PROMPT.format(**text, entities=entity_lines),
        }],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "extra_headers": {
            "HTTP-Referer": settings.FRONTEND_URL,
            "X-Title": "Orivory Relation Extraction",
        },
    }


def _entity_result(resp: Any, memory: Memory) -> EntityExtractionResult:
    parsed = parse_llm_json_object(resp.choices[0].message.content)
    if not parsed.ok or parsed.data is None:
        fallback = _fallback_entities(memory)
        return EntityExtractionResult(
            entities=fallback,
            fallback_used=True,
            error=parsed.error or "invalid_entity_json",
            raw_preview=parsed.raw_preview,
        )

    raw_entities = parsed.data.get("entities", [])
    if not isinstance(raw_entities, list):
        raw_entities = []
    entities = [entity for item in raw_entities if isinstance(item, dict) for entity in [_entity_from_mapping(item)] if entity]
    return EntityExtractionResult(
        entities=_dedupe_entities(entities),
        fallback_used=False,
        raw_preview=parsed.raw_preview,
    )


def _relation_result(resp: Any, entities: list[ExtractedEntity]) -> RelationExtractionResult:
    parsed = parse_llm_json_object(resp.choices[0].message.content)
    if not parsed.ok or parsed.data is None:
        fallback = _fallback_relations(entities)
        return RelationExtractionResult(
            relations=fallback,
            fallback_used=True,
            error=parsed.error or "invalid_relation_json",
            raw_preview=parsed.raw_preview,
        )

    allowed_names = {entity.name.casefold() for entity in entities}
    raw_relations = parsed.data.get("relations", [])
    if not isinstance(raw_relations, list):
        raw_relations = []
    relations = [
        relation
        for item in raw_relations
        if isinstance(item, dict)
        for relation in [_relation_from_mapping(item, allowed_names)]
        if relation
    ]
    # Keep a deterministic co-occurrence baseline even when the LLM returns
    # semantic relations. Different relation types are preserved by the
    # dedupe key, while duplicate triples keep the stronger weight.
    relations = relations + _fallback_relations(entities)
    return RelationExtractionResult(
        relations=_dedupe_relations(relations),
        fallback_used=False,
        raw_preview=parsed.raw_preview,
    )


def _fallback_relations(entities: list[ExtractedEntity]) -> list[ExtractedRelation]:
    top = _dedupe_entities(entities)[:8]
    relations: list[ExtractedRelation] = []
    for idx, source in enumerate(top):
        for target in top[idx + 1:]:
            weight = min(0.8, 0.35 + ((source.salience + target.salience) / 4.0))
            relations.append(
                ExtractedRelation(
                    source=source.name,
                    target=target.name,
                    relation="mentioned_in",
                    weight=weight,
                    reason="Co-mentioned in the same memory",
                )
            )
    return _dedupe_relations(relations)


async def extract_entities(memory: Memory, *, model: str | None = None) -> EntityExtractionResult:
    """Extract normalized graph entities from a memory.

    The LLM path is best-effort. Any API/JSON/schema issue falls back to
    deterministic tag/date/capitalized-phrase extraction.
    """
    text = _memory_text(memory)
    if not any(text.values()):
        return EntityExtractionResult(entities=[])

    try:
        resp = await _get_client().chat.completions.create(**_entity_request(memory, model))
        return _entity_result(resp, memory)
    except Exception as exc:
        log.warning("Entity extraction fallback", extra={"error": str(exc)})
        return EntityExtractionResult(
            entities=_fallback_entities(memory),
            fallback_used=True,
            error=str(exc),
        )


def extract_entities_sync(memory: Memory, *, model: str | None = None) -> EntityExtractionResult:
    """Extract entities without creating an event loop for sync workers."""
    text = _memory_text(memory)
    if not any(text.values()):
        return EntityExtractionResult(entities=[])

    try:
        resp = _get_sync_client().chat.completions.create(**_entity_request(memory, model))
        return _entity_result(resp, memory)
    except Exception as exc:
        log.warning("Entity extraction fallback", extra={"error": str(exc)})
        return EntityExtractionResult(
            entities=_fallback_entities(memory),
            fallback_used=True,
            error=str(exc),
        )


async def extract_relations(
    memory: Memory,
    entities: list[ExtractedEntity],
    *,
    model: str | None = None,
) -> RelationExtractionResult:
    """Extract normalized graph relations between already-detected entities."""
    entities = _dedupe_entities(entities)
    if len(entities) < 2:
        return RelationExtractionResult(relations=[])

    try:
        resp = await _get_client().chat.completions.create(**_relation_request(memory, entities, model))
        return _relation_result(resp, entities)
    except Exception as exc:
        log.warning("Relation extraction fallback", extra={"error": str(exc)})
        return RelationExtractionResult(
            relations=_fallback_relations(entities),
            fallback_used=True,
            error=str(exc),
        )


def extract_relations_sync(
    memory: Memory,
    entities: list[ExtractedEntity],
    *,
    model: str | None = None,
) -> RelationExtractionResult:
    """Extract relations without creating an event loop for sync workers."""
    entities = _dedupe_entities(entities)
    if len(entities) < 2:
        return RelationExtractionResult(relations=[])

    try:
        resp = _get_sync_client().chat.completions.create(**_relation_request(memory, entities, model))
        return _relation_result(resp, entities)
    except Exception as exc:
        log.warning("Relation extraction fallback", extra={"error": str(exc)})
        return RelationExtractionResult(
            relations=_fallback_relations(entities),
            fallback_used=True,
            error=str(exc),
        )
