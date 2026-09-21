"""Persistence layer for Orivory knowledge-graph extraction.

The builder takes Memory rows, extracts entities/relations, and persists them
into the existing graph tables without requiring migrations.
"""
from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.graph.extraction import (
    ExtractedEntity,
    ExtractedRelation,
    extract_entities,
    extract_relations,
    normalize_entity_name,
    normalize_entity_type,
    normalize_relation_type,
)
from app.models.entity import Entity, MemoryEntity, Relation
from app.models.memory import Memory

GRAPH_EXTRACTED_AT_KEY = "graph_extracted_at"


@dataclass(frozen=True)
class GraphBuildResult:
    """Serializable summary returned by graph extraction tasks."""

    memory_id: str
    user_id: str | None
    skipped: bool = False
    entities_extracted: int = 0
    entities_created: int = 0
    entity_links_created: int = 0
    relations_extracted: int = 0
    relations_created: int = 0
    relations_updated: int = 0
    fallback_used: bool = False
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


async def build_memory_graph(
    db: AsyncSession,
    memory_id: UUID | str,
    *,
    force: bool = False,
) -> GraphBuildResult:
    """Build graph entities/relations for one memory using an AsyncSession."""
    memory = await db.get(Memory, memory_id)
    if memory is None:
        return GraphBuildResult(memory_id=str(memory_id), user_id=None, skipped=True, error="memory_not_found")

    if _already_processed(memory) and not force:
        return GraphBuildResult(memory_id=str(memory.id), user_id=str(memory.user_id), skipped=True)

    entity_result = await extract_entities(memory)
    entities = entity_result.entities
    # Both extraction legs run BEFORE the first DB write: SQLite takes its
    # write lock at the first flush and holds it until the commit, so a flush
    # here would keep every other writer (the request path, the drain, a second
    # build) waiting on ``busy_timeout`` behind the provider round trip below.
    # Collect -> call the provider -> persist.
    relation_result = await extract_relations(memory, entities)

    created_entities = 0
    created_links = 0
    entity_by_key: dict[tuple[str, str], Entity] = {}

    for extracted in entities:
        entity, created = await _get_or_create_entity_async(db, memory, extracted)
        if created:
            created_entities += 1
        entity_by_key[_entity_key(extracted.name, extracted.entity_type)] = entity
        link_created = await _upsert_memory_entity_async(db, memory, entity, extracted.salience)
        if link_created:
            created_links += 1
            entity.mention_count = (entity.mention_count or 0) + 1
        _touch_entity(entity, memory.captured_at)

    relation_stats = await _persist_relations_async(db, memory, relation_result.relations, entity_by_key)

    # R27(p2): merge the processed marker into a FRESH read — same reason as
    # the sync face below (a concurrent supersede must not be clobbered).
    await db.refresh(memory, ["extra_metadata"])
    _mark_processed(memory, entity_result, relation_result)
    await db.commit()

    return GraphBuildResult(
        memory_id=str(memory.id),
        user_id=str(memory.user_id),
        entities_extracted=len(entities),
        entities_created=created_entities,
        entity_links_created=created_links,
        relations_extracted=len(relation_result.relations),
        relations_created=relation_stats[0],
        relations_updated=relation_stats[1],
        fallback_used=entity_result.fallback_used or relation_result.fallback_used,
    )


def _run_extraction(coro):
    """Drive one extraction coroutine to completion from sync code.

    ``build_memory_graph_sync`` owns a sync ``Session``, so it drives the async
    extraction with ``asyncio.run`` — which refuses to run inside a running
    event loop. That refusal is the failure mode that kept memory graphs
    silently empty on every loop-driven caller (the best-effort handler at the
    call site swallowed the bare ``RuntimeError``; P2/T9): fail loudly, with the
    fix in the message, and close the coroutine so nothing leaks a
    never-awaited ``RuntimeWarning``.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()
    raise RuntimeError(
        "build_memory_graph_sync() cannot run on a running event loop: it "
        "drives asyncio.run() internally. Offload it with asyncio.to_thread "
        "(write_back.safe_enqueue_graph_build schedules it that way)."
    )


def build_memory_graph_sync(
    db: Session,
    memory_id: UUID | str,
    *,
    force: bool = False,
) -> GraphBuildResult:
    """Synchronous builder for off-loop callers.

    Used from CLI/script contexts and from worker threads
    (``write_back.safe_enqueue_graph_build`` schedules it there via
    ``asyncio.to_thread`` when a loop is running, and calls it inline when
    none is); never from a running event loop — see ``_run_extraction``.
    """
    memory = db.get(Memory, memory_id)
    if memory is None:
        return GraphBuildResult(memory_id=str(memory_id), user_id=None, skipped=True, error="memory_not_found")

    if _already_processed(memory) and not force:
        return GraphBuildResult(memory_id=str(memory.id), user_id=str(memory.user_id), skipped=True)

    entity_result = _run_extraction(extract_entities(memory))
    entities = entity_result.entities
    # Collect -> call the provider -> persist (id 288): the entity flush used to
    # sit between the two legs, so the SQLite write lock was held across the
    # relations round trip and any other writer failed `database is locked`.
    relation_result = _run_extraction(extract_relations(memory, entities))

    created_entities = 0
    created_links = 0
    entity_by_key: dict[tuple[str, str], Entity] = {}

    for extracted in entities:
        entity, created = _get_or_create_entity_sync(db, memory, extracted)
        if created:
            created_entities += 1
        entity_by_key[_entity_key(extracted.name, extracted.entity_type)] = entity
        link_created = _upsert_memory_entity_sync(db, memory, entity, extracted.salience)
        if link_created:
            created_links += 1
            entity.mention_count = (entity.mention_count or 0) + 1
        _touch_entity(entity, memory.captured_at)

    relation_stats = _persist_relations_sync(db, memory, relation_result.relations, entity_by_key)

    # R27(p2): the write path no longer serializes this build against later
    # writes to the SAME row, so the processed marker is merged into a FRESH
    # read of the metadata: the snapshot loaded at extraction time would
    # clobber a ``cm_*`` marker (a supersede) that landed while the extraction
    # ran. Residual window = this SELECT->COMMIT span; a per-row lock is the
    # upgrade if that ever matters.
    db.refresh(memory, ["extra_metadata"])
    _mark_processed(memory, entity_result, relation_result)
    db.commit()

    return GraphBuildResult(
        memory_id=str(memory.id),
        user_id=str(memory.user_id),
        entities_extracted=len(entities),
        entities_created=created_entities,
        entity_links_created=created_links,
        relations_extracted=len(relation_result.relations),
        relations_created=relation_stats[0],
        relations_updated=relation_stats[1],
        fallback_used=entity_result.fallback_used or relation_result.fallback_used,
    )


def _already_processed(memory: Memory) -> bool:
    return bool((memory.extra_metadata or {}).get(GRAPH_EXTRACTED_AT_KEY))


def _entity_key(name: str, entity_type: str) -> tuple[str, str]:
    return (normalize_entity_name(name).casefold(), normalize_entity_type(entity_type))


def _entity_identity_query(user_id, name: str, entity_type: str):
    """The ONE entity-identity lookup, shared by both builders.

    Case-insensitive (the graph treats 'Mom' and 'mom' as one entity) and
    DETERMINISTIC: the unique key is on the raw name, so a pre-existing pair of
    case variants used to raise ``MultipleResultsFound`` and lose the whole
    memory's graph. The oldest row wins.
    """
    return (
        select(Entity)
        .where(
            Entity.user_id == user_id,
            func.lower(Entity.name) == name.casefold(),
            Entity.entity_type == entity_type,
        )
        .order_by(Entity.created_at, Entity.id)
        .limit(1)
    )


def _touch_entity(entity: Entity, seen_at: datetime) -> None:
    if entity.first_seen_at is None or seen_at < entity.first_seen_at:
        entity.first_seen_at = seen_at
    if entity.last_seen_at is None or seen_at > entity.last_seen_at:
        entity.last_seen_at = seen_at


def _merge_aliases(existing: Iterable[str] | None, new_aliases: Iterable[str]) -> list[str]:
    merged: list[str] = []
    for alias in list(existing or []) + list(new_aliases):
        alias = alias.strip()
        if alias and alias not in merged:
            merged.append(alias)
    return merged[:20]


async def _get_or_create_entity_async(
    db: AsyncSession,
    memory: Memory,
    extracted: ExtractedEntity,
) -> tuple[Entity, bool]:
    name = normalize_entity_name(extracted.name)
    entity_type = normalize_entity_type(extracted.entity_type)
    entity = (await db.execute(
        _entity_identity_query(memory.user_id, name, entity_type)
    )).scalars().first()
    if entity:
        entity.aliases = _merge_aliases(entity.aliases, extracted.aliases)
        if extracted.description and not entity.description:
            entity.description = extracted.description
        return entity, False

    entity = Entity(
        user_id=memory.user_id,
        name=name,
        entity_type=entity_type,
        aliases=_merge_aliases([], extracted.aliases),
        description=extracted.description,
        first_seen_at=memory.captured_at,
        last_seen_at=memory.captured_at,
        mention_count=0,
        extra_metadata={},
    )
    try:
        # The savepoint scopes the race: a rival build that landed the same
        # (user, name, type) between the SELECT above and this INSERT answers
        # IntegrityError, the savepoint rolls back, and THIS build converges on
        # the row the rival wrote instead of losing the memory's whole graph.
        async with db.begin_nested():
            db.add(entity)
            await db.flush()
    except IntegrityError:
        entity = (await db.execute(
            _entity_identity_query(memory.user_id, name, entity_type)
        )).scalars().first()
        if entity is None:
            raise
        return entity, False
    return entity, True


def _get_or_create_entity_sync(
    db: Session,
    memory: Memory,
    extracted: ExtractedEntity,
) -> tuple[Entity, bool]:
    name = normalize_entity_name(extracted.name)
    entity_type = normalize_entity_type(extracted.entity_type)
    entity = db.execute(
        _entity_identity_query(memory.user_id, name, entity_type)
    ).scalars().first()
    if entity:
        entity.aliases = _merge_aliases(entity.aliases, extracted.aliases)
        if extracted.description and not entity.description:
            entity.description = extracted.description
        return entity, False

    entity = Entity(
        user_id=memory.user_id,
        name=name,
        entity_type=entity_type,
        aliases=_merge_aliases([], extracted.aliases),
        description=extracted.description,
        first_seen_at=memory.captured_at,
        last_seen_at=memory.captured_at,
        mention_count=0,
        extra_metadata={},
    )
    try:
        # Same savepoint contract as the async face above (id 286).
        with db.begin_nested():
            db.add(entity)
            db.flush()
    except IntegrityError:
        entity = db.execute(
            _entity_identity_query(memory.user_id, name, entity_type)
        ).scalars().first()
        if entity is None:
            raise
        return entity, False
    return entity, True


async def _upsert_memory_entity_async(
    db: AsyncSession,
    memory: Memory,
    entity: Entity,
    salience: float,
) -> bool:
    link = (await db.execute(
        select(MemoryEntity).where(
            MemoryEntity.memory_id == memory.id,
            MemoryEntity.entity_id == entity.id,
        )
    )).scalar_one_or_none()
    if link:
        link.salience = max(link.salience or 0.0, salience)
        return False
    db.add(MemoryEntity(memory_id=memory.id, entity_id=entity.id, salience=salience))
    return True


def _upsert_memory_entity_sync(db: Session, memory: Memory, entity: Entity, salience: float) -> bool:
    link = db.execute(
        select(MemoryEntity).where(
            MemoryEntity.memory_id == memory.id,
            MemoryEntity.entity_id == entity.id,
        )
    ).scalar_one_or_none()
    if link:
        link.salience = max(link.salience or 0.0, salience)
        return False
    db.add(MemoryEntity(memory_id=memory.id, entity_id=entity.id, salience=salience))
    return True


async def _persist_relations_async(
    db: AsyncSession,
    memory: Memory,
    relations: list[ExtractedRelation],
    entity_by_key: dict[tuple[str, str], Entity],
) -> tuple[int, int]:
    created = 0
    updated = 0
    for relation in relations:
        result = await _upsert_relation_async(db, memory, relation, entity_by_key)
        if result == "created":
            created += 1
        elif result == "updated":
            updated += 1
    return created, updated


def _persist_relations_sync(
    db: Session,
    memory: Memory,
    relations: list[ExtractedRelation],
    entity_by_key: dict[tuple[str, str], Entity],
) -> tuple[int, int]:
    created = 0
    updated = 0
    for relation in relations:
        result = _upsert_relation_sync(db, memory, relation, entity_by_key)
        if result == "created":
            created += 1
        elif result == "updated":
            updated += 1
    return created, updated


def _entity_lookup(entity_by_key: dict[tuple[str, str], Entity], name: str) -> Entity | None:
    normalized = normalize_entity_name(name).casefold()
    for (candidate_name, _entity_type), entity in entity_by_key.items():
        if candidate_name == normalized:
            return entity
    return None


async def _upsert_relation_async(
    db: AsyncSession,
    memory: Memory,
    extracted: ExtractedRelation,
    entity_by_key: dict[tuple[str, str], Entity],
) -> str | None:
    source = _entity_lookup(entity_by_key, extracted.source)
    target = _entity_lookup(entity_by_key, extracted.target)
    if source is None or target is None or source.id == target.id:
        return None
    relation_type = normalize_relation_type(extracted.relation)
    relation = (await db.execute(
        select(Relation).where(
            Relation.user_id == memory.user_id,
            Relation.source_entity_id == source.id,
            Relation.target_entity_id == target.id,
            Relation.relation == relation_type,
        )
    )).scalar_one_or_none()
    if relation:
        relation.weight = max(relation.weight or 0.0, extracted.weight)
        relation.evidence_count = max(relation.evidence_count or 1, 1) + 1
        relation.last_evidence_at = memory.captured_at
        relation.extra_metadata = _merge_relation_metadata(relation.extra_metadata, extracted)
        return "updated"
    db.add(Relation(
        user_id=memory.user_id,
        source_entity_id=source.id,
        target_entity_id=target.id,
        relation=relation_type,
        weight=extracted.weight,
        evidence_count=1,
        last_evidence_at=memory.captured_at,
        extra_metadata=_merge_relation_metadata({}, extracted),
    ))
    return "created"


def _upsert_relation_sync(
    db: Session,
    memory: Memory,
    extracted: ExtractedRelation,
    entity_by_key: dict[tuple[str, str], Entity],
) -> str | None:
    source = _entity_lookup(entity_by_key, extracted.source)
    target = _entity_lookup(entity_by_key, extracted.target)
    if source is None or target is None or source.id == target.id:
        return None
    relation_type = normalize_relation_type(extracted.relation)
    relation = db.execute(
        select(Relation).where(
            Relation.user_id == memory.user_id,
            Relation.source_entity_id == source.id,
            Relation.target_entity_id == target.id,
            Relation.relation == relation_type,
        )
    ).scalar_one_or_none()
    if relation:
        relation.weight = max(relation.weight or 0.0, extracted.weight)
        relation.evidence_count = max(relation.evidence_count or 1, 1) + 1
        relation.last_evidence_at = memory.captured_at
        relation.extra_metadata = _merge_relation_metadata(relation.extra_metadata, extracted)
        return "updated"
    db.add(Relation(
        user_id=memory.user_id,
        source_entity_id=source.id,
        target_entity_id=target.id,
        relation=relation_type,
        weight=extracted.weight,
        evidence_count=1,
        last_evidence_at=memory.captured_at,
        extra_metadata=_merge_relation_metadata({}, extracted),
    ))
    return "created"


def _merge_relation_metadata(existing: dict | None, extracted: ExtractedRelation) -> dict:
    metadata = dict(existing or {})
    if extracted.reason:
        reasons = list(metadata.get("reasons") or [])
        if extracted.reason not in reasons:
            reasons.append(extracted.reason)
        metadata["reasons"] = reasons[-5:]
    return metadata


def _mark_processed(memory: Memory, entity_result, relation_result) -> None:
    # Metadata-only write, and it runs AFTER the memory's index intent has
    # been drained: bumping ``revision``/enqueuing here would loop
    # drain -> upsert -> graph -> enqueue forever. Leave both untouched.
    metadata = dict(memory.extra_metadata or {})
    # Decided HERE, not at the two call sites: a contract-breaking payload from
    # EITHER leg is not a finished extraction (ids 363 and its relations
    # sibling), and a third caller must not be able to stamp one as processed.
    complete = not (entity_result.schema_error or relation_result.schema_error)
    if complete:
        metadata[GRAPH_EXTRACTED_AT_KEY] = datetime.now(UTC).isoformat()
    else:
        # A contract-breaking payload is not a finished extraction: no sticky
        # marker, so the next build retries instead of leaving the memory
        # graph-less forever (``_already_processed`` skips every later build
        # unless forced). A provider outage does not land here — the
        # deterministic fallback IS the product on a key-less install.
        metadata.pop(GRAPH_EXTRACTED_AT_KEY, None)
    metadata["graph_entity_count"] = len(entity_result.entities)
    metadata["graph_relation_count"] = len(relation_result.relations)
    metadata["graph_extraction_fallback_used"] = bool(
        entity_result.fallback_used or relation_result.fallback_used
    )
    if entity_result.error:
        metadata["graph_entity_error"] = entity_result.error[:500]
    else:
        metadata.pop("graph_entity_error", None)
    if relation_result.error:
        metadata["graph_relation_error"] = relation_result.error[:500]
    else:
        metadata.pop("graph_relation_error", None)
    memory.extra_metadata = metadata
