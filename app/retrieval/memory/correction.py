"""cm_* metadata helpers + pure rules for correctable memory (spec 2026-09-12)."""
from __future__ import annotations
import re
from uuid import UUID

CM_ASSERTION = "cm_assertion"
CM_SUBJECT = "cm_subject"
CM_ATTRIBUTE = "cm_attribute"
CM_SCOPE = "cm_scope"
CM_VALID_FROM = "cm_valid_from"
CM_VALID_TO = "cm_valid_to"
CM_SUPERSEDES = "cm_supersedes"
CM_SUPERSEDED_BY = "cm_superseded_by"
CM_EVIDENCE_IDS = "cm_evidence_ids"
CM_DERIVED_FROM = "cm_derived_from"
CM_DERIVED_DIRTY = "cm_derived_dirty"
CM_NEEDS_CHECK = "cm_needs_check"

DEFAULT_SCOPE = "default"

# ponytail: pronoun list is heuristic; add words only when eval shows a miss.
_PRONOUNS = frozenset({
    "it", "its", "this", "that", "these", "those", "he", "him", "his",
    "she", "her", "they", "them", "their", "no", "nó", "chúng", "đó",
})
_WORD = re.compile(r"[a-zà-ỹ]+", re.IGNORECASE)


def normalize_slot(value: str | None) -> str:
    return " ".join((value or "").strip().lower().split())


def get_cm(memory) -> dict:
    return dict(memory.extra_metadata or {})


def set_cm(memory, patch: dict) -> None:
    memory.extra_metadata = {**(memory.extra_metadata or {}), **patch}


def state_of(memory) -> str:
    meta = get_cm(memory)
    if meta.get(CM_SUPERSEDED_BY):
        return "superseded"
    if meta.get(CM_NEEDS_CHECK):
        return "needs-check"
    return "current"


def needs_rewrite(query: str) -> bool:
    return any(w in _PRONOUNS for w in _WORD.findall(query or ""))


def find_derived_dependent_ids(memories, erased_ids: set[str]) -> list[str]:
    out = []
    for m in memories:
        try:
            meta = get_cm(m)
            deps = set(meta.get(CM_DERIVED_FROM) or [])
        except (TypeError, AttributeError):
            continue
        if deps & erased_ids and not meta.get(CM_DERIVED_DIRTY):
            out.append(str(m.id))
    return out
