"""cm_* metadata helpers + pure rules for correctable memory (spec 2026-09-12)."""
from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import NamedTuple
from uuid import uuid4

from sqlalchemy import select

from app.models.memory import Memory

CM_ASSERTION = "cm_assertion"
CM_SUBJECT = "cm_subject"
CM_ATTRIBUTE = "cm_attribute"
CM_SCOPE = "cm_scope"
CM_VALID_FROM = "cm_valid_from"
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
    "she", "her", "they", "them", "their", "nó", "chúng", "đó",
})
_WORD = re.compile(r"[a-zà-ỹ]+", re.IGNORECASE)


def normalize_slot(value: str | None) -> str:
    return " ".join((value or "").strip().lower().split())


class Slot(NamedTuple):
    """Identity triple for one correctable fact, normalized once at build.

    Empty scope means "unspecified" (drives the ambiguous-scope path);
    use ``Slot.of`` so normalization lives in exactly one place.
    """

    subject: str = ""
    attribute: str = ""
    scope: str = ""

    @classmethod
    def of(cls, subject: str | None = "", attribute: str | None = "",
           scope: str | None = DEFAULT_SCOPE) -> Slot | None:
        subj, attr = normalize_slot(subject), normalize_slot(attribute)
        if not subj or not attr:
            return None
        return cls(subj, attr, normalize_slot(scope))

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.subject, self.attribute, self.scope or DEFAULT_SCOPE)

    @property
    def has_scope(self) -> bool:
        return bool(self.scope)

    def matches(self, memory) -> bool:
        return _stored_key(memory) == self.key


def _stored_key(memory) -> tuple[str, str, str]:
    """Normalized identity triple as stored in a memory's metadata."""
    meta = get_cm(memory)
    return (normalize_slot(meta.get(CM_SUBJECT)),
            normalize_slot(meta.get(CM_ATTRIBUTE)),
            normalize_slot(meta.get(CM_SCOPE, DEFAULT_SCOPE)))


def get_cm(memory) -> dict:
    return dict(memory.extra_metadata or {})


def set_cm(memory, patch: dict) -> None:
    memory.extra_metadata = {**(memory.extra_metadata or {}), **patch}


def state_of(memory) -> str:
    """One question for a memory's lifecycle state.

    Precedence: superseded > dirty > needs-check > current. A superseded
    memory stays "superseded" even if also dirty; dirty (stale derived
    view) outranks needs-check because it must not be served either way.
    """
    meta = get_cm(memory)
    if meta.get(CM_SUPERSEDED_BY):
        return "superseded"
    if meta.get(CM_DERIVED_DIRTY):
        return "dirty"
    if meta.get(CM_NEEDS_CHECK):
        return "needs-check"
    return "current"


def _depends_on(memory, ids: set[str]) -> bool:
    """True when the memory derives from any id in ``ids``.

    Pure dependency only — no dirty check. Callers that must skip stale
    views combine this with ``state_of(memory) != "dirty"``; erasure
    deliberately does not (a stale view must still be forgotten).
    """
    try:
        deps = set(get_cm(memory).get(CM_DERIVED_FROM) or [])
    except (TypeError, AttributeError):
        return False
    return bool(deps & ids)


def needs_rewrite(query: str) -> bool:
    return any(w in _PRONOUNS for w in _WORD.findall(query or ""))


def find_derived_dependent_ids(memories, erased_ids: set[str]) -> list[str]:
    return [str(m.id) for m in memories
            if _depends_on(m, erased_ids) and state_of(m) != "dirty"]


def decide_correction(cands, *, slot: Slot | None = None, assertion: str = "fact",
                      valid_from: str | None = None, memory_id: str | None = None,
                      evidence_ids=None) -> tuple[str, dict, list]:
    """Pure matching step: (status, meta, exact_matches). No DB, no writes.

    ``slot=None`` is a plain add. Everything here is dict/list work so the
    rules are testable without a store; ``resolve_correction`` applies the
    returned decision in one commit.
    """
    meta: dict = {CM_ASSERTION: assertion or "fact",
                  CM_SUBJECT: slot.subject if slot else "",
                  CM_ATTRIBUTE: slot.attribute if slot else "",
                  CM_SCOPE: (slot.scope if slot else "") or DEFAULT_SCOPE,
                  CM_EVIDENCE_IDS: [str(e) for e in (evidence_ids or [])]}
    if valid_from:
        try:
            datetime.fromisoformat(str(valid_from).replace("Z", "+00:00"))
            meta[CM_VALID_FROM] = str(valid_from)
        except ValueError:
            meta[CM_NEEDS_CHECK] = True
    if memory_id:
        meta.setdefault(CM_EVIDENCE_IDS, []).append(str(memory_id))

    status = "added"
    exact: list = []
    if slot is not None:
        exact = [m for m in cands if slot.matches(m)]
        if memory_id and exact and str(exact[0].id) != str(memory_id) and len(exact) == 1:
            pass  # explicit target mismatch handled below as ambiguous
        if exact and not meta.get(CM_NEEDS_CHECK):
            clash = any(get_cm(m).get(CM_ASSERTION, "fact") != meta[CM_ASSERTION] for m in exact)
            target_ok = (not memory_id) or any(str(m.id) == str(memory_id) for m in exact)
            if not clash and target_ok:
                status = "superseded"
            else:
                meta[CM_NEEDS_CHECK] = True
                status = "needs-check"
        elif not exact:
            same_subject_attr = [m for m in cands
                       if _stored_key(m)[:2] == (slot.subject, slot.attribute)
                       and _stored_key(m)[2] not in ("", slot.key[2])]
            if same_subject_attr and not slot.has_scope:
                meta[CM_NEEDS_CHECK] = True
                status = "needs-check"
            # different non-empty scope, or no candidates at all: independent add
    return status, meta, exact


async def resolve_correction(db, *, user_id, title, content, tags=None,
        source_type="mcp_agent", source_ref=None, slot: Slot | None = None,
        assertion="fact", valid_from=None,
        evidence_ids=None, memory_id=None, summary=None) -> dict:
    """Single creation path for add + correct. One commit, never raises.

    ``slot=None`` is a plain add (no identity, never supersedes).
    """
    rows = (await db.execute(
        select(Memory).where(Memory.user_id == user_id)
    )).scalars().all()
    cands = [m for m in rows if state_of(m) != "superseded"]

    status, meta, exact = decide_correction(
        cands, slot=slot, assertion=assertion, valid_from=valid_from,
        memory_id=str(memory_id) if memory_id else None,
        evidence_ids=evidence_ids)

    now = datetime.now(UTC)
    new = Memory(id=uuid4(), user_id=user_id, title=title, content=content,
                 tags=list(tags or []), source_type=source_type,
                 source_ref=source_ref, captured_at=now, extra_metadata=meta,
                 summary=summary)
    db.add(new)
    superseded, dirtied = [], []
    if status == "superseded":
        for m in exact:
            set_cm(m, {CM_SUPERSEDED_BY: str(new.id)})
            superseded.append(str(m.id))
        meta[CM_SUPERSEDES] = superseded[0] if len(superseded) == 1 else superseded
        new.extra_metadata = {**new.extra_metadata, CM_SUPERSEDES: meta[CM_SUPERSEDES]}
        erased = set(superseded)
        for m in cands:
            if m in exact:
                continue
            if _depends_on(m, erased) and state_of(m) != "dirty":
                set_cm(m, {CM_DERIVED_DIRTY: True})
                dirtied.append(str(m.id))
    await db.commit()
    return {"status": status, "memory": new,
            "superseded": superseded, "dirtied": dirtied}


async def collect_derived_ids(db, user_id, erased_ids: list) -> list:
    """Return ids of memories deriving from erased ids. Never raises."""
    try:
        erased = {str(e) for e in erased_ids}
        rows = (await db.execute(
            select(Memory).where(Memory.user_id == user_id)
        )).scalars().all()
        return [m.id for m in rows
                if str(m.id) not in erased and _depends_on(m, erased)]
    except Exception:
        return []
