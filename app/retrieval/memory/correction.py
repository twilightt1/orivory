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


from datetime import UTC, datetime
from sqlalchemy import select

async def resolve_correction(db, *, user_id, title, content, tags=None,
        source_type="mcp_agent", source_ref=None, subject="", attribute="",
        scope=DEFAULT_SCOPE, assertion="fact", valid_from=None,
        evidence_ids=None, memory_id=None) -> dict:
    """Single creation path for add + correct. One commit, never raises."""
    from uuid import uuid4
    from app.models.memory import Memory

    subj, attr, sc = normalize_slot(subject), normalize_slot(attribute), normalize_slot(scope)
    rows = (await db.execute(
        select(Memory).where(Memory.user_id == user_id)
    )).scalars().all()
    cands = [m for m in rows if not get_cm(m).get(CM_SUPERSEDED_BY)]

    def _triple(m) -> tuple[str, str, str]:
        meta = get_cm(m)
        return (normalize_slot(meta.get(CM_SUBJECT)),
                normalize_slot(meta.get(CM_ATTRIBUTE)),
                normalize_slot(meta.get(CM_SCOPE, DEFAULT_SCOPE)))

    now = datetime.now(UTC)
    meta: dict = {CM_ASSERTION: assertion or "fact",
                  CM_SUBJECT: subj, CM_ATTRIBUTE: attr,
                  CM_SCOPE: sc or DEFAULT_SCOPE,
                  CM_EVIDENCE_IDS: [str(e) for e in (evidence_ids or [])]}
    if valid_from:
        try:
            datetime.fromisoformat(str(valid_from).replace("Z", "+00:00"))
            meta[CM_VALID_FROM] = str(valid_from)
        except ValueError:
            meta[CM_NEEDS_CHECK] = True
    if memory_id:
        meta.setdefault(CM_EVIDENCE_IDS, []).append(str(memory_id))

    status, superseded, dirtied = "added", [], []
    exact: list = []
    if subj and attr:
        exact = [m for m in cands if _triple(m) == (subj, attr, sc or DEFAULT_SCOPE)]
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
            same_sa = [m for m in cands
                       if _triple(m)[:2] == (subj, attr) and _triple(m)[2] not in ("", sc or DEFAULT_SCOPE)]
            if same_sa and not sc:
                meta[CM_NEEDS_CHECK] = True
                status = "needs-check"
            # different non-empty scope, or no candidates at all: independent add

    new = Memory(id=uuid4(), user_id=user_id, title=title, content=content,
                 tags=list(tags or []), source_type=source_type,
                 source_ref=source_ref, captured_at=now, extra_metadata=meta)
    db.add(new)
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
            try:
                deps = set(get_cm(m).get(CM_DERIVED_FROM) or [])
            except (TypeError, AttributeError):
                continue
            if deps & erased and not get_cm(m).get(CM_DERIVED_DIRTY):
                set_cm(m, {CM_DERIVED_DIRTY: True})
                dirtied.append(str(m.id))
    await db.commit()
    return {"status": status, "memory": new,
            "superseded": superseded, "dirtied": dirtied}


async def collect_derived_ids(db, user_id, erased_ids: list) -> list:
    """Return ids of memories deriving from erased ids. Never raises."""
    from app.models.memory import Memory
    try:
        erased = {str(e) for e in erased_ids}
        rows = (await db.execute(
            select(Memory).where(Memory.user_id == user_id)
        )).scalars().all()
        return [m.id for m in rows
                if str(m.id) not in erased
                and set(get_cm(m).get(CM_DERIVED_FROM) or []) & erased]
    except Exception:
        return []
