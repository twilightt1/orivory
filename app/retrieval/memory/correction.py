"""cm_* metadata helpers + pure rules for correctable memory (spec 2026-09-12)."""
from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import NamedTuple
from uuid import UUID, uuid4

from sqlalchemy import select

from app.models.memory import Memory
from app.retrieval.memory.namespaces import namespace_of, personal_namespace
from app.retrieval.memory.outbox import bump_revision, enqueue_upsert

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
CM_INVALIDATED = "cm_invalidated"
MEMORY_STATES = ("current", "superseded", "dirty", "needs-check", "invalidated")

DEFAULT_SCOPE = "default"
_MAX_CLOSURE_IDS = 5000

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

    Precedence: invalidated > superseded > dirty > needs-check > current.
    Invalidation preserves provenance but forbids serving. A superseded
    memory stays "superseded" even if also dirty; dirty (stale derived
    view) outranks needs-check because it must not be served either way.
    """
    meta = get_cm(memory)
    if meta.get(CM_INVALIDATED):
        return "invalidated"
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
    """Single creation path for add + correct. One commit; incomplete closure raises.

    ``slot=None`` is a plain add (no identity, never supersedes).

    The candidate read carries the namespace boundary (P4a/T4): a supersede or
    a dirty-mark is a WRITE to the candidate rows, and the decision is made over
    the user's OWN namespace only — a corrected fact can never reach across into
    another namespace of the same account (P4a: one namespace, so this is the
    same set the pre-P4 read saw).
    """
    # Local import: ``visibility`` imports this module's cm_* markers, so a
    # module-level import here would be a cycle. The predicate itself stays the
    # ONE spelling.
    from app.retrieval.memory.visibility import namespace_predicate

    rows = (await db.execute(
        select(Memory).where(Memory.user_id == user_id,
                             namespace_predicate(personal_namespace(user_id)))
    )).scalars().all()
    cands = [m for m in rows if state_of(m) not in ("superseded", "invalidated")]

    status, meta, exact = decide_correction(
        cands, slot=slot, assertion=assertion, valid_from=valid_from,
        memory_id=str(memory_id) if memory_id else None,
        evidence_ids=evidence_ids)

    closure = _dependency_closure(rows, [m.id for m in exact]) if status == "superseded" else None
    if closure is not None and closure.truncated:
        raise DerivedClosureError("dependency closure truncated; refusing partial correction")

    now = datetime.now(UTC)
    new = Memory(id=uuid4(), user_id=user_id, title=title, content=content,
                 tags=list(tags or []), source_type=source_type,
                 source_ref=source_ref, captured_at=now, extra_metadata=meta,
                 summary=summary)
    db.add(new)
    # The new fact is the only row whose vector payload changes (superseded /
    # dirtied rows only carry cm_* metadata, which never reaches the vector):
    # enqueue its durable intent in this same commit.
    bump_revision(new)
    await enqueue_upsert(db, new)
    superseded, dirtied = [], []
    if status == "superseded":
        for m in exact:
            set_cm(m, {CM_SUPERSEDED_BY: str(new.id)})
            superseded.append(str(m.id))
        meta[CM_SUPERSEDES] = superseded[0] if len(superseded) == 1 else superseded
        new.extra_metadata = {**new.extra_metadata, CM_SUPERSEDES: meta[CM_SUPERSEDES]}
        for m in rows:
            if m in exact:
                continue
            if m.id in closure.visited and state_of(m) not in ("dirty", "invalidated", "superseded"):
                set_cm(m, {CM_DERIVED_DIRTY: True})
                dirtied.append(str(m.id))
    await db.commit()
    return {"status": status, "memory": new,
            "superseded": superseded, "dirtied": dirtied}


class ClosureResult(NamedTuple):
    """BFS ids including roots, completeness flag, and cycle-defense set.

    No writes/commits: callers must refuse truncated results before mutating.
    Roots must already be authorized by the caller; never accept client ids
    without its ownership check.
    """

    affected: list[UUID]
    truncated: bool
    visited: set[UUID]


def _dependency_closure(rows, root_ids, kinds=("parent", "derived")) -> ClosureResult:
    # ponytail: scan metadata once per BFS level; add an adjacency index only
    # when large/deep namespaces make this O(rows * depth) walk a bottleneck.
    affected = list(dict.fromkeys(root_ids))
    visited = set(affected)
    frontier = set(affected)
    while frontier:
        if len(affected) > _MAX_CLOSURE_IDS:
            return ClosureResult(affected, True, visited)
        next_frontier = set()
        ids = {str(mid) for mid in frontier}
        for row in rows:
            if row.id in visited:
                continue
            if (("parent" in kinds and row.parent_id in frontier)
                    or ("derived" in kinds and _depends_on(row, ids))):
                visited.add(row.id)
                affected.append(row.id)
                next_frontier.add(row.id)
        frontier = next_frontier
    return ClosureResult(affected, False, visited)


async def collect_dependency_closure(db, root_ids, kinds=("parent", "derived")) -> ClosureResult:
    """Parent + derived transitive closure within one authorized personal scope.

    Infer scope from caller-authorized roots; refuse missing/mixed/other-
    namespace roots. Every expansion uses the same namespace predicate.
    Database errors propagate, never masquerading as an empty closure.
    """
    from app.retrieval.memory.visibility import namespace_predicate

    if not set(kinds) <= {"parent", "derived"}:
        raise ValueError("unknown dependency kind")
    roots = list(dict.fromkeys(UUID(str(mid)) for mid in root_ids))
    if not roots:
        return ClosureResult([], False, set())
    owner = None
    for mid in roots:
        row = await db.get(Memory, mid)
        if (row is None or namespace_of(row) != personal_namespace(row.user_id)
                or (owner is not None and row.user_id != owner)):
            raise ValueError("dependency roots must share one owned personal namespace")
        owner = row.user_id
    rows = (await db.execute(select(Memory).where(
        Memory.user_id == owner, namespace_predicate(personal_namespace(owner))
    ))).scalars().all()
    return _dependency_closure(rows, roots, kinds)


class DerivedClosureError(RuntimeError):
    """The derived-memory closure could not be enumerated.

    Erasure must record ``unknown`` for it — returning ``[]`` on a query
    failure would let a receipt claim a complete closure it never saw.
    """


async def collect_derived_ids(db, user_id, erased_ids: list) -> list:
    """Return ids of memories deriving from erased ids.

    Scoped to the user's OWN namespace (P4a/T4): the closure is enumerated over
    the rows an erasure may touch — another namespace's rows are another
    boundary's problem (P4b: an erasure walks per namespace). Callers that must
    REPORT what this narrowing leaves behind use
    :func:`collect_derived_ids_outside_namespace` (the same rule, complement).

    Raises :class:`DerivedClosureError` when the closure cannot be read — a
    silent ``[]`` is an unfalsifiable claim of completeness.
    """
    # Local import: ``visibility`` imports this module's cm_* markers (cycle).
    from app.retrieval.memory.visibility import namespace_predicate

    try:
        erased = {str(e) for e in erased_ids}
        rows = (await db.execute(
            select(Memory).where(Memory.user_id == user_id,
                                 namespace_predicate(personal_namespace(user_id)))
        )).scalars().all()
        return [m.id for m in rows
                if str(m.id) not in erased and _depends_on(m, erased)]
    except Exception as exc:
        raise DerivedClosureError(f"derived-memory closure query failed: {exc}") from exc


async def collect_derived_ids_outside_namespace(db, user_id, erased_ids: list) -> list:
    """Ids deriving from ``erased_ids`` in the user's OTHER namespaces (I3).

    The complement of the rule above, derived from the same
    ``namespace_predicate`` — the erasure walk does not reach these rows, so a
    receipt that only read :func:`collect_derived_ids` would report a complete
    closure while a derivative survived. They are reported as a residual
    instead (never deleted: another boundary owns them).

    Raises :class:`DerivedClosureError` on a failed read, same as the sibling.
    """
    # Local import: ``visibility`` imports this module's cm_* markers (cycle).
    from app.retrieval.memory.visibility import namespace_predicate

    try:
        erased = {str(e) for e in erased_ids}
        rows = (await db.execute(
            select(Memory).where(Memory.user_id == user_id,
                                 ~namespace_predicate(personal_namespace(user_id)))
        )).scalars().all()
        return [m.id for m in rows
                if str(m.id) not in erased and _depends_on(m, erased)]
    except Exception as exc:
        raise DerivedClosureError(
            f"out-of-namespace derived closure query failed: {exc}") from exc
