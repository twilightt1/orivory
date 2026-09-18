"""cm_* metadata helpers + pure rules for correctable memory (spec 2026-09-12)."""
from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import NamedTuple
from uuid import UUID, uuid4

from sqlalchemy import select, update

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
# Provenance of a derived summary (P4b/T5): the dedupe key over its source set
# + revisions + rule version, the source revisions themselves, and the rule
# generation that produced it (spec §8.1 — a summary keeps the generation of
# the rule/model that made it, and never claims to be a raw fact).
CM_DERIVED_KEY = "cm_derived_key"
CM_SOURCE_REVISIONS = "cm_source_revisions"
CM_RULE_VERSION = "cm_rule_version"
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


def client_metadata(raw: dict | None) -> dict:
    """Client-supplied metadata with the server-owned ``cm_*`` keys dropped.

    The ``cm_*`` vocabulary IS the lifecycle: a client update that could clear
    ``cm_invalidated`` (or forge ``cm_superseded_by``/``cm_derived_dirty``)
    would un-forget a row or fake history. Every CLIENT boundary (REST
    create/update, import) passes its metadata through here; the server's own
    writers (correction, forget, consolidation, retention) write ``cm_*``
    directly and never route through this.
    """
    return {key: value for key, value in (raw or {}).items()
            if not (isinstance(key, str) and key.startswith("cm_"))}


def state_of(memory) -> str:
    """One question for a memory's lifecycle state.

    Precedence: invalidated > superseded > dirty > needs-check > current.
    Invalidation preserves provenance but forbids serving. A superseded
    memory stays "superseded" even if also dirty; dirty (stale derived
    view) outranks needs-check because it must not be served either way.

    PRESENCE, not truthiness: the SQL predicates read these markers with
    ``_has`` (any non-NULL value counts), so a stored falsy value
    (``{"cm_invalidated": false}`` — client metadata can say anything) must
    not make Python and SQL disagree about the same row.
    """
    meta = get_cm(memory)
    if meta.get(CM_INVALIDATED) is not None:
        return "invalidated"
    if meta.get(CM_SUPERSEDED_BY) is not None:
        return "superseded"
    if meta.get(CM_DERIVED_DIRTY) is not None:
        return "dirty"
    if meta.get(CM_NEEDS_CHECK) is not None:
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


def _valid_from_dt(value) -> datetime | None:
    """``cm_valid_from`` as an aware datetime, or None when it is unusable.

    A naive stamp is read as UTC so it can still be compared with an
    offset-aware one instead of raising.
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _late_import(meta: dict, exact) -> bool:
    """True when the incoming fact's event time predates an exact candidate's.

    An old record imported after a newer fact is already stored must not
    supersede it by ingestion order — that un-learns the newer fact. Missing or
    unparseable stamps never raise the flag: there is nothing to compare, and
    that is the legacy shape (rows written before ``cm_valid_from`` existed).
    """
    incoming = _valid_from_dt(meta.get(CM_VALID_FROM))
    if incoming is None:
        return False
    return any((stamp := _valid_from_dt(get_cm(m).get(CM_VALID_FROM))) is not None
               and stamp > incoming for m in exact)


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
        if _valid_from_dt(valid_from) is None:
            meta[CM_NEEDS_CHECK] = True
        else:
            meta[CM_VALID_FROM] = str(valid_from)
    if memory_id:
        meta.setdefault(CM_EVIDENCE_IDS, []).append(str(memory_id))

    status = "added"
    exact: list = []
    if slot is not None:
        exact = [m for m in cands if slot.matches(m)]
        if exact and not meta.get(CM_NEEDS_CHECK):
            clash = any(get_cm(m).get(CM_ASSERTION, "fact") != meta[CM_ASSERTION] for m in exact)
            target_ok = (not memory_id) or any(str(m.id) == str(memory_id) for m in exact)
            if not clash and target_ok and not _late_import(meta, exact):
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


async def _cas_supersede(db, memory, successor_id: str, expected: int | None) -> bool:
    """Point one candidate at its successor iff it is still supersedable.

    BOTH cells that can move under the caller ride in the WHERE, so the CAS is
    the whole comparison: the row's ``revision`` is still ``expected``, and
    ``cm_superseded_by`` is still NULL (nobody claimed the row yet). The second
    cell is not optional — a supersede deliberately does NOT bump the revision
    (a bump without an outbox intent would read as stale to the drain), so a
    revision-only guard sees a rival's supersede as "unchanged" and overwrites
    a pointer that already names a successor: two current facts on the slot.
    ``rowcount == 1`` IS the swap answer — no row lock, and no read-then-write
    gap between the compared cells and the write. ``expected is None`` means the
    caller never named this candidate: it is refused rather than moved behind
    the caller's back.

    ``synchronize_session="fetch"`` because the pointer predicate is a JSON
    expression the in-Python evaluator refuses (``evaluate`` raises
    ``InvalidRequestError``); fetch keeps the in-session row — and the
    session's savepoint bookkeeping — in step with the write (on SQLite /
    Postgres the UPDATE answers with ``RETURNING``, so still one statement;
    only backends without RETURNING pay an extra SELECT).

    Ceiling: those two cells are all that is compared. The value written is the
    metadata THIS session holds (``{**get_cm(memory), ...}``), so a concurrent
    out-of-band edit to another ``cm_*`` key of the same row is overwritten,
    not detected.
    """
    # Local import: ``visibility`` imports this module's cm_* markers (cycle).
    # ``_has`` is the ONE spelling of "this marker is present".
    from app.retrieval.memory.visibility import _has

    if expected is None:
        return False
    result = await db.execute(
        update(Memory)
        .where(Memory.id == memory.id, Memory.revision == int(expected),
               ~_has(CM_SUPERSEDED_BY))
        .values(extra_metadata={**get_cm(memory), CM_SUPERSEDED_BY: successor_id})
        .execution_options(synchronize_session="fetch")
    )
    return result.rowcount == 1


async def resolve_correction(db, *, user_id, title, content, tags=None,
        source_type="mcp_agent", source_ref=None, slot: Slot | None = None,
        assertion="fact", valid_from=None,
        evidence_ids=None, memory_id=None, summary=None,
        expected_revisions: dict[UUID | str, int] | None = None) -> dict:
    """Single creation path for add + correct. One commit; incomplete closure raises.

    ``slot=None`` is a plain add (no identity, never supersedes).

    ``expected_revisions`` is the caller's snapshot of the slot it decided on
    (candidate id -> the revision it read). Keys are normalized to UUID, so a
    caller that stringified its ids names the same candidates (an un-normalized
    key reads as "never named" — a conflict with nothing to diagnose). The
    supersede/dirty apply is then a CAS: every exact candidate must be named in
    the snapshot, still carry that revision, AND still have no successor at the
    guarded UPDATE — otherwise the whole correction stands down as
    ``conflict``: the candidate writes that already landed are rolled back to
    the savepoint the apply opened (BEGIN NESTED snapshots the flushed state —
    the explicit flush here pins that boundary instead of relying on the
    implicit one — so the caller's other work sits outside the savepoint and is
    untouched), the new row
    lands beside the slot flagged ``cm_needs_check``, and no candidate keeps a
    pointer (two writers on one slot never leave two current facts by
    accident). The rows it held come back expired: a ``conflict`` means re-read,
    not the stale snapshot — on an async session a sync attribute read of an
    expired row raises ``MissingGreenlet``, so ``await db.refresh(row)`` (or a
    re-query) before touching it. ``None`` keeps the pre-CAS behaviour for
    callers that do not take part.

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

    if expected_revisions:
        normalized: dict[UUID | str, int] = {}
        for key, value in expected_revisions.items():
            try:
                normalized[UUID(str(key))] = int(value)
            except ValueError as exc:
                raise ValueError(
                    f"expected_revisions key {key!r} is not a UUID — a malformed "
                    "key would read as an un-named candidate (silent conflict)"
                ) from exc
        expected_revisions = normalized

    rows = (await db.execute(
        select(Memory).where(Memory.user_id == user_id,
                             namespace_predicate(personal_namespace(user_id)))
    )).scalars().all()
    cands = [m for m in rows if state_of(m) not in ("superseded", "invalidated")]

    status, meta, exact = decide_correction(
        cands, slot=slot, assertion=assertion, valid_from=valid_from,
        memory_id=str(memory_id) if memory_id else None,
        evidence_ids=evidence_ids)

    if expected_revisions is not None and memory_id is not None:
        # The caller named a target and carried its snapshot (the MCP boundary
        # does both). If that row is no longer an eligible candidate — a
        # concurrent forget/supersede took it out of ``cands``, or it was
        # deleted — the snapshot can never match at the CAS, and without this
        # guard the decision would degrade to a silent "added" that publishes a
        # new fact over a target the caller believed was still current.
        # Fail closed instead: conflict, needs-check, nothing superseded.
        named = UUID(str(memory_id))
        if all(m.id != named for m in cands):
            status = "conflict"
            meta[CM_NEEDS_CHECK] = True

    closure = _dependency_closure(rows, [m.id for m in exact]) if status == "superseded" else None
    if closure is not None and closure.truncated:
        raise DerivedClosureError("dependency closure truncated; refusing partial correction")

    now = datetime.now(UTC)
    new = Memory(id=uuid4(), user_id=user_id, title=title, content=content,
                 tags=list(tags or []), source_type=source_type,
                 source_ref=source_ref, captured_at=now, extra_metadata=meta,
                 summary=summary)
    superseded, dirtied = [], []
    if status == "superseded" and expected_revisions is not None:
        # CAS path. Apply BEFORE the new row joins the session, inside a
        # savepoint of its own: a refused candidate stands down the candidate
        # writes and nothing else. Flush first, so caller work in flight sits
        # OUTSIDE that savepoint (and cannot be expunged as savepoint-new work).
        await db.flush()
        savepoint = await db.begin_nested()
        refused = False
        for m in exact:
            if not await _cas_supersede(db, m, str(new.id),
                                        expected_revisions.get(m.id)):
                refused = True
                break
        if refused:
            # A candidate moved after the caller read it (or was never named):
            # stand down whole — never a silent supersede on a stale snapshot.
            # ONLY the savepoint goes back: the caller's other work in this
            # session is not this correction's to discard.
            await savepoint.rollback()
            status, closure = "conflict", None
            meta[CM_NEEDS_CHECK] = True
        else:
            await savepoint.commit()
            superseded = [str(m.id) for m in exact]
    elif status == "superseded":
        # No snapshot: no CAS, so nothing can refuse — the in-session pointer
        # write IS the apply (one commit at the end is the whole transaction).
        for m in exact:
            set_cm(m, {CM_SUPERSEDED_BY: str(new.id)})
        superseded = [str(m.id) for m in exact]
    db.add(new)
    # The new fact is the only row whose vector payload changes (superseded /
    # dirtied rows only carry cm_* metadata, which never reaches the vector):
    # enqueue its durable intent in this same commit.
    bump_revision(new)
    await enqueue_upsert(db, new)
    if status == "superseded":
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
