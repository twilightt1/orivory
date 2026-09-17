"""P4b — the consolidation producer: derived summaries with provenance (§8.1).

The producer ``cm_derived_from`` was waiting for. P4a shipped READERS of the
marker only, and ruling R40 held the switch-on until T1's cascade fix landed —
``app/retrieval/memory/correction.py`` now walks the derived closure completely,
so a summary this module publishes is also a row the erasure walk can find.

One rule, ``RULE_VERSION``: a user's own SERVABLE memories are grouped by tag
and each group of ``MIN_SOURCES`` or more is summarized through the shared LLM
seam into a derived memory carrying its whole provenance:

    cm_derived_from      the source ids this view is a summary OF
    cm_source_revisions  each source's revision at the moment of the summary
    cm_rule_version      the rule generation that produced it
    cm_derived_key       sha256(source set | revisions | rule) — the dedupe key
    cm_assertion         ``derived``: never served as a raw fact (§8.2)

A summary is a VIEW, so it is never allowed to outrank its sources: the rule
does not read other summaries (rule v1 is non-recursive), the row is labeled
``derived`` everywhere provenance is reported, and its evidence is the source
rows themselves.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from typing import NamedTuple
from uuid import UUID, uuid4

from sqlalchemy import distinct, select

from app.agents.llm_client import complete as _complete
from app.models.memory import Memory
from app.retrieval.memory.correction import (
    CM_ASSERTION,
    CM_DERIVED_DIRTY,
    CM_DERIVED_FROM,
    CM_DERIVED_KEY,
    CM_EVIDENCE_IDS,
    CM_RULE_VERSION,
    CM_SOURCE_REVISIONS,
    _dependency_closure,
    get_cm,
    normalize_slot,
    set_cm,
    state_of,
)
from app.retrieval.memory.namespaces import PERSONAL, personal_namespace
from app.retrieval.memory.outbox import bump_revision, enqueue_upsert
from app.retrieval.memory.visibility import (
    current_memory_predicate,
    namespace_predicate,
    suppressed_source_predicate,
)

log = logging.getLogger(__name__)

# The rule generation. Bump it when the grouping or the prompt changes: it is
# part of the dedupe key, so a bump re-summarizes instead of skipping.
RULE_VERSION = "tag-summary.v1"

# The smallest group a summary is worth: one memory is its own evidence.
MIN_SOURCES = 2

# How many users one drain pass offers the producer (its own bound: the budget
# below is per user per pass).
CONSOLIDATION_USERS_PER_PASS = 10

_PROMPT = """Summarize these notes as one compact, factual memory. Keep names, \
numbers and dates exactly as written. If the notes disagree, say so instead of \
picking a winner. Return only the summary text.

## Theme
{tag}

## Notes
{sources}"""


class ConsolidationReport(NamedTuple):
    """What one run did — counted, so the drain log carries it (never a raise).

    ``published`` holds the ids of the rows this run created; ``skipped`` the
    groups an already-published key answered; ``refused`` the attempts the
    publish-time guard stood down; ``dirtied`` the stale views those stand-downs
    marked; ``errors`` the groups whose summary never arrived; ``deferred`` the
    groups a spent budget left for a later run; ``truncated`` a dirty-
    propagation walk that hit the closure cap (an incomplete walk, reported —
    next run re-walks and previously dirtied rows are already out of the walk).
    """

    published: list[str]
    skipped: int
    refused: int
    dirtied: int
    errors: int
    deferred: int
    truncated: bool


def _dedupe_key(source_ids: list[str], revisions: dict[str, int]) -> str:
    """The plan's idempotence key: sorted sources + their revisions + the rule.

    Same sources at the same revisions under the same rule ⇒ the same key, so a
    re-run recognizes its own output instead of publishing a second copy.
    """
    ids = sorted(source_ids)
    payload = (",".join(ids) + "|"
               + ",".join(f"{sid}:{int(revisions[sid])}" for sid in ids)
               + "|" + RULE_VERSION)
    return hashlib.sha256(payload.encode()).hexdigest()


def _tag_groups(rows) -> list[tuple[str, list[Memory]]]:
    """The rule (v1): one group per tag, deterministic order, ≥ ``MIN_SOURCES``.

    Determinism is not decoration: the budget walks these groups in order, so
    the same store state must propose the same work in the same order.
    """
    groups: dict[str, list[Memory]] = {}
    for row in rows:
        for tag in sorted({normalize_slot(t) for t in (row.tags or [])} - {""}):
            groups.setdefault(tag, []).append(row)
    return [(tag, sorted(members, key=lambda m: str(m.id)))
            for tag, members in sorted(groups.items())
            if len(members) >= MIN_SOURCES]


async def _summarize(tag: str, members) -> str | None:
    """One LLM call through the shared seam; ``None`` when no usable text came back."""
    sources = "\n".join(f"- {m.title}: {m.content}" for m in members)
    try:
        response = await _complete(
            agent="consolidation",
            messages=[{"role": "user", "content": _PROMPT.format(tag=tag, sources=sources)}],
        )
    except Exception as exc:  # an outage is a skipped group, never a failed run
        log.warning("consolidation summary failed for tag %r: %s", tag, exc)
        return None
    choices = getattr(response, "choices", None) or []
    text = (getattr(choices[0].message, "content", None) or "").strip() if choices else ""
    return text or None


def _publish(db, user_id, tag: str, text: str, key: str, revisions: dict):
    """Build the derived row with its provenance and its own index intent."""
    ids = sorted(revisions)
    now = datetime.now(UTC)
    derived = Memory(
        id=uuid4(), user_id=user_id, title=f"Summary: {tag}", content=text,
        tags=[tag], source_type="consolidation", captured_at=now,
        extra_metadata={
            CM_ASSERTION: "derived",
            CM_DERIVED_FROM: ids,
            CM_SOURCE_REVISIONS: {sid: int(revisions[sid]) for sid in ids},
            CM_RULE_VERSION: RULE_VERSION,
            CM_DERIVED_KEY: key,
            CM_EVIDENCE_IDS: ids,
        },
    )
    db.add(derived)
    bump_revision(derived)
    return derived


async def users_with_servable_memories(
        db, *, limit: int = CONSOLIDATION_USERS_PER_PASS) -> list[UUID]:
    """The producer's candidate queue: users with at least one servable memory.

    Personal namespace only — the producer is personal-only (P4a), and the
    predicate is built from ``namespaces`` like every other boundary check.
    """
    return list((await db.execute(
        select(distinct(Memory.user_id))
        .where(namespace_predicate(PERSONAL), current_memory_predicate())
        .order_by(Memory.user_id)
        .limit(max(1, int(limit)))
    )).scalars().all())


async def _moved_source_ids(db, user_id, revisions: dict[str, int]) -> list[UUID]:
    """Source ids that moved, vanished or stopped being servable (spec §8.1).

    The publish-time half of the guard. The summary was generated from the
    snapshot in ``revisions``, and between that snapshot and this read anything
    may have happened to a source — a correction (supersede), a soft forget
    (invalidated), an edit (bumped revision), an erasure (row gone: the
    strongest form, the evidence no longer exists). Any of those makes the
    summary a claim about evidence that is no longer what it says it is, so the
    publish stands down. A row that left the namespace reads as gone here too,
    and refusing is the safe direction: the summary may not claim evidence the
    boundary can no longer show.

    The read carries ``populate_existing`` because these ids are already in this
    session's identity map — a plain re-SELECT answers with the loaded
    attributes, i.e. exactly the snapshot this guard is checking, and the guard
    would be checking nothing. Refreshing from the DB is what makes it a read.
    """
    if not revisions:
        return []
    rows = (await db.execute(
        select(Memory)
        .where(Memory.id.in_([UUID(sid) for sid in revisions]),
               namespace_predicate(personal_namespace(user_id)))
        .execution_options(populate_existing=True)
    )).scalars().all()
    live = {str(row.id): row for row in rows}
    return [UUID(sid) for sid, expected in revisions.items()
            if (row := live.get(sid)) is None
            or state_of(row) not in ("current", "needs-check")
            or int(row.revision or 0) != int(expected)]


async def run_consolidation(db, user_id, budget: int = 10) -> ConsolidationReport:
    """Summarize this user's tagged servable memories, at most ``budget`` groups.

    The guards are the whole design (spec §8.1):

    - **idempotence** — a group whose dedupe key is already published (a
      servable derived row carries it) is skipped before any LLM work, so a
      re-run never publishes a second copy of the same view;
    - **stale / evidence guard** — the sources are re-read right before the
      publish (:func:`_moved_source_ids`); anything moved, gone or no longer
      servable stands the publish down and marks the views that depend on it
      ``dirty`` (the closure walk is the T1 one: transitive, cycle-defended,
      and a truncated walk is reported, never silently partial — the rows it
      did mark are out of the next walk's servable set, so it makes progress);
    - **budget** — soft and per run: at most ``budget`` groups are attempted,
      so at most that many memories are published. A spent budget defers the
      rest to the next pass instead of dropping them;
    - **non-CAS publish, deliberately**: a summary is a view, not a slot fact —
      there is no candidate row for T2's CAS to guard, and the re-read above is
      what protects the claim. A source that lands between that read and the
      commit leaves a view whose revisions are behind the store: the next run
      computes a different key, re-publishes, and the moved-source walk marks
      the stale view dirty — the residual window is one pass wide, on a view
      labeled ``derived``, never on a raw fact.

    Never raises out of a group: an LLM outage is an ``errors`` count and the
    next run retries. One commit per publish, so a later failure cannot take an
    earlier summary (or its index intent) down with it.
    """
    rows = (await db.execute(
        select(Memory).where(
            Memory.user_id == user_id,
            namespace_predicate(personal_namespace(user_id)),
            current_memory_predicate(),
            ~suppressed_source_predicate(user_id),
        )
    )).scalars().all()

    published: list[str] = []
    skipped = refused = dirtied = errors = deferred = 0
    truncated = False
    cap = max(0, int(budget))
    attempts = 0
    keys = {get_cm(row).get(CM_DERIVED_KEY) for row in rows}
    for tag, members in _tag_groups([row for row in rows
                                     if get_cm(row).get(CM_DERIVED_FROM) is None]):
        revisions = {str(m.id): int(m.revision or 1) for m in members}
        key = _dedupe_key(list(revisions), revisions)
        if key in keys:
            skipped += 1
            continue
        if attempts >= cap:
            deferred += 1
            continue
        attempts += 1
        text = await _summarize(tag, members)
        if text is None:
            errors += 1
            continue
        moved = await _moved_source_ids(db, user_id, revisions)
        if moved:
            closure = _dependency_closure(rows, moved)
            moved_set = set(moved)
            stale = [row for row in rows
                     if row.id in closure.visited and row.id not in moved_set
                     and state_of(row) not in ("dirty", "invalidated", "superseded")]
            for row in stale:
                set_cm(row, {CM_DERIVED_DIRTY: True})
            if stale:
                await db.commit()
            refused += 1
            dirtied += len(stale)
            truncated = truncated or closure.truncated
            continue
        derived = _publish(db, user_id, tag, text, key, revisions)
        await enqueue_upsert(db, derived)
        await db.commit()
        keys.add(key)
        published.append(str(derived.id))
    return ConsolidationReport(published, skipped, refused, dirtied, errors,
                               deferred, truncated)
