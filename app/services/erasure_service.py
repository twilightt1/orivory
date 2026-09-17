"""Memory erasure with verifiable receipts — Open Memory Hub MVP item 5.

One call → one ``ErasureReceipt``, one closure transaction per target:
ownership check (foreign/missing ids recorded, never deleted) → the SAME-USER
transitive closure collected BEFORE any delete (``parent_id`` BFS with a
visited set and no silent depth cap, plus the derived-memory set; papers
§3.2) → row deletes + one durable delete intent per affected id + a
suppression row when a forgotten projection's source still exists, all in one
commit → best-effort ``safe_delete_from_index`` per affected id → adversarial
verification (papers §3.3): re-query Qdrant + re-count residual DB rows.
v0 verification = absence-checks of every derived artifact; KG re-inference
probing is a tracked follow-up.

Receipt ``status`` — precedence in this order:

- ``completed_with_errors``: at least one target raised, or its closure could
  not be enumerated inside ``_MAX_CLOSURE_IDS`` (recorded with
  ``truncated=True``; the erase is refused rather than half-applied). The
  remaining targets are still erased.
- ``completed_with_residual``: no errors, but verification found residual
  vectors or DB rows, or a target's derived-memory closure came back
  ``unknown`` (something may survive, so a clean completion is not claimed).
- ``completed_unverified``: no errors and no known residual, but the vector
  side was not positively verified (a target is ``pending`` or ``unknown``).
  Spec §5.4 / the P1 gate: ``completed`` must not be reported when the
  residual check failed or the traversal did not finish.
- ``completed``: every target erased AND the vector side positively verified
  (absence readback confirmed for every deleted target); see the additive
  ``verification`` field. A call that erased nothing carries no
  ``verification``/``index_pending`` rollup at all — it verified nothing.

Per target, ``vector_state`` distinguishes what was actually confirmed:

- ``residual``: the vector is still present in the backend.
- ``pending``: the purge failed, so a durable delete intent still owes it —
  ``drain_pending`` retries it.
- ``unknown``: verification could not be run (``None`` ≠ verified).
- ``verified``: purge landed and the backend confirmed absence.

Erasure is best-effort per target: one failing target never aborts the call or
the other targets. The receipt commit is the only unrecorded failure mode —
if it raises, the exception propagates and no receipt exists.

Ownership scope: the descendant BFS filters every frontier query to the
erasing user's OWN namespace (P4a/T5, R35) — a same-account row another
namespace owns is never collected, purged or verified by this walk. Children
the walk does not own (another user's, or this user's other namespaces) are
still removed by the DB-level ON DELETE CASCADE when the parent row goes —
counted as residuals before the delete, but never purgeable from this scope:
their vectors are not verifiable here. Root fix is parent ownership validation
at memory creation (``create_memory``), so cross-user parenting cannot arise in
the first place.

Three residual classes exist because the walk is scoped: rows another user
parents onto an erased id (``cross_user_children``, R29c), children of either
class the DB cascade removes with their parent
(``cascaded_out_of_namespace``, F1), and rows of the STORING user's other
namespaces deriving from an erased id (``derived_out_of_namespace``, I3). All
are counted BEFORE the delete and recorded in ``db_residual``, so a receipt
that left data behind can never read as a clean completion.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.document_memory import DOC_MEMORY_SOURCE_TYPE, suppress_source_async
from app.models.document import Document
from app.models.entity import Entity, MemoryEntity, Relation
from app.models.erasure_receipt import ErasureReceipt
from app.models.index_outbox import IndexOutbox
from app.models.memory import Memory
from app.models.source import MemorySource
from app.retrieval.memory.correction import (
    CM_INVALIDATED,
    DerivedClosureError,
    collect_dependency_closure,
    collect_derived_ids,
    collect_derived_ids_outside_namespace,
    set_cm,
)
from app.retrieval.memory.namespaces import namespace_of, personal_namespace
from app.retrieval.memory.outbox import (
    KIND_MEMORY,
    OPERATION_DELETE,
    bump_revision,
    enqueue_delete,
    enqueue_upsert,
)
from app.retrieval.memory.visibility import namespace_predicate, not_dirty_predicate
from app.retrieval.memory.write_back import safe_delete_from_index

log = logging.getLogger(__name__)

ERASURE_STATUS_COMPLETED = "completed"
ERASURE_STATUS_UNVERIFIED = "completed_unverified"
ERASURE_STATUS_RESIDUAL = "completed_with_residual"
ERASURE_STATUS_ERRORS = "completed_with_errors"
NOT_FOUND_OR_FOREIGN = "not_found_or_foreign"

# Per-target vector verification states (receipt detail).
VECTOR_STATE_VERIFIED = "verified"
VECTOR_STATE_PENDING = "pending"
VECTOR_STATE_UNKNOWN = "unknown"
VECTOR_STATE_RESIDUAL = "residual"
# Worst-first: the receipt's rollup ``verification`` reports the first state
# any deleted target carries.
_VECTOR_STATE_PRECEDENCE = (
    VECTOR_STATE_RESIDUAL,
    VECTOR_STATE_PENDING,
    VECTOR_STATE_UNKNOWN,
    VECTOR_STATE_VERIFIED,
)

_MAX_LIMIT = 200
# Hard safety bound on one closure: a bigger tree is refused (target `error`,
# `truncated=True`) instead of being reported as a complete erasure.
_MAX_CLOSURE_IDS = 5000


@dataclass
class _DescendantTraversal:
    """Result of the ownership-checked descendant BFS from one root."""

    ids: list[uuid.UUID] = field(default_factory=list)      # same-user descendants, BFS order
    revisions: dict = field(default_factory=dict)           # id -> revision at collection time
    depth: int = 0                                          # levels that yielded nodes
    truncated: bool = False                                 # closure exceeded _MAX_CLOSURE_IDS


async def _vector_present_ids(memory_ids: list[str]) -> set[str]:
    """Verification seam — re-query the Qdrant collection for residual ids.

    Monkeypatched in tests; the vector-store helper is imported lazily so a
    missing/failed vector-store import cannot break the DB erasure.
    """
    from app.retrieval.memory.vector_store import get_memory_ids_present

    return await get_memory_ids_present(memory_ids)


async def _verify_absent(memory_ids: list[uuid.UUID]) -> set[str] | None:
    """Return ids still present in the index, or ``None`` when it is down.

    ``None`` = verification unknown — recorded as ``vector_state="unknown"``
    (or ``"pending"`` when a purge also failed); it is never reported as
    ``verified``. Postgres remains the source of truth.
    """
    try:
        return await _vector_present_ids([str(m) for m in memory_ids])
    except Exception as exc:
        log.warning("Vector residual check failed: %s", exc, extra={"memory_ids": [str(m) for m in memory_ids]})
        return None


async def _db_residual_counts(
    db: AsyncSession, memory_ids: list[uuid.UUID], *, cross_user_children: int = 0,
    cascaded_out_of_namespace: int = 0, derived_out_of_namespace: int = 0,
) -> dict[str, int]:
    """Re-count cascade targets after deletion; anything > 0 is a residual.

    ``cross_user_children``, ``cascaded_out_of_namespace`` and
    ``derived_out_of_namespace`` are passed in because they can only be counted
    through the erasing scope's OWN view, BEFORE the delete — see
    :func:`_cross_user_cascade_count` and
    :func:`_cascaded_out_of_namespace_count`. The two counts below are
    deliberately UNPREDICATED by namespace: they are absence checks over
    everything the erase should have taken, so a row another namespace owns
    still counts as a residual (a predicate here would read a leak as clean).
    """
    children = (await db.execute(
        select(func.count(Memory.id)).where(Memory.parent_id.in_(memory_ids))
    )).scalar_one()
    entity_links = (await db.execute(
        select(func.count()).select_from(MemoryEntity).where(MemoryEntity.memory_id.in_(memory_ids))
    )).scalar_one()
    source_links = (await db.execute(
        select(func.count()).select_from(MemorySource).where(MemorySource.memory_id.in_(memory_ids))
    )).scalar_one()
    return {
        "children": int(children),
        "entity_links": int(entity_links),
        "source_links": int(source_links),
        "cross_user_children": int(cross_user_children),
        "cascaded_out_of_namespace": int(cascaded_out_of_namespace),
        "derived_out_of_namespace": int(derived_out_of_namespace),
    }


async def _cross_user_cascade_count(
    db: AsyncSession, memory_ids: list[uuid.UUID], *, user_id: uuid.UUID
) -> int:
    """Rows another user parents onto the erased ids (R29c).

    The DB-level ON DELETE CASCADE removes them, but the erasing user's scope
    cannot enumerate them — so their vectors are never deleted and never
    verified. Counting them BEFORE the delete is the only way the receipt can
    report that residual class instead of claiming a clean erasure.
    """
    return int((await db.execute(
        select(func.count(Memory.id)).where(
            Memory.parent_id.in_(memory_ids),
            Memory.user_id != user_id,
        )
    )).scalar_one())


async def _cascaded_out_of_namespace_count(
    db: AsyncSession, memory_ids: list[uuid.UUID], *, user_id: uuid.UUID
) -> int:
    """Count every cascade-only descendant, including below derived nodes (S5).

    The erasure union is unchanged (R36). Any row outside that union loses its
    SQL row but has no vector delete intent, even when it shares the namespace.
    Recursive UNION (not UNION ALL) terminates cycles and counts each id once.
    This is deliberately an unscoped count, never a serving/ownership read.
    """
    cascade = select(Memory.id).where(Memory.id.in_(memory_ids)).cte("cascade", recursive=True)
    cascade = cascade.union(select(Memory.id).join(cascade, Memory.parent_id == cascade.c.id))
    return int((await db.execute(
        select(func.count(Memory.id)).where(
            Memory.id.in_(select(cascade.c.id)),
            Memory.id.not_in(memory_ids),
        )
    )).scalar_one())


async def _pending_delete_intents(db: AsyncSession, memory_ids: list[uuid.UUID]) -> int:
    """Delete intents these ids still owe the drain (0 = index work committed)."""
    entity_ids = [uuid.UUID(str(m)).hex for m in memory_ids]
    return int((await db.execute(
        select(func.count()).select_from(IndexOutbox).where(
            IndexOutbox.kind == KIND_MEMORY,
            IndexOutbox.operation == OPERATION_DELETE,
            IndexOutbox.status == "pending",
            IndexOutbox.entity_id.in_(entity_ids),
        )
    )).scalar_one())


async def _delete_orphan_entities(db: AsyncSession, user_id: uuid.UUID) -> int:
    """Delete this user's entities left with no memory links, plus their relations.

    Part of the same closure transaction: after the memory rows (and their
    ``memory_entities`` links) are gone, an entity that nothing links to is a
    leftover topic the user erased. Relations touching it go too.
    """
    orphan_ids = list((await db.execute(
        select(Entity.id).where(
            Entity.user_id == user_id,
            ~select(MemoryEntity.id).where(MemoryEntity.entity_id == Entity.id).exists(),
        )
    )).scalars().all())
    if not orphan_ids:
        return 0
    await db.execute(delete(Relation).where(
        Relation.user_id == user_id,
        or_(Relation.source_entity_id.in_(orphan_ids),
            Relation.target_entity_id.in_(orphan_ids)),
    ))
    await db.execute(delete(Entity).where(Entity.id.in_(orphan_ids), Entity.user_id == user_id))
    return len(orphan_ids)


async def _suppress_forgotten_projection(db: AsyncSession, user_id: uuid.UUID, row: Memory) -> str | None:
    """Pin a forgotten document projection: re-ingest must not resurrect it.

    Only when the memory IS a document projection (``file_upload`` with a
    ``source_ref``) whose source document still exists — losing the original
    upload must not be recorded as a forget. The raw upload itself is never
    deleted (spec §12.3). Returns the suppressed ``source_ref``, or ``None``.
    Rides the caller's transaction.
    """
    if row.source_type != DOC_MEMORY_SOURCE_TYPE or not row.source_ref:
        return None
    try:
        document_id = uuid.UUID(str(row.source_ref))
    except (ValueError, TypeError, AttributeError):
        return None
    if await db.get(Document, document_id) is None:
        return None
    await suppress_source_async(db, user_id=user_id, source_ref=row.source_ref, reason="forgotten")
    return row.source_ref


def _owned(row: Memory, user_id: uuid.UUID) -> bool:
    """Is ``row`` this caller's own row AND in their namespace? (P4a/T5, R35)

    A primary-key read (``db.get``) cannot carry a predicate, so the loaded row
    is checked against the same ``namespaces`` value the SQL predicate is built
    from — the spelling ``_owned`` uses in ``app/api/v1/memories.py`` and
    ``app/mcp_hub/tools.py``. A row that predates the column reads as personal.
    """
    return row.user_id == user_id and namespace_of(row) == personal_namespace(user_id)


async def _collect_descendants(db: AsyncSession, user_id: uuid.UUID, root_id: uuid.UUID) -> _DescendantTraversal:
    """Full transitive descendant closure over ``parent_id`` — no silent cap.

    The frontier query carries ``Memory.user_id == user_id`` AND the namespace
    boundary, so descendants another user's or another namespace's rows are
    never collected, deleted, or disclosed by this service (P4a/T5, R35: the
    walk by id carries the same boundary as every other read).
    A visited set terminates a cycle; ``_MAX_CLOSURE_IDS`` is the only bound,
    and hitting it is reported (``truncated``) — never swallowed. Revisions
    ride along so every affected id can get its delete intent with the
    revision it actually had.
    """
    result = _DescendantTraversal()
    visited = {root_id}
    frontier = [root_id]
    while frontier:
        rows = (await db.execute(
            select(Memory.id, Memory.revision).where(
                Memory.parent_id.in_(frontier),
                Memory.user_id == user_id,  # cross-user descendants are never collected
                namespace_predicate(personal_namespace(user_id)),  # R35: nor other namespaces'
            )
        )).all()
        next_frontier: list[uuid.UUID] = []
        for child_id, revision in rows:
            if child_id in visited:
                continue  # cycle: already collected (or the root itself)
            visited.add(child_id)
            next_frontier.append(child_id)
            result.revisions[child_id] = int(revision or 1)
        if not next_frontier:
            break
        result.depth += 1
        result.ids.extend(next_frontier)
        if len(result.ids) > _MAX_CLOSURE_IDS:
            result.truncated = True
            break
        frontier = next_frontier
    return result


async def _erase_one(db: AsyncSession, user_id: uuid.UUID, memory_id: uuid.UUID) -> dict[str, Any]:
    """Erase one owned memory in ONE closure transaction; return its receipt entry.

    Enqueues a durable delete intent for every affected id in the same commit
    as the row deletes, so a crash before the vector purge is replayable by
    ``drain_pending``. The vector purge and its verification follow the commit
    and only describe the outcome (``vector_state``).
    """
    row = await db.get(Memory, memory_id)
    if row is None or not _owned(row, user_id):
        return {"memory_id": str(memory_id), "status": NOT_FOUND_OR_FOREIGN}

    traversal = await _collect_descendants(db, user_id, memory_id)
    if traversal.truncated:
        # A partial closure cannot be erased safely: refusing it is the only
        # honest option (never report a truncated walk as completed).
        log.error("Closure for memory %s exceeds _MAX_CLOSURE_IDS=%d; refusing",
                  memory_id, _MAX_CLOSURE_IDS)
        return {
            "memory_id": str(memory_id),
            "status": "error",
            "error": f"closure exceeds _MAX_CLOSURE_IDS={_MAX_CLOSURE_IDS}; refusing a partial erase",
            "truncated": True,
            "closure_size": len(traversal.ids),
            "vector_state": VECTOR_STATE_UNKNOWN,
            "vector_residual_checked": False,
            "db_residual": None,
        }

    entity_links = (await db.execute(
        select(func.count()).select_from(MemoryEntity).where(MemoryEntity.memory_id == memory_id)
    )).scalar_one()
    source_links = (await db.execute(
        select(func.count()).select_from(MemorySource).where(MemorySource.memory_id == memory_id)
    )).scalar_one()

    affected = [memory_id, *traversal.ids]
    revisions = {memory_id: int(row.revision or 1), **traversal.revisions}
    derived_ids: list[uuid.UUID] = []
    derived_closure = "complete"
    derived_outside = 0
    try:
        for derived_id in await collect_derived_ids(db, user_id, affected):
            if derived_id in affected or derived_id in derived_ids:
                continue
            derived_row = await db.get(Memory, derived_id)
            if derived_row is None or not _owned(derived_row, user_id):
                continue  # vanished or outside this walk's scope: not ours to erase
            derived_ids.append(derived_id)
            revisions[derived_id] = int(derived_row.revision or 1)
        # I3: the closure above walks ONE namespace. Derivatives the walk cannot
        # reach are REPORTED as a residual (never deleted — another boundary
        # owns them) instead of a receipt claiming a closure it never saw.
        derived_outside = len(await collect_derived_ids_outside_namespace(db, user_id, affected))
    except DerivedClosureError as exc:
        # Surface it as a typed failure so the receipt records unknown instead
        # of silently claiming a complete closure.
        derived_closure = "unknown"
        log.warning("Derived-dependent collection failed: %s", exc,
                    extra={"memory_id": str(memory_id)})

    affected.extend(derived_ids)
    for affected_id in affected:
        await enqueue_delete(db, entity_id=str(affected_id), tenant_id=str(user_id),
                             revision=revisions.get(affected_id, 1))

    suppressed_source = await _suppress_forgotten_projection(db, user_id, row)
    # Count the rows the DB cascade is about to remove for OTHER users (R29c) —
    # and for this user's OTHER namespaces (F1): after the delete they are gone
    # and their vectors were never enumerable from this scope — the receipt
    # must still carry them as a residual.
    cross_user_children = await _cross_user_cascade_count(db, affected, user_id=user_id)
    cascaded_out_of_namespace = await _cascaded_out_of_namespace_count(
        db, affected, user_id=user_id)

    # One DELETE for the whole closure: children and links go with it through
    # the DB-level ON DELETE CASCADE. The DB is also the only deleter that can
    # order a cyclic parent chain — the ORM unit of work raises
    # CircularDependencyError on mutually-parented rows.
    await db.execute(delete(Memory).where(Memory.id.in_(affected)))
    orphan_entities = await _delete_orphan_entities(db, user_id)
    await db.commit()

    vectors_deleted: list[str] = []
    purge_failed = False
    for vid in affected:
        if await safe_delete_from_index(vid) is not True:
            purge_failed = True
        vectors_deleted.append(str(vid))

    present = await _verify_absent(affected)
    db_residual = await _db_residual_counts(db, affected,
                                            cross_user_children=cross_user_children,
                                            cascaded_out_of_namespace=cascaded_out_of_namespace,
                                            derived_out_of_namespace=derived_outside)
    if present:
        vector_state = VECTOR_STATE_RESIDUAL
    elif purge_failed:
        # Nothing known bad, but the purge did not confirm: the durable intent
        # enqueued above is what will finish this.
        vector_state = VECTOR_STATE_PENDING
    elif present is None:
        vector_state = VECTOR_STATE_UNKNOWN
    else:
        vector_state = VECTOR_STATE_VERIFIED

    return {
        "memory_id": str(memory_id),
        "status": "deleted",
        "affected_memory_ids": [str(c) for c in traversal.ids],  # transitive, excluding the target
        "derived_memory_ids": [str(d) for d in derived_ids],
        "traversal_depth": traversal.depth,
        "truncated": traversal.truncated,
        "entity_links": int(entity_links),
        "source_links": int(source_links),
        "orphan_entities": orphan_entities,
        "suppressed_source": suppressed_source,
        "derived_closure": derived_closure,
        "vectors_deleted": vectors_deleted,
        "vector_state": vector_state,
        "vector_residual": sorted(present) if present is not None else [],
        "vector_residual_checked": present is not None,
        "db_residual": db_residual,
        "index_pending": await _pending_delete_intents(db, affected),
    }


async def erase_memories(
    db: AsyncSession,
    user_id: uuid.UUID,
    memory_ids: list[uuid.UUID],
    *,
    requested_by: str,
) -> ErasureReceipt:
    """Erase every owned memory among ``memory_ids`` and write one receipt.

    Best-effort per target: a target whose erasure raises is recorded as a
    per-target error entry (status ``error`` + message) and the remaining
    targets are still processed. The receipt commit is the only unrecorded
    failure mode — if it raises, the exception propagates and no receipt exists.
    """
    unique_ids = list(dict.fromkeys(memory_ids))
    targets: list[dict[str, Any]] = []
    for mid in unique_ids:
        try:
            targets.append(await _erase_one(db, user_id, mid))
        except Exception as exc:  # best-effort: record + continue
            log.exception("Erasure failed for memory %s", mid, extra={"memory_id": str(mid)})
            # A failed per-target commit poisons the session (PendingRollbackError);
            # roll back so the remaining targets (and the receipt commit) still work.
            await db.rollback()
            targets.append({
                "memory_id": str(mid),
                "status": "error",
                "error": str(exc),
                "vector_state": VECTOR_STATE_UNKNOWN,
                "vector_residual_checked": False,
                "db_residual": None,
            })
    erased = [t for t in targets if t["status"] == "deleted"]
    residual_vectors = sum(len(t["vector_residual"]) for t in erased)
    residual_rows = sum(sum(t["db_residual"].values()) for t in erased)
    any_errors = any(t["status"] == "error" for t in targets)
    any_closure_unknown = any(t.get("derived_closure") == "unknown" for t in erased)
    vector_states = {t["vector_state"] for t in erased}

    if any_errors:
        status = ERASURE_STATUS_ERRORS
    elif residual_vectors or residual_rows or any_closure_unknown:
        status = ERASURE_STATUS_RESIDUAL
    elif vector_states - {VECTOR_STATE_VERIFIED}:
        # Spec §5.4 / P1 gate: `completed` is only stored after a POSITIVE
        # presence readback. A pending purge or an unknown verification
        # (vector store unreachable) reports `completed_unverified` instead.
        status = ERASURE_STATUS_UNVERIFIED
    else:
        status = ERASURE_STATUS_COMPLETED

    detail: dict[str, Any] = {
        "requested_by": requested_by,
        "targets": targets,
        "summary": {
            "requested": len(unique_ids),
            "erased": len(erased),
            "skipped": len(unique_ids) - len(erased),
            "errors": sum(1 for t in targets if t["status"] == "error"),
            "residual_vectors": residual_vectors,
            "residual_rows": residual_rows,
        },
    }
    if erased:
        # Additive: what the index side still owes for this call, and the
        # worst vector state any deleted target could confirm. Omitted when
        # nothing was erased — a forget that deleted nothing verified nothing,
        # so it must not read as a positive verification.
        detail["verification"] = next(
            (s for s in _VECTOR_STATE_PRECEDENCE if s in vector_states),
            VECTOR_STATE_VERIFIED,
        )
        detail["index_pending"] = sum(int(t.get("index_pending", 0)) for t in erased)

    receipt = ErasureReceipt(
        user_id=user_id,
        requested_memory_ids=[str(m) for m in unique_ids],
        status=status,
        detail=detail,
    )
    db.add(receipt)
    await db.commit()
    await db.refresh(receipt)
    return receipt


async def list_receipts(
    db: AsyncSession,
    user_id: uuid.UUID,
    *,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[ErasureReceipt], int]:
    """List a user's receipts newest first, with the unpaginated total count.

    ``limit`` is clamped into [1, 200] and a negative ``offset`` to 0 — same
    contract as ``access_ledger_service.list_entries``.
    """
    total = (await db.execute(
        select(func.count(ErasureReceipt.id)).where(ErasureReceipt.user_id == user_id)
    )).scalar_one()
    rows = (await db.execute(
        select(ErasureReceipt)
        .where(ErasureReceipt.user_id == user_id)
        .order_by(ErasureReceipt.created_at.desc())
        .offset(max(offset, 0))
        .limit(max(1, min(limit, _MAX_LIMIT)))
    )).scalars().all()
    return list(rows), total


# ── soft forget (P4b/T3): invalidate + suppress, keep provenance ────────────

SOFT_FORGET_STATUS_INVALIDATED = "invalidated"


async def _serving_memory_ids(
    db: AsyncSession, memory_ids: list[uuid.UUID]
) -> set[uuid.UUID] | None:
    """Which of ``memory_ids`` a serving surface would still return.

    Verification seam (monkeypatched in tests, like ``_vector_present_ids``):
    the ONE rule every serving surface applies is ``not_dirty_predicate()``.
    ``None`` = the readback failed — an unknown that is never reported as
    ``completed`` (spec §5.4). Callers catch around the call and roll the
    poisoned transaction back before writing the receipt.
    """
    return set((await db.execute(
        select(Memory.id).where(Memory.id.in_(memory_ids), not_dirty_predicate())
    )).scalars().all())


async def _soft_forget_one(
    db: AsyncSession, user_id: uuid.UUID, memory_id: uuid.UUID
) -> dict[str, Any]:
    """Invalidate ONE owned memory and its closure; return its receipt entry.

    One commit per target, like the hard path: the invalidated rows, their
    suppression rows and their payload-refresh intents land together or not at
    all. A truncated closure is REFUSED before any write — a partial closure
    cannot be half-forgotten (the hard path refuses a partial erase the same
    way, so neither path can report a closure it never saw).
    """
    row = await db.get(Memory, memory_id)
    if row is None or not _owned(row, user_id):
        return {"memory_id": str(memory_id), "status": NOT_FOUND_OR_FOREIGN}

    closure = await collect_dependency_closure(db, [memory_id])
    if closure.truncated:
        log.error("Closure for memory %s is truncated; refusing the forget", memory_id,
                  extra={"memory_id": str(memory_id)})
        return {
            "memory_id": str(memory_id),
            "status": "error",
            "error": "closure exceeds _MAX_CLOSURE_IDS; refusing a partial forget",
            "truncated": True,
            "closure_size": len(closure.affected),
        }

    invalidated: list[uuid.UUID] = []
    suppressed: list[str] = []
    for affected_id in closure.affected:
        target = await db.get(Memory, affected_id)
        if target is None or not _owned(target, user_id):
            continue  # the closure is scoped by construction; belt and braces
        set_cm(target, {CM_INVALIDATED: True})
        # R37: the state must REACH the vector payload. The bump makes this a
        # real write and the durable upsert carries it to the applier, which
        # re-reads the row — so the payload's visibility_state reads
        # `invalidated`. No vector is purged here (reconciliation repair owns
        # that, R34).
        bump_revision(target)
        await enqueue_upsert(db, target)
        invalidated.append(affected_id)
        if target.source_ref:
            # Universal (R38): every affected source identity is pinned, not
            # only the root projection — including refs that are not
            # `file_upload` documents. content_hash stays NULL until an upload
            # computes one (never backfilled).
            await suppress_source_async(db, user_id=user_id, source_ref=target.source_ref,
                                        reason="forgotten", namespace=namespace_of(target))
            suppressed.append(target.source_ref)
    await db.commit()
    return {
        "memory_id": str(memory_id),
        "status": SOFT_FORGET_STATUS_INVALIDATED,
        "affected_memory_ids": [str(i) for i in closure.affected],
        "suppressed_sources": sorted(set(suppressed)),
        "payload_refresh_enqueued": len(invalidated),
    }


async def soft_forget(
    db: AsyncSession,
    user_id: uuid.UUID,
    memory_ids: list[uuid.UUID],
    *,
    requested_by: str,
) -> ErasureReceipt:
    """Forget memories softly: invalidate the closure, suppress every source (§12).

    The user-facing "forget" is soft and provenance-preserving:

    - the rows STAY (title/content/tags/provenance/evidence untouched) and are
      marked ``invalidated`` — so no serving surface returns them again;
    - every affected ``source_ref`` — the whole closure, not only the root
      projection — gets a ``MemorySuppression`` row, so a re-import cannot
      resurrect what was forgotten (R38; the guards that READ the ledger are
      Task 4);
    - every invalidated row gets ``bump_revision`` + a durable ``enqueue_upsert``
      in the same commit (R37): the point is not purged here, its payload is
      refreshed to ``visibility_state=invalidated`` by the drain;
    - the receipt reports ``completed`` ONLY after a serving-off readback
      (``not_dirty_predicate()`` says the rows left serving). Leftovers →
      ``completed_with_residual``; an unreadable check → ``completed_unverified``
      — never a faked ``completed``.

    Best-effort per target like the hard path: one failing target never aborts
    the others, and the receipt commit stays the only unrecorded failure mode.
    """
    unique_ids = list(dict.fromkeys(memory_ids))
    targets: list[dict[str, Any]] = []
    for mid in unique_ids:
        try:
            targets.append(await _soft_forget_one(db, user_id, mid))
        except Exception as exc:  # best-effort: record + continue
            log.exception("Soft forget failed for memory %s", mid, extra={"memory_id": str(mid)})
            await db.rollback()
            targets.append({"memory_id": str(mid), "status": "error", "error": str(exc)})

    invalidated = [t for t in targets if t["status"] == SOFT_FORGET_STATUS_INVALIDATED]
    affected = [uuid.UUID(str(value)) for t in invalidated
                for value in t.get("affected_memory_ids") or []]
    residual: set[uuid.UUID] | None = None
    if affected:
        try:
            residual = await _serving_memory_ids(db, affected)
        except Exception as exc:
            # An unreadable check is UNKNOWN, never a clean completion (§5.4).
            # The failed read leaves the session unusable (PendingRollbackError
            # on the next statement), so roll back before the receipt commit —
            # an unknown check must not take the receipt down with it.
            log.warning("Serving readback failed: %s", exc,
                        extra={"memory_ids": [str(m) for m in affected]})
            await db.rollback()
            residual = None
    residual_ids = sorted(str(m) for m in residual or set())

    any_errors = any(t["status"] == "error" for t in targets)
    if any_errors:
        status = ERASURE_STATUS_ERRORS
    elif residual is None:
        status = ERASURE_STATUS_UNVERIFIED
    elif residual_ids:
        status = ERASURE_STATUS_RESIDUAL
    else:
        status = ERASURE_STATUS_COMPLETED

    detail: dict[str, Any] = {
        "requested_by": requested_by,
        "mode": "soft",
        "targets": targets,
        "summary": {
            "requested": len(unique_ids),
            "invalidated": len(invalidated),
            "skipped": len(unique_ids) - len(invalidated),
            "errors": sum(1 for t in targets if t["status"] == "error"),
            "suppressed": sum(len(t.get("suppressed_sources") or []) for t in invalidated),
            "payload_refresh_enqueued": sum(int(t.get("payload_refresh_enqueued") or 0)
                                            for t in invalidated),
            "serving_residual": len(residual_ids),
        },
        "serving_residual": residual_ids,
    }
    if invalidated:
        # Additive, like the hard receipt's `verification`: what this call could
        # actually confirm. Omitted when nothing was invalidated — a forget
        # that changed nothing verified nothing.
        detail["serving_residual_checked"] = residual is not None

    receipt = ErasureReceipt(
        user_id=user_id,
        requested_memory_ids=[str(m) for m in unique_ids],
        status=status,
        detail=detail,
    )
    db.add(receipt)
    await db.commit()
    await db.refresh(receipt)
    return receipt


# ── reconciliation: the drain's progress revises open receipts (R16) ────────

# Receipt statuses reconcile may revise. Everything else is terminal: a
# verified receipt is never downgraded, and a residual/error verdict is never
# rewritten from a later, weaker read.
_OPEN_RECEIPT_STATUSES = ("pending", ERASURE_STATUS_UNVERIFIED)


def _receipt_ids(targets: list[dict[str, Any]]) -> list[uuid.UUID] | None:
    """The ids these receipt targets recorded, ``None`` when the detail is unusable.

    ``vectors_deleted`` is the per-target evidence the erase wrote (every id of
    the closure it purged), so reconciliation re-checks exactly what was erased.
    """
    ids: list[uuid.UUID] = []
    for target in targets:
        for value in target.get("vectors_deleted") or []:
            try:
                ids.append(uuid.UUID(str(value)))
            except (ValueError, TypeError, AttributeError):
                return None
    return ids


async def reconcile_erasure_receipts(*, limit: int = 50) -> dict[str, int]:
    """Re-verify open receipts after the drain lands their owed deletes (R16).

    Upgrade-only and bounded: only receipts still in ``pending`` /
    ``completed_unverified`` are scanned — OLDEST first (R20, ties broken by
    id: newest-first starves an old open receipt out of the window under
    sustained erase load) with ``limit`` clamped like ``list_receipts`` — and
    one is rewritten to ``completed`` ONLY when the re-read is clean: every
    affected vector absent AND no DB residual rows.

    A pass that DOES find residual vectors records what it observed in the
    detail (``vector_residual_checked`` / ``vector_residual``, R19) and leaves
    the status alone — never relabelled to the terminal
    ``completed_with_residual``, so the receipt stays open and re-checkable. A
    pass whose readback failed observed nothing and writes nothing. Either way
    reconcile never downgrades, never invents a status, and a terminal receipt
    is not even scanned. Returns ``{"checked", "upgraded", "still_unverified"}``.

    Runs on its own session: the drain loop and the admin endpoint call it
    without one (ruling R15 keeps it off the request-path barrier). Each
    receipt is committed as it goes (the drain's shape): a pass holds no
    transaction across the next receipt, so a concurrent writer is never left
    waiting on it and a failure on a later receipt cannot take the earlier
    upgrades — or the evidence they wrote — with it.
    """
    # Late import: the test fixtures' sessionmaker lives on the module.
    from app.database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        rows = (await db.execute(
            select(ErasureReceipt)
            .where(ErasureReceipt.status.in_(_OPEN_RECEIPT_STATUSES))
            .order_by(ErasureReceipt.created_at.asc(), ErasureReceipt.id.asc())
            .limit(max(1, min(limit, _MAX_LIMIT)))
        )).scalars().all()
        # Close the scan's transaction before the per-receipt work below: a
        # pass-long transaction is what a concurrent writer waits out (5s busy
        # timeout, then `database is locked`) and what loses every upgrade the
        # pass made when a later receipt raises (M3).
        await db.commit()
        upgraded = 0
        for receipt in rows:
            upgraded += await _upgrade_receipt(db, receipt)
            await db.commit()  # one commit per receipt (outbox._apply's shape)
    return {
        "checked": len(rows),
        "upgraded": upgraded,
        "still_unverified": len(rows) - upgraded,
    }


def _residual_evidence(targets: list[dict[str, Any]], present: set[str]) -> list[dict[str, Any]]:
    """R19: what THIS pass observed, written into the detail of an open receipt."""
    evidence: list[dict[str, Any]] = []
    for target in targets:
        ids = [str(value) for value in target.get("vectors_deleted") or []]
        if target.get("status") != "deleted" or not ids:
            evidence.append(target)  # nothing was deleted here: nothing observed
            continue
        evidence.append({
            **target,
            "vector_residual_checked": True,
            "vector_residual": sorted(v for v in ids if v in present),
        })
    return evidence


async def _upgrade_receipt(db: AsyncSession, receipt: ErasureReceipt) -> bool:
    """Rewrite ONE open receipt to ``completed`` when the re-read is clean.

    Clean means: every affected vector is absent AND every residual count is
    zero. Anything else (a residual still present, an unreachable store,
    leftover rows) returns ``False`` — the receipt keeps the status it has:
    never a downgrade, and never a completion on a failed read. R19: a pass
    that OBSERVED residuals writes that observation into the detail and leaves
    the status alone (the row stays open and re-checkable); a pass whose
    readback failed observed nothing, so the row stays byte-identical.
    """
    targets = list((receipt.detail or {}).get("targets") or [])
    affected = _receipt_ids(targets)
    if not affected:
        return False  # nothing recorded to verify: never guess what was erased
    present = await _verify_absent(affected)  # None = the store did not answer
    if present is None:
        return False  # unreadable store: no observation, so no evidence to write
    if present:
        receipt.detail = {
            **(receipt.detail or {}),
            "targets": _residual_evidence(targets, present),
        }
        return False  # evidence, not a verdict: the receipt stays open (R19)
    rechecked: list[dict[str, Any]] = []
    for target in targets:
        ids = _receipt_ids([target])
        if target.get("status") != "deleted" or not ids:
            rechecked.append(target)  # a non-deleted target verified no vector
            continue
        counts = await _db_residual_counts(
            db,
            ids,
            # Cross-user rows were counted BEFORE the delete and cannot be
            # re-counted now: preserve what the erase recorded (R29c). Same for
            # the derivatives another namespace of this user still holds (I3):
            # they exist, this scope cannot see them, so the receipt keeps the
            # residual and stays open instead of upgrading over live data.
            cross_user_children=int(
                (target.get("db_residual") or {}).get("cross_user_children", 0)
            ),
            cascaded_out_of_namespace=int(
                (target.get("db_residual") or {}).get("cascaded_out_of_namespace", 0)
            ),
            derived_out_of_namespace=int(
                (target.get("db_residual") or {}).get("derived_out_of_namespace", 0)
            ),
        )
        if sum(counts.values()):
            return False  # a residual is not a completion
        rechecked.append({
            **target,
            "vector_state": VECTOR_STATE_VERIFIED,
            "vector_residual": [],
            "vector_residual_checked": True,
            "db_residual": counts,
            "index_pending": await _pending_delete_intents(db, ids),
        })
    receipt.status = ERASURE_STATUS_COMPLETED
    receipt.detail = {
        **(receipt.detail or {}),
        "targets": rechecked,
        "verification": VECTOR_STATE_VERIFIED,
        "index_pending": sum(int(t.get("index_pending") or 0) for t in rechecked),
    }
    return True


__all__ = [
    "ERASURE_STATUS_COMPLETED",
    "ERASURE_STATUS_ERRORS",
    "ERASURE_STATUS_RESIDUAL",
    "ERASURE_STATUS_UNVERIFIED",
    "NOT_FOUND_OR_FOREIGN",
    "VECTOR_STATE_PENDING",
    "VECTOR_STATE_RESIDUAL",
    "VECTOR_STATE_UNKNOWN",
    "VECTOR_STATE_VERIFIED",
    "erase_memories",
    "list_receipts",
    "reconcile_erasure_receipts",
    "soft_forget",
]
