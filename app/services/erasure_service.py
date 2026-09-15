"""Memory erasure with verifiable receipts — Open Memory Hub MVP item 5.

One call → one ``ErasureReceipt``, one closure transaction per target:
ownership check (foreign/missing ids recorded, never deleted) → the SAME-USER
transitive closure collected BEFORE any delete (``parent_id`` BFS with a
visited set and no silent depth cap, plus the derived-memory set; papers
§3.2) → row deletes + one durable delete intent per affected id + a
suppression row when a forgotten projection's source still exists, all in one
commit → best-effort ``safe_delete_from_chroma`` per affected id → adversarial
verification (papers §3.3): re-query Chroma + re-count residual DB rows.
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
erasing user. Cross-user children are still removed by the DB-level ON DELETE
CASCADE when the parent row goes, but they are unknowable here — their vectors
are not verifiable from the erasing user's scope. Root fix is parent
ownership validation at memory creation (``create_memory``), so cross-user
parenting cannot arise in the first place.
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
from app.retrieval.memory.correction import DerivedClosureError, collect_derived_ids
from app.retrieval.memory.outbox import (
    KIND_MEMORY,
    OPERATION_DELETE,
    enqueue_delete,
)
from app.retrieval.memory.write_back import safe_delete_from_chroma

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


async def _chroma_present_ids(memory_ids: list[str]) -> set[str]:
    """Verification seam — re-query the Chroma collection for residual ids.

    Monkeypatched in tests; the vector-store helper is imported lazily so a
    missing/failed Chroma import cannot break the DB erasure.
    """
    from app.retrieval.memory.vector_store import get_memory_ids_present

    return await get_memory_ids_present(memory_ids)


async def _verify_absent(memory_ids: list[uuid.UUID]) -> set[str] | None:
    """Return ids still present in Chroma, or ``None`` when Chroma is down.

    ``None`` = verification unknown — recorded as ``vector_state="unknown"``
    (or ``"pending"`` when a purge also failed); it is never reported as
    ``verified``. Postgres remains the source of truth.
    """
    try:
        return await _chroma_present_ids([str(m) for m in memory_ids])
    except Exception as exc:
        log.warning("Chroma residual check failed: %s", exc, extra={"memory_ids": [str(m) for m in memory_ids]})
        return None


async def _db_residual_counts(
    db: AsyncSession, memory_ids: list[uuid.UUID], *, cross_user_children: int = 0
) -> dict[str, int]:
    """Re-count cascade targets after deletion; anything > 0 is a residual.

    ``cross_user_children`` is passed in because it can only be counted BEFORE
    the delete (the cascade has already removed those rows by now) — see
    :func:`_cross_user_cascade_count`.
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


async def _collect_descendants(db: AsyncSession, user_id: uuid.UUID, root_id: uuid.UUID) -> _DescendantTraversal:
    """Full transitive descendant closure over ``parent_id`` — no silent cap.

    The frontier query carries ``Memory.user_id == user_id`` so descendants of
    another user are never collected, deleted, or disclosed by this service.
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
    if row is None or row.user_id != user_id:
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
    try:
        for derived_id in await collect_derived_ids(db, user_id, affected):
            if derived_id in affected or derived_id in derived_ids:
                continue
            derived_row = await db.get(Memory, derived_id)
            if derived_row is None or derived_row.user_id != user_id:
                continue  # vanished or cross-user: not ours to erase or record
            derived_ids.append(derived_id)
            revisions[derived_id] = int(derived_row.revision or 1)
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
    # Count the rows the DB cascade is about to remove for OTHER users (R29c):
    # after the delete they are gone and their vectors were never enumerable
    # from this scope — the receipt must still carry them as a residual.
    cross_user_children = await _cross_user_cascade_count(db, affected, user_id=user_id)

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
        if await safe_delete_from_chroma(vid) is not True:
            purge_failed = True
        vectors_deleted.append(str(vid))

    present = await _verify_absent(affected)
    db_residual = await _db_residual_counts(db, affected, cross_user_children=cross_user_children)
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
        # (Chroma unreachable) reports `completed_unverified` instead.
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
]
