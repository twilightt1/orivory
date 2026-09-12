"""Memory erasure with verifiable receipts — Open Memory Hub MVP item 5.

One call → one ``ErasureReceipt``. Per memory id: ownership check (foreign/
missing ids recorded, never deleted), the same-user transitive descendant set
is collected BEFORE deletion via an ownership-checked ``parent_id`` BFS
(papers §3.2), row delete via the ORM cascade, ``safe_delete_from_chroma``
for every affected id, then an adversarial verification pass (papers §3.3):
re-query Chroma + re-count
residual DB rows. v0 verification = absence-checks of every derived artifact;
KG re-inference probing is a tracked follow-up.

Receipt ``status`` — precedence in this order:

- ``completed_with_errors``: at least one target's erasure raised (recorded
  per-target and skipped; the remaining targets are still erased).
- ``completed_with_residual``: no errors, but verification found residual
  vectors or DB rows, or a target's traversal hit the BFS depth cap
  (``depth_capped`` — descendants beyond the cap were never enumerated, so
  their vectors are unverified; reported as ``vectors_unverified_depth_cap``).
- ``completed``: every target erased and verified clean.

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

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity import MemoryEntity
from app.models.erasure_receipt import ErasureReceipt
from app.models.memory import Memory
from app.models.source import MemorySource
from app.retrieval.memory.write_back import safe_delete_from_chroma

log = logging.getLogger(__name__)

ERASURE_STATUS_COMPLETED = "completed"
ERASURE_STATUS_RESIDUAL = "completed_with_residual"
ERASURE_STATUS_ERRORS = "completed_with_errors"
NOT_FOUND_OR_FOREIGN = "not_found_or_foreign"

_MAX_LIMIT = 200
_MAX_BFS_DEPTH = 5  # descendants deeper than this stay untracked and are reported as depth-capped


@dataclass
class _DescendantTraversal:
    """Result of the ownership-checked descendant BFS from one root."""

    ids: list[uuid.UUID] = field(default_factory=list)   # same-user descendants, BFS order
    depth: int = 0                                       # levels that yielded nodes (0 = no children)
    truncated: bool = False                              # True when the frontier was non-empty at _MAX_BFS_DEPTH
    untracked_estimate: int = 0                          # frontier size at the cap (lower bound on untracked rows)


async def _chroma_present_ids(memory_ids: list[str]) -> set[str]:
    """Verification seam — re-query the Chroma collection for residual ids.

    Monkeypatched in tests; the vector-store helper is imported lazily so a
    missing/failed Chroma import cannot break the DB erasure.
    """
    from app.retrieval.memory.vector_store import get_memory_ids_present

    return await get_memory_ids_present(memory_ids)


async def _verify_absent(memory_ids: list[uuid.UUID]) -> set[str] | None:
    """Return ids still present in Chroma, or ``None`` when Chroma is down.

    ``None`` = verification unknown — recorded as ``vector_residual_checked:
    false`` without failing the erasure (Postgres is the source of truth).
    """
    try:
        return await _chroma_present_ids([str(m) for m in memory_ids])
    except Exception as exc:
        log.warning("Chroma residual check failed: %s", exc, extra={"memory_ids": [str(m) for m in memory_ids]})
        return None


async def _db_residual_counts(db: AsyncSession, memory_ids: list[uuid.UUID]) -> dict[str, int]:
    """Re-count cascade targets after deletion; anything > 0 is a residual."""
    children = (await db.execute(
        select(func.count(Memory.id)).where(Memory.parent_id.in_(memory_ids))
    )).scalar_one()
    entity_links = (await db.execute(
        select(func.count()).select_from(MemoryEntity).where(MemoryEntity.memory_id.in_(memory_ids))
    )).scalar_one()
    source_links = (await db.execute(
        select(func.count()).select_from(MemorySource).where(MemorySource.memory_id.in_(memory_ids))
    )).scalar_one()
    return {"children": int(children), "entity_links": int(entity_links), "source_links": int(source_links)}


async def _collect_descendants(db: AsyncSession, user_id: uuid.UUID, root_id: uuid.UUID) -> _DescendantTraversal:
    """BFS over ``parent_id`` from ``root_id`` — every same-user transitive descendant.

    The frontier query carries ``Memory.user_id == user_id`` so descendants of
    another user are never collected, deleted, or disclosed by this service.
    Cycles are impossible via the FK, but a visited-set keeps the frontier loop
    honest anyway, and ``_MAX_BFS_DEPTH`` bounds runaway/accidental deep trees:
    when the loop exits with a non-empty frontier (cap hit), the traversal is
    reported as truncated with a frontier-size estimate of the untracked rows —
    it is never silently swallowed.
    """
    result = _DescendantTraversal()
    visited = {root_id}
    frontier = [root_id]
    depth = 0
    while frontier and depth < _MAX_BFS_DEPTH:
        rows = list((await db.execute(
            select(Memory.id).where(
                Memory.parent_id.in_(frontier),
                Memory.user_id == user_id,  # cross-user descendants are never collected
            )
        )).scalars().all())
        frontier = [rid for rid in rows if rid not in visited]
        if not frontier:
            break
        depth += 1
        visited.update(frontier)
        result.ids.extend(frontier)
    result.depth = depth
    result.truncated = bool(frontier)  # loop exited on the depth cap, not on an empty frontier
    result.untracked_estimate = len(frontier) if result.truncated else 0
    return result


async def _erase_one(db: AsyncSession, user_id: uuid.UUID, memory_id: uuid.UUID) -> dict[str, Any]:
    """Erase one owned memory and return its per-target receipt entry."""
    row = await db.get(Memory, memory_id)
    if row is None or row.user_id != user_id:
        return {"memory_id": str(memory_id), "status": NOT_FOUND_OR_FOREIGN}

    # Collect the same-user transitive descendant set BEFORE deletion (receipt
    # must describe what went away; a depth-capped subtree is flagged, never
    # silently dropped).
    traversal = await _collect_descendants(db, user_id, memory_id)
    child_ids = traversal.ids
    entity_links = (await db.execute(
        select(func.count()).select_from(MemoryEntity).where(MemoryEntity.memory_id == memory_id)
    )).scalar_one()
    source_links = (await db.execute(
        select(func.count()).select_from(MemorySource).where(MemorySource.memory_id == memory_id)
    )).scalar_one()

    affected = [memory_id, *child_ids]
    try:
        from app.retrieval.memory.correction import collect_derived_ids

        for derived_id in await collect_derived_ids(db, user_id, affected):
            if derived_id not in affected:
                affected.append(derived_id)
    except Exception as exc:
        log.warning("Derived-dependent collection failed: %s", exc)
    await db.delete(row)  # ORM cascade: children + links go with it (DB CASCADE too)
    await db.commit()
    for derived_id in affected[1:]:
        if derived_id not in child_ids and derived_id != memory_id:
            derived_row = await db.get(Memory, derived_id)
            if derived_row is not None and derived_row.user_id == user_id:
                await db.delete(derived_row)
    await db.commit()

    vectors_deleted = []
    for vid in affected:
        await safe_delete_from_chroma(vid)  # never raises (write_back.py)
        vectors_deleted.append(str(vid))

    present = await _verify_absent(affected)
    db_residual = await _db_residual_counts(db, affected)
    return {
        "memory_id": str(memory_id),
        "status": "deleted",
        "affected_memory_ids": [str(c) for c in child_ids],  # transitive, excluding the target
        "traversal_depth": traversal.depth,
        "depth_capped": traversal.truncated,
        "vectors_unverified_depth_cap": traversal.untracked_estimate,
        "entity_links": int(entity_links),
        "source_links": int(source_links),
        "vectors_deleted": vectors_deleted,
        "vector_residual": sorted(present) if present is not None else [],
        "vector_residual_checked": present is not None,
        "db_residual": db_residual,
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
                "vector_residual_checked": False,
                "db_residual": None,
            })
    erased = sum(1 for t in targets if t["status"] == "deleted")
    residual_vectors = sum(len(t["vector_residual"]) for t in targets if t["status"] == "deleted")
    residual_rows = sum(sum(t["db_residual"].values()) for t in targets if t["status"] == "deleted")
    any_errors = any(t["status"] == "error" for t in targets)
    any_depth_capped = any(t["status"] == "deleted" and t["depth_capped"] for t in targets)

    if any_errors:
        status = ERASURE_STATUS_ERRORS
    elif residual_vectors or residual_rows or any_depth_capped:
        status = ERASURE_STATUS_RESIDUAL
    else:
        status = ERASURE_STATUS_COMPLETED

    receipt = ErasureReceipt(
        user_id=user_id,
        requested_memory_ids=[str(m) for m in unique_ids],
        status=status,
        detail={
            "requested_by": requested_by,
            "targets": targets,
            "summary": {
                "requested": len(unique_ids),
                "erased": erased,
                "skipped": len(unique_ids) - erased,
                "errors": sum(1 for t in targets if t["status"] == "error"),
                "residual_vectors": residual_vectors,
                "residual_rows": residual_rows,
            },
        },
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
    "NOT_FOUND_OR_FOREIGN",
    "erase_memories",
    "list_receipts",
]
