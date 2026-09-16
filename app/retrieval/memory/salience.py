"""Salience feedback loop (P2.1).

A memory's ``salience`` ∈ [0, 1] is meant to reflect how useful it has proven
over time, and it feeds ranking via ``time_decay_score``. This module owns the
two halves of the loop:

  - ``bump_salience`` — when a memory is recalled AND used in an answer, nudge
    its salience up (asymptotically toward 1.0) and stamp ``last_used_at``.
  - decay was a periodic job alongside ``bump_salience``; it was removed
    with the task queue (no scheduler in Lite), so only the bump half runs.

The bump is asymptotic: ``new = old + step * (1 - old)``. This rewards repeated
usefulness while never exceeding 1.0 and giving diminishing returns, so a
single lucky match can't dominate.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.memory import Memory
from app.retrieval.memory.namespaces import personal_namespace
from app.retrieval.memory.visibility import namespace_predicate

log = logging.getLogger(__name__)

DEFAULT_BUMP_STEP = 0.1
SALIENCE_MAX = 1.0


def next_salience(current: float, *, step: float = DEFAULT_BUMP_STEP) -> float:
    """Asymptotic increment toward 1.0. Pure function (unit-testable)."""
    current = max(0.0, min(SALIENCE_MAX, float(current)))
    return round(current + step * (SALIENCE_MAX - current), 6)


async def bump_salience(
    db: AsyncSession,
    user_id: UUID,
    memory_ids: list[UUID | str],
    *,
    step: float = DEFAULT_BUMP_STEP,
    now: datetime | None = None,
) -> int:
    """Increase salience for memories that were used in an answer.

    Scoped by ``user_id`` AND the user's own namespace (P4a: ``personal``,
    resolved through ``namespaces.personal_namespace``) so a stray id can't
    touch another user's data — or a row of theirs outside this surface's
    namespace. Increments ``recall_count`` and stamps ``last_used_at``. Returns
    the number of rows updated. Best-effort: the caller should not let a
    failure here break the chat turn.
    """
    ids: list[UUID] = []
    for mid in memory_ids:
        if isinstance(mid, UUID):
            ids.append(mid)
        else:
            try:
                ids.append(UUID(str(mid)))
            except (ValueError, TypeError):
                continue
    if not ids:
        return 0

    stamp = now or datetime.now(UTC)

    rows = (
        await db.execute(
            select(Memory.id, Memory.salience).where(
                Memory.user_id == user_id,
                namespace_predicate(personal_namespace(user_id)),
                Memory.id.in_(ids),
            )
        )
    ).all()

    updated = 0
    for mem_id, salience in rows:
        await db.execute(
            update(Memory)
            .where(Memory.id == mem_id)
            .values(
                salience=next_salience(salience, step=step),
                recall_count=Memory.recall_count + 1,
                last_used_at=stamp,
            )
        )
        updated += 1

    if updated:
        await db.commit()
    return updated


__all__ = ["bump_salience", "next_salience", "DEFAULT_BUMP_STEP", "SALIENCE_MAX"]
