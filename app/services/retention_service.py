"""P4b/T6 — opt-in retention: expire memories the system has held too long.

Auto expiration is OPT-IN (spec §8.1) and this module never decides otherwise:
a user without ``retention_enabled`` is not scanned at all, and a user enabled
without a window says nothing — no sweep. What the sweep does, per expired row:

    soft invalidate (the same state the forget path uses, T3) + the reason
    ``retention_expired`` in the append-only access ledger (``MemoryAccessLog``)

The row keeps its content and provenance: retention is a serving decision, not
a shredder. Its vectors keep existing while the payload state refreshes by
INTENT (``bump_revision`` + ``enqueue_upsert``, R37) — the same shape
``soft_forget`` uses, so the applier carries ``visibility_state=invalidated``
to the point and T1's predicates close serving.

Two deliberate differences from ``soft_forget``:

- NO suppression row. Retention is an automatic expiry, not the user forgetting
  a source; writing the suppression ledger here would block a re-import the
  user never asked to forget (T4 reads that ledger as a user decision).
- Pin EXEMPTS (spec §8.1: "Pin bảo vệ khỏi auto-retention nhưng không chặn
  explicit forget"). The pin is read here and nowhere else: explicit forget and
  erase (T3/T4) still win over it.

The clock is ``indexed_at`` — how long this install has HELD the memory — not
``captured_at``: an old email imported last week has not been retained here for
its content's age, and a retention window is a statement about the system's
copy, not about when the note was written.

Retention is personal-only like every P4b surface: the sweep is scoped to each
user's own namespace, so a future workspace row is never expired by a personal
setting.
"""
from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import NamedTuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.memory import Memory
from app.models.memory_access_log import MemoryAccessLog
from app.models.user import User
from app.retrieval.memory.correction import CM_INVALIDATED, set_cm
from app.retrieval.memory.namespaces import personal_namespace
from app.retrieval.memory.outbox import bump_revision, enqueue_upsert
from app.retrieval.memory.visibility import namespace_predicate

log = logging.getLogger(__name__)

# The reason every row expired by this sweep carries, in the audit ledger.
RETENTION_REASON = "retention_expired"


class RetentionReport(NamedTuple):
    """What one sweep did — counted, so the drain log carries it (never a raise)."""

    users: int        # enabled users the sweep ran for
    invalidated: int  # rows expired (and audited) by this run


def _not_already_invalidated():
    """Rows not already ``invalidated``: a second expiry would rewrite nothing."""
    return Memory.extra_metadata[CM_INVALIDATED].as_string().is_(None)


async def enabled_users(db: AsyncSession) -> list[tuple[uuid.UUID, int]]:
    """The opt-in queue: ``(user_id, days)`` for users who turned retention on.

    Both halves must be set. An enabled user with no window is a setting that
    says nothing about WHEN to expire, and "no setting → không chạy" is the
    rule: the sweep would have to invent a window, and an invented window is
    exactly the auto-deletion the spec forbids by default.
    """
    rows = (await db.execute(
        select(User.id, User.retention_days)
        .where(User.retention_enabled.is_(True),
               User.retention_days.is_not(None),
               User.retention_days > 0)
        .order_by(User.id)
    )).all()
    return [(row.id, int(row.retention_days)) for row in rows]


async def _expire_user(db: AsyncSession, user_id: uuid.UUID, days: int,
                       now: datetime) -> int:
    """Invalidate ``user_id``'s rows older than ``days``; return how many.

    One commit for the whole user: the invalidated states, their payload-refresh
    intents and their audit rows land together or not at all. Already-invalidated
    rows are not selected, so a re-run is a no-op by construction.

    ponytail: row-level expiry — a derived view whose sources expired keeps
    serving until its own age expires (the T5 publish-time evidence guard owns
    the publish path). Closure-walking every expired row is the upgrade path if
    that residual ever matters.
    """
    cutoff = now - timedelta(days=days)
    rows = (await db.execute(
        select(Memory).where(
            Memory.user_id == user_id,
            namespace_predicate(personal_namespace(user_id)),
            Memory.indexed_at < cutoff,
            Memory.pinned.is_(False),  # pin protects ONLY against auto-retention
            _not_already_invalidated(),
        )
    )).scalars().all()
    for row in rows:
        set_cm(row, {CM_INVALIDATED: True})
        # R37: the state must REACH the vector payload. The bump makes this a
        # real write and the durable upsert carries it to the applier (the same
        # shape soft_forget uses). No vector is purged here.
        bump_revision(row)
        await enqueue_upsert(db, row)
        db.add(MemoryAccessLog(
            user_id=user_id,
            action=RETENTION_REASON,
            memory_id=row.id,
            detail={
                "reason": RETENTION_REASON,
                "retention_days": int(days),
                "indexed_at": row.indexed_at.isoformat() if row.indexed_at else None,
            },
        ))
    if rows:
        await db.commit()
    return len(rows)


async def run_retention(db: AsyncSession, now: datetime | None = None) -> RetentionReport:
    """Expire every enabled user's rows older than their window (spec §8.1).

    ``now`` is a parameter so the clock is testable; the default is UTC now. The
    sweep is global across enabled users (the drain hook has no user context)
    and idempotent by construction: an already-invalidated row is never
    selected again. A failure is the CALLER's to catch (the drain hook is
    fail-soft): a partially finished sweep just leaves the rest for the next
    pass, and every completed user is already committed.
    """
    now = now or datetime.now(UTC)
    users = invalidated = 0
    for user_id, days in await enabled_users(db):
        users += 1
        try:
            invalidated += await _expire_user(db, user_id, days, now)
        except Exception as exc:
            # One broken window must not starve the users behind it: roll the
            # failed pass back (an invalidated row's own commit already stands)
            # and leave the rest for the next sweep. A persisted window the
            # schema now refuses (docs: le=36_500) lands here as OverflowError
            # instead of aborting every later account, every pass.
            await db.rollback()
            log.warning("Retention sweep failed for user %s: %s", user_id, exc,
                        extra={"user_id": str(user_id), "retention_days": days})
    return RetentionReport(users=users, invalidated=invalidated)


__all__ = ["RETENTION_REASON", "RetentionReport", "enabled_users", "run_retention"]
