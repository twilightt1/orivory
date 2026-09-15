"""
Phase 3 — Personal context layer.

Fetches a small, high-signal slice of the user's memories to inject
into the LLM prompt and (optionally) return alongside recall results.

Strategy:
    1. Pinned memories (always include)
    2. Memories captured in the last ``lookback_days`` days
    3. The ``recent_limit`` most recent memories

Results are deduplicated by ``memory.id`` and capped at ``cap`` items,
sorted by ``captured_at`` descending.

Superseded and dirty rows are excluded IN SQL, before the per-query limits —
a post-hoc filter after the cap lets stale rows crowd out the current slice.

This is the **only** module in the memory retrieval package that
talks to the relational DB for read. Everything else (scoring, vector
search, rewriting) is pure or talks to the vector store.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.memory import Memory
from app.retrieval.memory.visibility import current_memory_predicate

log = logging.getLogger(__name__)


async def fetch_personal_context(
    db: AsyncSession,
    user_id: UUID,
    *,
    lookback_days: int = 7,
    recent_limit: int = 20,
    cap: int = 30,
) -> list[Memory]:
    """Return a deduplicated, recency-sorted slice of the user's memories.

    Args:
        db: Async DB session.
        user_id: Owning user.
        lookback_days: Include memories captured in the last N days.
        recent_limit: Also include the N most recent regardless of age.
        cap: Final result cap (default 30).

    Returns:
        List of ``Memory`` objects, sorted by ``captured_at`` desc.
    """
    now = datetime.now(UTC)
    cutoff = now - timedelta(days=lookback_days)

    # 1) Pinned memories (cap at `cap` to avoid runaway).
    pinned_q = (
        select(Memory)
        .where(Memory.user_id == user_id, Memory.pinned.is_(True),
               current_memory_predicate())
        .order_by(Memory.captured_at.desc())
        .limit(cap)
    )

    # 2) Recent-in-window memories (cap at 50 to avoid huge lists).
    recent_q = (
        select(Memory)
        .where(Memory.user_id == user_id, Memory.captured_at >= cutoff,
               current_memory_predicate())
        .order_by(Memory.captured_at.desc())
        .limit(50)
    )

    # 3) Last N by captured_at (overlap with recent-window is OK;
    #    we dedup by id below).
    last_n_q = (
        select(Memory)
        .where(Memory.user_id == user_id, current_memory_predicate())
        .order_by(Memory.captured_at.desc())
        .limit(recent_limit)
    )

    pinned = (await db.execute(pinned_q)).scalars().all()
    recent = (await db.execute(recent_q)).scalars().all()
    last_n = (await db.execute(last_n_q)).scalars().all()

    # Dedup by id (preserve insertion order: pinned > recent > last_n).
    seen: set[UUID] = set()
    combined: list[Memory] = []
    for m in list(pinned) + list(recent) + list(last_n):
        if m.id not in seen:
            seen.add(m.id)
            combined.append(m)

    # Final sort + cap.
    combined.sort(key=lambda m: m.captured_at, reverse=True)
    if len(combined) > cap:
        combined = combined[:cap]

    log.info(
        "fetch_personal_context",
        extra={
            "user_id": str(user_id),
            "pinned": len(pinned),
            "recent_in_window": len(recent),
            "last_n": len(last_n),
            "returned": len(combined),
        },
    )
    return combined


def format_personal_context(
    memories: list[Memory] | None,
    *,
    max_items: int = 30,
    snippet_chars: int = 120,
) -> str:
    """Render memories into a compact context string for prompt injection.

    One line per memory::

        <YYYY-MM-DD> <title>: <snippet>

    Empty list (or None) returns ``"(empty)"``.
    """
    if not memories:
        return "(empty)"
    lines: list[str] = []
    for m in memories[:max_items]:
        date = m.captured_at.strftime("%Y-%m-%d") if m.captured_at else "????-??-??"
        title = m.title or "(untitled)"
        body = (m.content or "").replace("\n", " ")
        snippet = body[:snippet_chars]
        if len(body) > snippet_chars:
            snippet += "..."
        lines.append(f"{date} {title}: {snippet}")
    return "\n".join(lines)
