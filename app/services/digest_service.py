"""Proactive surfacing digest (P2.2).

Turns the passive "ask and it answers" model into one that also *brings things
back*. Two complementary signals, both derived from data already stored — no
LLM cost:

  1. **This week** — what the user captured recently, plus the top themes
     (tags) in that window, so the digest can say "you saved 3 things about
     Postgres indexing this week."

  2. **On this day** — memories captured around the same calendar date in
     prior years/months ("1 year ago today you read X"), using ``captured_at``.
     This is the time-aware resurfacing the second-brain framing promises.
"""
from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.memory import Memory
from app.schemas.Orivory import (
    DigestResponse,
    DigestResurfacedMemory,
    DigestThemeCount,
    MemoryResponse,
)

DEFAULT_WINDOW_DAYS = 7
RECENT_LIMIT = 20
TOP_THEMES_LIMIT = 5
RESURFACE_LIMIT = 5
# A past memory counts as "on this day" if its captured_at falls within this
# many days of today's month/day in a previous year.
RESURFACE_DAY_TOLERANCE = 2
RESURFACE_MIN_AGE_DAYS = 180  # only resurface things genuinely from the past


def _memory_response(memory: Memory) -> MemoryResponse:
    return MemoryResponse(
        id=memory.id,
        user_id=memory.user_id,
        parent_id=memory.parent_id,
        source_type=memory.source_type,
        source_ref=memory.source_ref,
        source_url=memory.source_url,
        title=memory.title,
        content=memory.content,
        summary=memory.summary,
        tags=memory.tags or [],
        salience=memory.salience,
        pinned=memory.pinned,
        recall_count=memory.recall_count,
        last_used_at=memory.last_used_at,
        captured_at=memory.captured_at,
        indexed_at=memory.indexed_at,
        updated_at=memory.updated_at,
        revision=memory.revision or 1,  # unsaved/detached rows carry the column default
        metadata=memory.extra_metadata or {},
    )


def _age_label(age_days: int) -> str:
    if age_days >= 365:
        years = round(age_days / 365)
        return f"{years} year{'s' if years != 1 else ''} ago"
    months = max(1, round(age_days / 30))
    return f"{months} month{'s' if months != 1 else ''} ago"


def _same_calendar_day(a: datetime, b: datetime, tolerance_days: int) -> bool:
    """True if ``a``'s month/day is within tolerance of ``b``'s, any year."""
    try:
        this_year_anniversary = a.replace(year=b.year)
    except ValueError:
        # Feb 29 in a non-leap comparison year — fall back to Feb 28.
        this_year_anniversary = a.replace(year=b.year, day=28)
    return abs((this_year_anniversary.date() - b.date()).days) <= tolerance_days


async def build_digest(
    db: AsyncSession,
    user_id: UUID,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    now: datetime | None = None,
) -> DigestResponse:
    now = now or datetime.now(UTC)
    window_start = now - timedelta(days=window_days)

    # ── recent window ────────────────────────────────────────────────────
    recent_rows = (
        await db.execute(
            select(Memory)
            .where(Memory.user_id == user_id, Memory.captured_at >= window_start)
            .order_by(Memory.captured_at.desc())
            .limit(RECENT_LIMIT)
        )
    ).scalars().all()

    # Count themes across ALL memories in the window (not just the page) so the
    # "you saved N about X" line is accurate. Cheap: select tags only.
    tag_rows = (
        await db.execute(
            select(Memory.tags).where(
                Memory.user_id == user_id, Memory.captured_at >= window_start
            )
        )
    ).scalars().all()
    recent_count = len(tag_rows)
    theme_counter: Counter[str] = Counter()
    for tags in tag_rows:
        for tag in tags or []:
            theme_counter[tag] += 1
    top_themes = [
        DigestThemeCount(theme=theme, count=count)
        for theme, count in theme_counter.most_common(TOP_THEMES_LIMIT)
    ]

    # ── on this day (resurfacing) ────────────────────────────────────────
    cutoff_old = now - timedelta(days=RESURFACE_MIN_AGE_DAYS)
    old_rows = (
        await db.execute(
            select(Memory)
            .where(Memory.user_id == user_id, Memory.captured_at <= cutoff_old)
            .order_by(Memory.salience.desc(), Memory.captured_at.desc())
            .limit(500)
        )
    ).scalars().all()

    resurfaced: list[DigestResurfacedMemory] = []
    for memory in old_rows:
        captured = memory.captured_at
        if captured.tzinfo is None:
            captured = captured.replace(tzinfo=UTC)
        if _same_calendar_day(captured, now, RESURFACE_DAY_TOLERANCE):
            age_days = (now - captured).days
            resurfaced.append(
                DigestResurfacedMemory(
                    memory=_memory_response(memory),
                    age_label=_age_label(age_days),
                    age_days=age_days,
                )
            )
        if len(resurfaced) >= RESURFACE_LIMIT:
            break

    return DigestResponse(
        generated_at=now,
        window_days=window_days,
        recent_count=recent_count,
        top_themes=top_themes,
        recent_memories=[_memory_response(m) for m in recent_rows],
        resurfaced=resurfaced,
    )
