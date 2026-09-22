"""The personal-context cap must not drop a pinned memory (finding 527).

The module documents pinned memories as "always include", but the final
``sort(captured_at desc)`` + truncate to ``cap`` silently dropped one that was
older than the newest ``cap`` rows: 30 newer memories + one pinned 42 days old
with ``cap=30`` returned the 30 newer ones and no pinned row.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest_asyncio
from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from app import database
from app.database import Base
from app.models.memory import Memory
from app.models.user import User
from app.retrieval.memory import namespaces
from app.retrieval.memory.context import fetch_personal_context

CONTEXT_DB = "personal-context.sqlite"


@pytest_asyncio.fixture
async def db(tmp_path, monkeypatch):
    """A private per-test SQLite file (the test_visibility.py pattern)."""
    import app.models  # noqa: F401 — register every ORM table on Base

    url = f"sqlite+aiosqlite:///{tmp_path / CONTEXT_DB}"
    eng = create_async_engine(
        url, connect_args={"check_same_thread": False}, poolclass=NullPool
    )
    event.listen(eng.sync_engine, "connect", database._configure_sqlite_connection)
    sync_eng = create_engine(
        url.replace("+aiosqlite", ""),
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )
    event.listen(sync_eng, "connect", database._configure_sqlite_connection)
    sessions = async_sessionmaker(eng, class_=AsyncSession, expire_on_commit=False, autoflush=False)

    monkeypatch.setattr(database, "engine", eng)
    monkeypatch.setattr(database, "AsyncSessionLocal", sessions)
    monkeypatch.setattr(
        database,
        "_get_sync_sessionmaker",
        lambda: sessionmaker(bind=sync_eng, expire_on_commit=False, autoflush=False),
    )

    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        async with sessions() as session:
            yield session
    finally:
        await eng.dispose()
        sync_eng.dispose()


async def _owner(db) -> uuid.UUID:
    uid = uuid.uuid4()
    db.add(User(id=uid, email=f"{uid.hex}@test.invalid", hashed_password="x",
                display_name="Owner", is_verified=True, is_active=True))
    await db.commit()
    return uid


def _mem(owner, title: str, *, minutes: int = 0, pinned: bool = False) -> Memory:
    return Memory(
        id=uuid.uuid4(), user_id=owner, title=title, content=f"{title} body",
        tags=[], salience=0.5, pinned=pinned,
        captured_at=datetime.now(UTC) - timedelta(minutes=minutes),
        extra_metadata={}, namespace=namespaces.PERSONAL,
    )


async def test_a_pinned_row_older_than_the_cap_is_still_included(db):
    """30 newer memories + one pinned 42 days old, cap=30: the pinned row stays."""
    owner = await _owner(db)
    pinned = _mem(owner, "pinned 42 days old", minutes=42 * 24 * 60, pinned=True)
    newer = [_mem(owner, f"recent {i}", minutes=i) for i in range(30)]
    db.add_all([pinned, *newer])
    await db.commit()

    out = await fetch_personal_context(db, owner, cap=30)
    ids = [m.id for m in out]

    assert pinned.id in ids, "'pinned' is documented as always include"
    assert len(out) == 30, "the cap still holds"
    assert ids[0] == newer[0].id, "the slice is still recency-sorted"
    assert ids == sorted(ids, key=lambda i: next(
        m.captured_at for m in out if m.id == i), reverse=True)


async def test_pinned_rows_alone_never_exceed_the_cap(db):
    """A pinned-only slice is capped too — the reservation cannot run away."""
    owner = await _owner(db)
    db.add_all([_mem(owner, f"pinned {i}", minutes=i, pinned=True) for i in range(35)])
    await db.commit()

    out = await fetch_personal_context(db, owner, cap=30)

    assert len(out) == 30
