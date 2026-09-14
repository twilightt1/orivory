"""Self-contained SQLite sync/async parity and durability gates."""
from __future__ import annotations

import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import create_engine, delete, event, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from app import database
from app.database import Base


@pytest_asyncio.fixture
async def sqlite_engines(tmp_path: Path):
    """Create isolated canonical sync/async engines and bootstrap their schema."""
    from app import models  # noqa: F401 - register all ORM tables

    db_path = tmp_path / "parity.sqlite"
    async_engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )
    sync_engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )
    event.listen(async_engine.sync_engine, "connect", database._configure_sqlite_connection)
    event.listen(sync_engine, "connect", database._configure_sqlite_connection)

    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield async_engine, sync_engine
    finally:
        await async_engine.dispose()
        sync_engine.dispose()


async def _pragma_async(async_engine, name: str):
    async with async_engine.connect() as conn:
        return (await conn.execute(text(f"PRAGMA {name}"))).scalar()


def _pragma_sync(sync_engine, name: str):
    with sync_engine.connect() as conn:
        return conn.execute(text(f"PRAGMA {name}")).scalar()


@pytest.mark.asyncio
async def test_both_engines_use_durable_sqlite_pragmas(sqlite_engines):
    async_engine, sync_engine = sqlite_engines

    assert await _pragma_async(async_engine, "foreign_keys") == 1
    assert await _pragma_async(async_engine, "journal_mode") == "wal"
    assert await _pragma_async(async_engine, "busy_timeout") == 5000
    assert await _pragma_async(async_engine, "synchronous") == 2  # FULL

    assert _pragma_sync(sync_engine, "foreign_keys") == 1
    assert _pragma_sync(sync_engine, "journal_mode") == "wal"
    assert _pragma_sync(sync_engine, "busy_timeout") == 5000
    assert _pragma_sync(sync_engine, "synchronous") == 2  # FULL


@pytest.mark.asyncio
async def test_async_and_sync_writes_read_same_file_both_directions(sqlite_engines):
    async_engine, sync_engine = sqlite_engines
    from app.models.memory import Memory
    from app.models.user import User

    async_user, async_memory = uuid.uuid4(), uuid.uuid4()
    sync_user, sync_memory = uuid.uuid4(), uuid.uuid4()
    try:
        async with AsyncSession(async_engine, expire_on_commit=False) as db:
            db.add(User(id=async_user, email=f"async-{async_user.hex}@test.invalid", hashed_password="x"))
            db.add(Memory(id=async_memory, user_id=async_user, title="async", content="async write", tags=[]))
            await db.commit()

        with Session(sync_engine, expire_on_commit=False) as db:
            row = db.get(Memory, async_memory)
            assert row is not None and row.content == "async write"
            db.add(User(id=sync_user, email=f"sync-{sync_user.hex}@test.invalid", hashed_password="x"))
            db.add(Memory(id=sync_memory, user_id=sync_user, title="sync", content="sync write", tags=[]))
            db.commit()

        async with AsyncSession(async_engine) as db:
            row = await db.get(Memory, sync_memory)
            assert row is not None and row.content == "sync write"
    finally:
        with Session(sync_engine) as db:
            db.execute(delete(User).where(User.id.in_([async_user, sync_user])))
            db.commit()


@pytest.mark.asyncio
async def test_foreign_keys_are_enforced_on_both_write_paths(sqlite_engines):
    async_engine, sync_engine = sqlite_engines
    from app.models.memory import Memory

    with pytest.raises(IntegrityError):
        with Session(sync_engine) as db:
            db.add(Memory(id=uuid.uuid4(), user_id=uuid.uuid4(), content="orphan", tags=[]))
            db.commit()

    with pytest.raises(IntegrityError):
        async with AsyncSession(async_engine) as db:
            db.add(Memory(id=uuid.uuid4(), user_id=uuid.uuid4(), content="orphan", tags=[]))
            await db.commit()


@pytest.mark.asyncio
async def test_bootstrap_versions_fresh_schema_and_rejects_divergent_unversioned_existing(tmp_path, monkeypatch):
    fresh_path = tmp_path / "fresh.sqlite"
    fresh_engine = create_async_engine(
        f"sqlite+aiosqlite:///{fresh_path}",
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )
    event.listen(fresh_engine.sync_engine, "connect", database._configure_sqlite_connection)
    monkeypatch.setattr(database, "engine", fresh_engine)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    try:
        await database.bootstrap_sqlite()
        assert await _pragma_async(fresh_engine, "user_version") == database.SQLITE_SCHEMA_VERSION
        await database.bootstrap_sqlite()  # versioned rerun is a no-op
    finally:
        await fresh_engine.dispose()

    existing_path = tmp_path / "existing.sqlite"
    existing_engine = create_async_engine(
        f"sqlite+aiosqlite:///{existing_path}",
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )
    event.listen(existing_engine.sync_engine, "connect", database._configure_sqlite_connection)
    try:
        async with existing_engine.begin() as conn:
            await conn.execute(text("CREATE TABLE legacy_marker (id INTEGER PRIMARY KEY)"))
        monkeypatch.setattr(database, "engine", existing_engine)
        with pytest.raises(RuntimeError, match="versioned SQLite migration"):
            await database.bootstrap_sqlite()
    finally:
        await existing_engine.dispose()
