"""Sync/async SQLite parity: FK + WAL on both engines (P0 gate)."""
from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from app.database import IS_SQLITE, AsyncSessionLocal, engine, get_sync_engine, sync_session

pytestmark = pytest.mark.skipif(not IS_SQLITE, reason="sqlite DATABASE_URL required")


async def _pragma_async(name: str):
    async with engine.connect() as c:
        return (await c.execute(text(f"PRAGMA {name}"))).scalar()


def test_sync_engine_uses_sqlite_driver_and_pragmas():
    eng = get_sync_engine()
    assert eng.url.get_backend_name() == "sqlite", f"sync engine is {eng.url}"
    assert "aiosqlite" not in str(eng.url), "sync engine must not use async driver"
    with eng.connect() as c:
        assert c.execute(text("PRAGMA foreign_keys")).scalar() == 1
        assert c.execute(text("PRAGMA journal_mode")).scalar().lower() == "wal"
        assert c.execute(text("PRAGMA busy_timeout")).scalar() == 5000


async def test_async_and_sync_see_same_row():
    from app.models.memory import Memory
    from app.models.user import User

    assert await _pragma_async("foreign_keys") == 1
    assert (await _pragma_async("journal_mode")).lower() == "wal"
    assert await _pragma_async("busy_timeout") == 5000

    uid, mid = uuid.uuid4(), uuid.uuid4()
    async with AsyncSessionLocal() as db:
        db.add(User(id=uid, email=f"p0-{uid.hex[:8]}@t.test", hashed_password="x"))
        db.add(Memory(id=mid, user_id=uid, title="p0", content="parity probe", tags=[]))
        await db.commit()
    with sync_session() as s:
        row = s.get(Memory, mid)
        assert row is not None and row.content == "parity probe"
