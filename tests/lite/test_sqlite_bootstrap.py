"""Lite-mode smoke tests: SQLite bootstrap + zero-external-services wiring.

These run the REAL app code against a temporary SQLite file — no Docker,
no Postgres, no Redis, no MinIO, no Chroma server.
"""
from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from app.database import IS_SQLITE, AsyncSessionLocal, Base, engine

pytestmark = pytest.mark.skipif(not IS_SQLITE, reason="lite-mode tests require a sqlite DATABASE_URL")


@pytest_asyncio.fixture
async def _tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


async def test_bootstrap_creates_all_hub_tables(_tables):

    from sqlalchemy import inspect as sa_inspect

    async with engine.connect() as conn:
        names = set(await conn.run_sync(lambda c: sa_inspect(c).get_table_names()))
    expected = {
        "users", "memories", "agent_clients", "memory_access_logs",
        "erasure_receipts", "conversations", "messages", "documents",
        "document_chunks", "entities", "relations", "sources",
        "user_quotas",
    }
    missing = expected - names
    assert not missing, f"missing tables: {missing}"


async def test_memory_roundtrip_on_sqlite(_tables):
    from datetime import UTC, datetime

    from sqlalchemy import select

    from app.models.memory import Memory
    from app.models.user import User

    user_id = uuid.uuid4()
    async with AsyncSessionLocal() as db:
        db.add(User(id=user_id, email="lite@example.com", hashed_password="x",
                     display_name="Lite", is_verified=True, is_active=True))
        db.add(Memory(user_id=user_id, title="lite", content="works on sqlite",
                      tags=["test"], captured_at=datetime.now(UTC)))
        await db.commit()

    async with AsyncSessionLocal() as db:
        row = (await db.execute(
            select(Memory).where(Memory.user_id == user_id)
        )).scalars().first()
        assert row is not None and row.content == "works on sqlite"
