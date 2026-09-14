"""Adoption of pre-versioning SQLite installs (full schema, user_version=0)."""
from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app import database, models  # noqa: F401 — register every model on Base
from app.database import Base

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def legacy_db(tmp_path: Path, monkeypatch):
    """A pre-versioning install: full create_all schema, user_version left at 0.

    Self-contained: own SQLite file + patched module engine, so it runs
    regardless of the ambient DATABASE_URL (CI has no lite job).
    """
    eng = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'legacy.sqlite'}",
        poolclass=NullPool,
    )
    monkeypatch.setattr(database, "engine", eng)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


async def _user_version(eng) -> int:
    async with eng.connect() as conn:
        return int((await conn.execute(text("PRAGMA user_version"))).scalar_one())


async def test_unversioned_matching_schema_is_adopted(legacy_db):
    assert await _user_version(legacy_db) == 0

    await database.bootstrap_sqlite()

    assert await _user_version(legacy_db) == database.SQLITE_SCHEMA_VERSION


async def test_unversioned_divergent_schema_fails_closed(legacy_db):
    async with legacy_db.begin() as conn:
        await conn.execute(text("ALTER TABLE memories RENAME COLUMN content TO content_v0"))

    with pytest.raises(RuntimeError, match="does not match SQLITE_SCHEMA_VERSION"):
        await database.bootstrap_sqlite()

    assert await _user_version(legacy_db) == 0  # untouched, not silently stamped
