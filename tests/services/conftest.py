"""Service test environment defaults + isolated SQLite DB fixtures.

``--confcutdir=tests/services`` runs this suite without the shared root
conftest, so the fixtures below build a PRIVATE SQLite file on the test's
``tmp_path`` and point the module-level engines at it — nothing here can read
or write an ambient database (pattern: ``tests/rag/conftest.py``).
"""

import os

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://postgres:password@localhost:5432/ragdb_test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-change-in-production")
os.environ.setdefault("ENVIRONMENT", "test")

import pytest
import pytest_asyncio
from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from app import database, models  # noqa: F401 — register every table on Base.metadata
from app.database import Base

SERVICE_DB = "services-test.sqlite"


def _url(tmp_path, *, async_driver: bool) -> str:
    scheme = "sqlite+aiosqlite" if async_driver else "sqlite"
    return f"{scheme}:///{tmp_path / SERVICE_DB}"


@pytest_asyncio.fixture
async def sessions(tmp_path, monkeypatch):
    """Private per-test SQLite file + the sessionmaker bound to it."""
    engine = create_async_engine(_url(tmp_path, async_driver=True), poolclass=NullPool)
    event.listen(engine.sync_engine, "connect", database._configure_sqlite_connection)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False, autoflush=False)
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    monkeypatch.setattr(database, "AsyncSessionLocal", factory)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield factory
    await engine.dispose()


@pytest_asyncio.fixture
async def db(sessions):
    """A real AsyncSession on the private temp SQLite file (API/service face)."""
    async with sessions() as session:
        yield session


@pytest.fixture
def sync_db(tmp_path, monkeypatch):
    """A real sync Session on the same temp file (Celery/ingestion face)."""
    engine = create_engine(_url(tmp_path, async_driver=False),
                           connect_args={"check_same_thread": False})
    event.listen(engine, "connect", database._configure_sqlite_connection)
    Base.metadata.create_all(engine)
    maker = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    monkeypatch.setattr(database, "_get_sync_sessionmaker", lambda: maker)
    session = maker()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()
