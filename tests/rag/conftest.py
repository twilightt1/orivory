"""RAG test environment defaults + isolated SQLite DB fixtures.

``--confcutdir=tests/rag`` runs this suite without the shared root conftest
(no Postgres provisioning, no ambient ``db`` fixture), so the two DB fixtures
below build a PRIVATE SQLite file on the test's ``tmp_path`` and point the
module-level engines at it. Nothing in this suite can read or write an
ambient database (pattern: ``tests/retrieval/test_index_outbox.py``).
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

RAG_DB = "rag-test.sqlite"


def _url(tmp_path, *, async_driver: bool) -> str:
    scheme = "sqlite+aiosqlite" if async_driver else "sqlite"
    return f"{scheme}:///{tmp_path / RAG_DB}"


@pytest.fixture
def sync_db(tmp_path, monkeypatch):
    """A real sync Session on a private temp SQLite file (Celery/ingestion face)."""
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


@pytest_asyncio.fixture
async def db(tmp_path, monkeypatch):
    """A real AsyncSession on a private temp SQLite file (API face)."""
    engine = create_async_engine(_url(tmp_path, async_driver=True), poolclass=NullPool)
    event.listen(engine.sync_engine, "connect", database._configure_sqlite_connection)
    sessions = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False, autoflush=False)
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    monkeypatch.setattr(database, "AsyncSessionLocal", sessions)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with sessions() as session:
        yield session
    await engine.dispose()
