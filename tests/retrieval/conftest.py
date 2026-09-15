"""Shared fixtures for the retrieval suites.

``MemoryRetriever.recall`` runs the P3 freshness barrier before anything reads
the index, and the barrier counts this tenant's pending intents through
``freshness``'s OWN sessionmaker (``AsyncSessionLocal``) — one ``index_outbox``
read that has to SUCCEED. A suite that fakes the retriever's DB (or keeps a
private one for itself) still has to satisfy that read: pointed at a schema
without the table it raises, and by ruling R14 the barrier fails closed after
its budget instead of returning — a slow red for a suite that never meant to
test the outbox at all.
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app import models as _models  # noqa: F401 — register every table on Base
from app.database import Base
from app.retrieval.memory import freshness


@pytest.fixture
def barrier_outbox(tmp_path, monkeypatch):
    """A real, bootstrapped SQLite outbox for the freshness barrier.

    Empty table → 0 pending → the barrier's fast path: one count query, no
    drain, no wait. Built with a sync engine on the same file (the house
    pattern) so this stays a plain sync fixture, usable from async tests.
    """
    url = f"sqlite+aiosqlite:///{tmp_path / 'freshness-outbox.db'}"
    setup = create_engine(url.replace("+aiosqlite", ""))
    Base.metadata.create_all(setup)
    setup.dispose()
    monkeypatch.setattr(
        freshness,
        "AsyncSessionLocal",
        async_sessionmaker(
            create_async_engine(url, poolclass=NullPool),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        ),
    )
