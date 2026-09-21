"""Graph builder/extraction tests: private SQLite file, no provider calls.

Every test builds its own file on ``tmp_path`` and binds the app's engines to
it (pattern: ``tests/rag/conftest.py``), so nothing here can touch an ambient
database. The LLM client is always stubbed — the suite makes no network calls.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault(
    "DATABASE_URL", f"sqlite+aiosqlite:///{tempfile.mkdtemp(prefix='orivory-graph-')}/graph-ambient.db"
)
os.environ.setdefault("ENVIRONMENT", "test")

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app import database, models  # noqa: F401 — register every table on Base.metadata
from app.database import Base
from app.models.memory import Memory
from app.models.user import User


@pytest.fixture
def graph_store(tmp_path, monkeypatch):
    """One private SQLite file + a sync sessionmaker + a committed memory row."""
    path = tmp_path / "graph-build.db"
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
    event.listen(engine, "connect", database._configure_sqlite_connection)
    Base.metadata.create_all(engine)
    maker = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    monkeypatch.setattr(database, "_get_sync_sessionmaker", lambda: maker)

    user_id = uuid.uuid4()
    memory_id = uuid.uuid4()
    with maker() as session:
        session.add(User(id=user_id, email=f"{user_id.hex}@test.invalid", hashed_password="x",
                         display_name="Owner", is_verified=True, is_active=True))
        session.add(Memory(
            id=memory_id,
            user_id=user_id,
            content="Project Atlas shipped the lantern walk on 2026-01-02.",
            title="Atlas",
            tags=["project"],
            pinned=False,
            is_shared=False,
            recall_count=0,
            captured_at=datetime.now(UTC),
        ))
        session.commit()

    try:
        yield SimpleNamespace(
            path=str(path),
            maker=maker,
            engine=engine,
            user_id=user_id,
            memory_id=memory_id,
            memory_id_hex=memory_id.hex,
        )
    finally:
        engine.dispose()


def add_memory(store, *, content: str = "Second memory about Bob.") -> uuid.UUID:
    """A second committed memory for the same owner (a second build target)."""
    memory_id = uuid.uuid4()
    with store.maker() as session:
        session.add(Memory(
            id=memory_id,
            user_id=store.user_id,
            content=content,
            title="Second",
            tags=[],
            pinned=False,
            is_shared=False,
            recall_count=0,
            captured_at=datetime.now(UTC),
        ))
        session.commit()
    return memory_id


class FakeLLM:
    """An OpenAI-shaped stub: returns the queued payloads, call by call."""

    def __init__(self, payloads, *, on_call=None):
        self.payloads = list(payloads)
        self.calls: list[dict] = []
        self._on_call = on_call
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._on_call is not None:
            self._on_call(len(self.calls))
        content = self.payloads.pop(0) if self.payloads else "{}"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def stub_llm(monkeypatch, payloads, *, on_call=None) -> FakeLLM:
    """Patch the extraction module's client seam (no provider, no network)."""
    client = FakeLLM(payloads, on_call=on_call)
    monkeypatch.setattr("app.graph.extraction._get_client", lambda: client)
    return client
