"""P2/T9 — the graph build must really run on the memory write paths.

Both write paths that create a ``Memory`` end in
``write_back.safe_enqueue_graph_build``: the API/MCP path through
``index_new_memory`` (shared with the importer) and the connector sync through
``SourceSyncService._index_memories``. Driving that helper *on the event loop*
left the build silently dead — the sync builder drives the extraction with
``asyncio.run`` (``app/graph/builder.py``), which raises inside a running loop,
and the helper's best-effort ``except`` swallowed the ``RuntimeError``: the
memory row committed, ``graph_extracted_at`` stayed unset, and no entity or
relation row was ever written for it.

Nothing about the stores or the builder is faked here: a private SQLite file
bound as the app's own (the P1b gate's ``_bind_engines``), the real
``build_memory_graph_sync``, and the real deterministic extraction fallback.
Two seams are substituted only because no claim in this file is about them:
the LLM provider (``app.graph.extraction._get_client`` raises, so extraction
takes its offline fallback) and the embed leg (T1 owns that one, with the real
embedder, in ``test_event_loop_responsiveness.py``).
"""
from __future__ import annotations

import logging
import threading
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import select

from app import database
from app.config import settings
from app.database import sync_session
from app.graph.builder import build_memory_graph_sync
from app.ingestion.dispatcher import SourceSyncService
from app.models.entity import Entity, MemoryEntity, Relation
from app.models.memory import Memory
from app.models.user import User
from app.retrieval.memory import write_back
from tests.retrieval.test_p1b_gate import _bind_engines

CONTENT = "Project Atlas shipped the lantern walk on 2026-01-02."


@pytest_asyncio.fixture
async def store(tmp_path, monkeypatch):
    """A private SQLite file bound as the app's own + an offline LLM seam."""
    db_path = tmp_path / "t9-graph-build.db"
    url = f"sqlite+aiosqlite:///{db_path}"
    monkeypatch.setattr(settings, "DATABASE_URL", url)
    engine, sync_engine, sessions = _bind_engines(url, monkeypatch)

    calls: list[dict[str, object]] = []

    def _no_llm_seam():
        """Record where extraction ran, then fail into the offline fallback."""
        calls.append(
            {"thread": threading.get_ident(), "name": threading.current_thread().name}
        )
        raise RuntimeError("no provider in this test")

    monkeypatch.setattr("app.graph.extraction._get_client", _no_llm_seam)

    async def _embedded(_memory):
        return True

    # The embed leg is T1's claim (real embedder, its own module); here it only
    # has to succeed so the graph leg is what this file measures.
    monkeypatch.setattr(write_back, "safe_upsert_to_index", _embedded)

    await database.bootstrap_sqlite()
    try:
        yield SimpleNamespace(sessions=sessions, sync_engine=sync_engine, calls=calls)
    finally:
        await engine.dispose()
        sync_engine.dispose()


async def _seed_memory(store) -> tuple[uuid.UUID, uuid.UUID]:
    """A committed user + memory: the state the write paths index from."""
    async with store.sessions() as db:
        user = User(
            id=uuid.uuid4(), email=f"t9-{uuid.uuid4().hex[:8]}@t9.invalid",
            hashed_password="x", display_name="t9", is_verified=True, is_active=True,
        )
        db.add(user)
        await db.flush()
        memory = Memory(
            id=uuid.uuid4(), user_id=user.id, title="Atlas note", content=CONTENT,
            tags=["atlas"], captured_at=datetime.now(UTC),
        )
        db.add(memory)
        await db.commit()
        return user.id, memory.id


def _graph_state(memory_id) -> tuple[dict, int, int, int]:
    """(metadata, entities, entity links, relations) straight from the store."""
    with sync_session() as db:
        row = db.get(Memory, memory_id)
        assert row is not None, f"memory {memory_id} is missing"
        entities = db.execute(select(Entity)).scalars().all()
        links = db.execute(
            select(MemoryEntity).where(MemoryEntity.memory_id == memory_id)
        ).scalars().all()
        relations = db.execute(select(Relation)).scalars().all()
        return dict(row.extra_metadata or {}), len(entities), len(links), len(relations)


async def test_api_and_mcp_write_path_builds_the_graph_off_the_loop(store):
    """The API/MCP path (``index_new_memory``): graph built, and not on the loop."""
    _user_id, memory_id = await _seed_memory(store)
    async with store.sessions() as db:
        memory = await db.get(Memory, memory_id)
        assert await write_back.index_new_memory(memory) is True  # embed leg landed

    metadata, entities, links, relations = _graph_state(memory_id)
    assert metadata.get("graph_extracted_at"), "graph build never ran on the API/MCP path"
    assert entities >= 1, "no entity row was written for the memory"
    assert links >= 1, "no memory_entity link was written for the memory"
    assert relations >= 1, "no relation row was written for the memory"
    assert store.calls, "extraction was never reached"
    assert all(call["thread"] != threading.get_ident() for call in store.calls), (
        "the graph build ran on the event loop; it must run in a worker thread"
    )


async def test_connector_sync_write_path_builds_the_graph(store):
    """The connector-sync path (``_index_memories``), driven from a live loop."""
    user_id, memory_id = await _seed_memory(store)
    async with store.sessions() as db:
        service = SourceSyncService(db)
        await service._index_memories([str(memory_id)], user_id=user_id)

    metadata, entities, links, relations = _graph_state(memory_id)
    assert metadata.get("graph_extracted_at"), "graph build never ran on the sync path"
    assert entities >= 1 and links >= 1 and relations >= 1


async def test_the_sync_builder_refuses_a_running_loop_with_the_fix(store):
    """The builder cannot be driven from the loop: the error names the fix.

    Pre-fix this raised the bare ``asyncio.run() cannot be called from a
    running event loop`` (and leaked a never-awaited coroutine); the point is
    that a future caller gets told exactly what to do instead of a mystery.
    """
    _user_id, memory_id = await _seed_memory(store)
    with sync_session() as db, pytest.raises(RuntimeError, match=r"asyncio\.to_thread"):
        build_memory_graph_sync(db, str(memory_id))


async def test_graph_failure_is_loud_and_never_fails_the_write(store, monkeypatch, caplog):
    """A failed graph build keeps the write alive and reaches the operator."""
    _user_id, memory_id = await _seed_memory(store)

    def boom(*_args, **_kwargs):
        raise RuntimeError("graph store down")

    monkeypatch.setattr("app.graph.builder.build_memory_graph_sync", boom)
    async with store.sessions() as db:
        memory = await db.get(Memory, memory_id)
        with caplog.at_level(logging.ERROR, logger="app.retrieval.memory.write_back"):
            assert await write_back.index_new_memory(memory) is True  # never fails the write

    loud = [
        record for record in caplog.records
        if record.levelno >= logging.ERROR and str(memory_id) in record.getMessage()
    ]
    assert loud, f"the graph failure was not loud:\n{caplog.text}"
