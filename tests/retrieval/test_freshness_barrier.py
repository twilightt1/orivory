"""Task 3 — the recall freshness barrier: wait for YOUR OWN writes, then fail loud.

Pinned contract:

(a) A committed memory with a pending index intent is NOT served as a no-match:
    recall waits (bounded) for the drain to land it and returns it.
(b) A vector outage leaves the intent pending → recall raises
    ``IndexFreshnessTimeout`` and the endpoint answers 503
    ``{"error": "index_freshness_timeout"}`` — never a ``200`` with ``[]``.
(c) Nothing pending → no drain, no wait (the barrier's near-zero cost).
(d) The trace carries ``queue_wait`` in MILLISECONDS, filled once, from the
    barrier's own measurement (never a fabricated 0).
(e) The event loop keeps ticking while the barrier waits (ruling R13: a task
    that counts its own turns, not a wall-clock threshold).

The harness is the drain-loop suite's ``env`` (private SQLite engines + a real
embedded Qdrant on ``tmp_path``, deterministic embeddings) and is IMPORTED
rather than copied, so both suites prove the same environment. Ruling R1 — the
barrier drains through ``drain_loop.drain_once`` and never claims the outbox
itself — is what the (c) spy patches: ``drain_once`` calls that global.
"""
from __future__ import annotations

import asyncio
import contextlib
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app import database
from app.config import settings
from app.database import Base, get_db
from app.main import app
from app.models.memory import Memory
from app.retrieval.memory import drain_loop, freshness, outbox
from app.retrieval.memory import retriever as retriever_module
from app.retrieval.memory.outbox import IndexFreshnessTimeout
from app.retrieval.memory.retriever import MemoryRetriever
from app.retrieval.vector_retriever import VectorUnavailableError
from app.utils.dependencies import enforce_llm_quota, get_current_verified_user
from tests.retrieval.test_drain_loop import (
    _activate_memory_manifest,
    _statuses,
    _vector_for,
)

# The drain-loop suite's ``env`` / ``owner`` fixtures, registered as a plugin
# rather than imported by name (an imported fixture name would be shadowed by
# the test parameters and ruff would read it as a redefinition).
pytest_plugins = ["tests.retrieval.test_drain_loop"]

CONTENT = "walked the chestnut orchard at dusk"


# ── the recall path's LLM / embedding seams (no external service) ───────────


async def _empty_context(*_args, **_kwargs):
    return []


async def _fallback_rewrite(query, context=None, **_kwargs):
    return {
        "rewritten_query": query,
        "entities": [],
        "reasoning": None,
        "_fallback_used": True,
    }


async def _query_embedding(_query: str) -> list[float]:
    """The query embeds as the memory's own document: cosine 1.0 once it landed."""
    return _vector_for(CONTENT)


async def _write_memory(sessions, user_id: uuid.UUID, content: str) -> uuid.UUID:
    """One committed memory + its pending upsert intent — the write face, no vector."""
    async with sessions() as db:
        memory = Memory(id=uuid.uuid4(), user_id=user_id, content=content, tags=[])
        db.add(memory)
        outbox.bump_revision(memory)
        await outbox.enqueue_upsert(db, memory)
        await db.commit()
        return memory.id


async def _recall(db, user_id: uuid.UUID, query: str = "where did i walk at dusk"):
    return await MemoryRetriever(db, user_id).recall(query, top_k=5)


@pytest.fixture(autouse=True)
def _barrier_reads_this_suites_own_outbox(env, monkeypatch):
    """Point the barrier's own session at the private SQLite ``env`` already wired.

    ``freshness`` binds ``AsyncSessionLocal`` at import, so the private
    sessionmaker must be patched into that namespace too (as the drain harness
    does for ``outbox``).
    """
    monkeypatch.setattr(freshness, "AsyncSessionLocal", database.AsyncSessionLocal)


# ── (a) a just-written memory is not a no-match ─────────────────────────────


async def test_a_recall_returns_a_just_written_memory_once_the_barrier_drains(env, owner, monkeypatch):
    """The write-through never ran: the barrier's drain is what puts it in the index."""
    monkeypatch.setattr(retriever_module, "fetch_personal_context", _empty_context)
    monkeypatch.setattr(retriever_module, "rewrite_query", _fallback_rewrite)
    monkeypatch.setattr(retriever_module, "embed_query", _query_embedding)
    await _activate_memory_manifest(env)
    memory_id = await _write_memory(env, owner, CONTENT)

    # Sanity: without the barrier's drain the index really is empty for this write.
    assert await _statuses() == ["pending"]

    async with env() as db:
        response = await _recall(db, owner)

    assert [str(m.id) for m in response.results] == [str(memory_id)]
    assert await _statuses() == ["done"]  # the barrier's drain landed the intent
    assert response.trace.stage_ms["queue_wait"] >= 0


# ── (b) an unlandable intent fails loud, typed, end to end ──────────────────


async def test_b_an_unlandable_intent_raises_index_freshness_timeout(env, owner, monkeypatch):
    """Vector store down: the intent cannot land → recall raises, never ``[]``."""
    monkeypatch.setattr(settings, "RECALL_FRESHNESS_BUDGET_SECONDS", 0.2)
    await _activate_memory_manifest(env)
    await _write_memory(env, owner, CONTENT)

    async def _store_down(_memory):
        raise VectorUnavailableError("Qdrant refused")

    monkeypatch.setattr(outbox, "upsert_memory", _store_down)

    async with env() as db:
        with pytest.raises(IndexFreshnessTimeout):
            await _recall(db, owner)

    # The drain's retry path ran for real: the intent is still the promise that
    # the vector is owed, with its backoff.
    assert await _statuses() == ["pending"]


@asynccontextmanager
async def _recall_client(tmp_path, monkeypatch, *, user_id: uuid.UUID):
    """App client on a private temp-SQLite DB, with auth/quota/DB seams overridden."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'barrier.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )
    for module in (database, outbox, freshness):
        monkeypatch.setattr(module, "AsyncSessionLocal", sessions)

    async def _db_override():
        async with sessions() as session:
            yield session

    app.dependency_overrides[get_current_verified_user] = lambda: SimpleNamespace(id=user_id)
    app.dependency_overrides[enforce_llm_quota] = lambda: None
    app.dependency_overrides[get_db] = _db_override
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client, sessions
    finally:
        app.dependency_overrides.clear()
        await engine.dispose()


async def test_b_the_endpoint_answers_503_index_freshness_timeout(tmp_path, monkeypatch):
    """(b) …and it reaches the API as a typed 503 body, not a 200 with ``[]``."""
    monkeypatch.setattr(settings, "RECALL_FRESHNESS_BUDGET_SECONDS", 0.15)
    owner_id = uuid.uuid4()

    async def _store_down(_memory):
        raise VectorUnavailableError("Qdrant refused")

    monkeypatch.setattr(outbox, "upsert_memory", _store_down)

    async with _recall_client(tmp_path, monkeypatch, user_id=owner_id) as (client, sessions):
        await _write_memory(sessions, owner_id, CONTENT)
        response = await client.post(
            "/api/v1/memories/recall", json={"query": "where did i walk at dusk"}
        )

    assert response.status_code == 503
    assert response.json() == {"error": "index_freshness_timeout"}


# ── (c) nothing pending: no drain, no wait ──────────────────────────────────


async def test_c_nothing_pending_never_drains_and_never_waits(env, owner, monkeypatch):
    calls: list[int] = []

    async def _spy(*, batch_size):
        calls.append(batch_size)
        raise AssertionError("the barrier drained with nothing pending")

    monkeypatch.setattr(drain_loop, "drain_pending", _spy)

    waited = await freshness.await_freshness(user_id=str(owner), timeout=5.0)

    assert calls == []
    assert 0 <= waited < 0.05  # one count query's cost, not a fabricated 0


async def test_c_a_recall_with_nothing_pending_records_its_near_zero_wait(env, owner, monkeypatch):
    """The trace's ``queue_wait`` is measured even on the no-pending fast path."""
    monkeypatch.setattr(retriever_module, "fetch_personal_context", _empty_context)
    monkeypatch.setattr(retriever_module, "rewrite_query", _fallback_rewrite)
    monkeypatch.setattr(retriever_module, "embed_query", _query_embedding)

    async with env() as db:
        response = await _recall(db, owner)

    assert response.results == []
    assert 0 <= response.trace.stage_ms["queue_wait"] < 50.0


# ── (d) the trace: milliseconds, once, from the barrier ─────────────────────


async def test_d_the_trace_carries_the_barriers_wait_in_milliseconds(env, owner, monkeypatch):
    """The value is the barrier's return (×1000) — a second write of 0.0 would fail here."""
    monkeypatch.setattr(retriever_module, "fetch_personal_context", _empty_context)
    monkeypatch.setattr(retriever_module, "rewrite_query", _fallback_rewrite)
    monkeypatch.setattr(retriever_module, "embed_query", _query_embedding)

    async def _barrier(**_kwargs):
        return 0.123

    monkeypatch.setattr(retriever_module, "await_freshness", _barrier)

    async with env() as db:
        response = await _recall(db, owner)

    assert response.trace.stage_ms["queue_wait"] == pytest.approx(123.0)
    assert list(response.trace.stage_ms).count("queue_wait") == 1


# ── (e) the loop keeps ticking while the barrier waits (R13) ────────────────


async def test_e_the_event_loop_keeps_ticking_while_the_barrier_waits(env, owner, monkeypatch):
    await _activate_memory_manifest(env)
    await _write_memory(env, owner, CONTENT)

    landed = outbox.upsert_memory  # the real store, slowed to give the loop a window

    async def _slow_store(memory):
        await asyncio.sleep(0.2)
        await landed(memory)

    monkeypatch.setattr(outbox, "upsert_memory", _slow_store)

    ticks = 0

    async def _ticker():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    ticker = asyncio.create_task(_ticker())
    try:
        waited = await freshness.await_freshness(user_id=str(owner), timeout=5.0)
    finally:
        ticker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ticker

    assert waited >= 0.15  # it really waited for the store to land the intent
    assert ticks >= 50  # …and the loop was free the whole time
    assert await _statuses() == ["done"]


# ── the queue itself: the barrier goes through the single-flight door (R1) ──


async def test_the_barrier_drains_through_drain_once_not_the_outbox(env, owner, monkeypatch):
    """R1: one claimer in-process — the barrier uses the loop's own door."""
    seen: list[int] = []
    real_once = drain_loop.drain_once

    async def _spy(*, batch_size):
        seen.append(batch_size)
        return await real_once(batch_size=batch_size)

    async def _outbox_door(*_args, **_kwargs):
        raise AssertionError("the barrier claimed the outbox directly")

    monkeypatch.setattr(drain_loop, "drain_once", _spy)
    monkeypatch.setattr(outbox, "drain_pending", _outbox_door)
    await _activate_memory_manifest(env)
    await _write_memory(env, owner, CONTENT)

    waited = await freshness.await_freshness(user_id=str(owner), timeout=5.0)

    assert seen == [settings.OUTBOX_DRAIN_BATCH_SIZE]
    assert waited > 0
    assert await _statuses() == ["done"]
