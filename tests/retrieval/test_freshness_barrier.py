"""Task 3 — the recall freshness barrier: wait for YOUR OWN writes, then fail loud.

Pinned contract:

(a) A committed memory with a pending index intent is NOT served as a no-match:
    recall waits (bounded) for the drain to land it and returns it.
(b) A vector outage leaves the intent pending → recall raises
    ``IndexFreshnessTimeout`` and the endpoint answers 503
    ``{"error": "index_freshness_timeout"}`` — never a ``200`` with ``[]``.
    Same for an unreadable queue (count read raises), a drain that raises, and
    a drain that hangs: the barrier FAILS CLOSED (ruling R14) with the budget
    as a hard deadline over the drain, not only over the loop's own sleeps.
(c) Nothing pending → no drain, no wait (the barrier's near-zero cost). Only
    this tenant's MEMORY intents count: an untended chunk backlog (a bulk
    import) must not 503 memory recall.
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
import time
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
from app.models.index_outbox import IndexOutbox
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


async def test_b_the_endpoint_never_answers_200_when_the_queue_is_unreadable(tmp_path, monkeypatch):
    """R14 end to end: a queue the barrier cannot read is a typed 503, never ``[]``."""
    monkeypatch.setattr(settings, "RECALL_FRESHNESS_BUDGET_SECONDS", 0.1)

    async def _boom(_tenant):
        raise RuntimeError("outbox unreadable")

    monkeypatch.setattr(freshness, "_pending_count", _boom)

    async with _recall_client(tmp_path, monkeypatch, user_id=uuid.uuid4()) as (client, _sessions):
        response = await client.post(
            "/api/v1/memories/recall", json={"query": "where did i walk at dusk"}
        )

    assert response.status_code == 503
    assert response.json() == {"error": "index_freshness_timeout"}


# ── (b) fail closed: an UNPROVEN write is not a no-match (R14) ──────────────


async def test_b_an_unreadable_outbox_retries_then_fails_closed(env, owner, monkeypatch):
    """The count read raises every round: warn and retry the budget out, then raise.

    The old swallow returned after ~9 ms of a 2 s budget with the intent still
    pending; the retry count pins that a transient read failure is neither
    fatal NOR an early return.
    """
    calls: list[str] = []

    async def _boom(tenant):
        calls.append(tenant)
        raise RuntimeError("outbox unreadable")

    monkeypatch.setattr(freshness, "_pending_count", _boom)

    t0 = time.perf_counter()
    with pytest.raises(IndexFreshnessTimeout):
        await freshness.await_freshness(user_id=str(owner), timeout=0.2, poll=0.02)
    elapsed = time.perf_counter() - t0

    assert len(calls) >= 2  # it retried, it did not give up on the first failure
    assert elapsed >= 0.2  # the budget was used, not skipped
    assert elapsed < 0.6  # …and the deadline held


async def test_b_a_drain_error_is_retried_until_the_intent_lands(env, owner, monkeypatch):
    """A transient drain failure → the loop retries, the write lands, recall proceeds."""
    await _activate_memory_manifest(env)
    await _write_memory(env, owner, CONTENT)

    real_once = drain_loop.drain_once
    calls: list[int] = []

    async def _flaky(*, batch_size):
        calls.append(batch_size)
        if len(calls) == 1:
            raise RuntimeError("drain hiccup")
        return await real_once(batch_size=batch_size)

    monkeypatch.setattr(drain_loop, "drain_once", _flaky)

    waited = await freshness.await_freshness(user_id=str(owner), timeout=5.0)

    assert len(calls) >= 2  # the error was retried, not swallowed into "fresh"
    assert waited > 0
    assert await _statuses() == ["done"]


async def test_b_a_drain_error_with_a_pending_intent_never_answers_empty(env, owner, monkeypatch):
    """Every drain raises and the intent persists → typed timeout, no ``results: []``."""
    monkeypatch.setattr(settings, "RECALL_FRESHNESS_BUDGET_SECONDS", 0.2)
    await _activate_memory_manifest(env)
    await _write_memory(env, owner, CONTENT)

    calls: list[int] = []

    async def _boom(*, batch_size):
        calls.append(batch_size)
        raise RuntimeError("drain exploded")

    monkeypatch.setattr(drain_loop, "drain_once", _boom)

    t0 = time.perf_counter()
    async with env() as db:
        with pytest.raises(IndexFreshnessTimeout):
            await _recall(db, owner)
    elapsed = time.perf_counter() - t0

    assert len(calls) >= 2
    assert elapsed >= 0.2  # it waited the budget (the 9.5 ms give-up cannot pass)
    assert elapsed < 0.6  # …bounded by the deadline
    assert await _statuses() == ["pending"]  # nothing acked away


async def test_b_a_hung_drain_cannot_outlive_the_budget(env, owner, monkeypatch):
    """I2: the budget is a HARD deadline over the drain, not just over the sleeps.

    Local Qdrant mode has no client timeout, so a stalled drain used to hang
    recall (or overshoot the budget ~5×) — ``wait_for`` on the remaining budget
    makes it a typed timeout instead.
    """
    await _activate_memory_manifest(env)
    await _write_memory(env, owner, CONTENT)

    async def _hang(*, batch_size):
        await asyncio.Event().wait()  # never returns

    monkeypatch.setattr(drain_loop, "drain_once", _hang)

    t0 = time.perf_counter()
    with pytest.raises(IndexFreshnessTimeout):
        await freshness.await_freshness(user_id=str(owner), timeout=0.2)
    elapsed = time.perf_counter() - t0

    assert elapsed >= 0.2
    assert elapsed < 1.0  # bounded by the budget, not by the store


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


async def test_c_the_guarantee_off_switch_skips_the_barrier(env, owner, monkeypatch):
    """``OUTBOX_DRAIN_ENABLED=false``: the barrier neither waits nor drains.

    A quiesced deployment keeps intents pending on purpose (a cutover owns the
    store), so waiting could only end in a spurious 503. The pending intent is
    asserted, so the skip cannot pass by having nothing to wait for.
    """
    await _activate_memory_manifest(env)
    await _write_memory(env, owner, CONTENT)
    assert await _statuses() == ["pending"]

    monkeypatch.setattr(settings, "OUTBOX_DRAIN_ENABLED", False)
    calls: list[int] = []

    async def _spy(*, batch_size):
        calls.append(batch_size)
        raise AssertionError("a quiesced deployment must not drain from the read path")

    monkeypatch.setattr(drain_loop, "drain_once", _spy)

    waited = await freshness.await_freshness(user_id=str(owner), timeout=0.05)

    assert calls == []
    assert waited < 0.05
    assert await _statuses() == ["pending"]


async def test_c_a_chunk_backlog_does_not_hold_up_memory_recall(env, owner, monkeypatch):
    """The count is scoped to MEMORY intents: a busy chunk import is not this
    read path's business (memory recall only reads memory vectors)."""
    async with env() as db:
        db.add(
            IndexOutbox(
                kind=outbox.KIND_CHUNK,
                entity_id=uuid.uuid4().hex,
                tenant_id=owner.hex,
                revision=0,
                operation=outbox.OPERATION_DELETE,
                target_generation=outbox.CHUNK_TARGET_GENERATION,
                status="pending",
            )
        )
        await db.commit()
    assert await _statuses() == ["pending"]  # there IS a pending intent — of another kind

    calls: list[int] = []

    async def _spy(*, batch_size):
        calls.append(batch_size)
        raise AssertionError("a chunk backlog is not a memory-index staleness")

    monkeypatch.setattr(drain_loop, "drain_once", _spy)

    waited = await freshness.await_freshness(user_id=str(owner), timeout=0.05)

    assert calls == []
    assert waited < 0.05


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
    """R1: one claimer in-process — the barrier uses the loop's own door.

    The pin is the spy on ``drain_loop.drain_once`` (the door ``freshness``
    calls): exactly one claim, with the batch size, and it is the door that
    did the work (the intent landed). ``freshness`` holds no other claim path
    to patch — it does not even import the outbox's claim function.
    """
    seen: list[int] = []
    real_once = drain_loop.drain_once

    async def _spy(*, batch_size):
        seen.append(batch_size)
        return await real_once(batch_size=batch_size)

    monkeypatch.setattr(drain_loop, "drain_once", _spy)
    await _activate_memory_manifest(env)
    await _write_memory(env, owner, CONTENT)

    waited = await freshness.await_freshness(user_id=str(owner), timeout=5.0)

    assert seen == [settings.OUTBOX_DRAIN_BATCH_SIZE]
    assert waited > 0
    assert await _statuses() == ["done"]
