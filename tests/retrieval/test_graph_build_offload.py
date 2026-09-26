"""P2/T9 — the graph build must really run on the memory write paths.

The write path that creates a ``Memory`` ends in
``write_back.safe_enqueue_graph_build``: the API/MCP path through
``index_new_memory`` (shared with the importer). Driving that helper *on the event loop*
left the build silently dead — the sync builder drives the extraction with
``asyncio.run`` (``app/graph/builder.py``), which raises inside a running loop,
and the helper's best-effort ``except`` swallowed the ``RuntimeError``: the
memory row committed, ``graph_extracted_at`` stayed unset, and no entity or
relation row was ever written for it.

Nothing about the stores or the builder is faked here: a private SQLite file
bound as the app's own (the P1b gate's ``_bind_engines``), the real
``build_memory_graph_sync``, and the real deterministic extraction fallback.
Two seams are substituted only because no claim in this file is about them:
the LLM provider (``app.graph.extraction._get_client`` — it raises in the
graph-build tests, so extraction takes its offline fallback, and it counts
in-flight creates in the concurrency test) and the embed leg (T1 owns that one,
with the real embedder, in ``test_event_loop_responsiveness.py``).

R27(p2) took the last ``await`` off the write path: ``safe_enqueue_graph_build``
is a plain function now — on a running loop it SCHEDULES the build as a
background task (``asyncio.to_thread`` held in a strong-ref set, failures
reported by the done callback), so the write returns immediately; with no loop
it still runs the build inline. ``_settle_graph_builds`` below is how a test
waits for those background builds; a build cut off at process exit is an
accepted best-effort loss.
"""
from __future__ import annotations

import asyncio
import gc
import json
import logging
import threading
import uuid
import warnings
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import select

from app import database
from app.agents import llm_client
from app.config import settings
from app.database import sync_session
from app.graph.builder import build_memory_graph_sync
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


async def _settle_graph_builds() -> None:
    """Wait for every graph build the write paths scheduled on THIS loop.

    The write path no longer awaits the build (R27(p2)), so a test that asserts
    on graph state must drain the helper's own strong-ref set first — exactly
    the tasks the production loop would be running.
    """
    while write_back._pending_graph_builds:
        await asyncio.gather(*tuple(write_back._pending_graph_builds),
                             return_exceptions=True)


async def test_api_and_mcp_write_path_builds_the_graph_off_the_loop(store):
    """The API/MCP path (``index_new_memory``): graph built, and not on the loop."""
    _user_id, memory_id = await _seed_memory(store)
    async with store.sessions() as db:
        memory = await db.get(Memory, memory_id)
        assert await write_back.index_new_memory(memory) is True  # embed leg landed
    await _settle_graph_builds()

    metadata, entities, links, relations = _graph_state(memory_id)
    assert metadata.get("graph_extracted_at"), "graph build never ran on the API/MCP path"
    assert entities >= 1, "no entity row was written for the memory"
    assert links >= 1, "no memory_entity link was written for the memory"
    assert relations >= 1, "no relation row was written for the memory"
    assert store.calls, "extraction was never reached"
    assert all(call["thread"] != threading.get_ident() for call in store.calls), (
        "the graph build ran on the event loop; it must run in a worker thread"
    )


async def test_the_sync_builder_refuses_a_running_loop_with_the_fix(store):
    """The builder cannot be driven from the loop: the error names the fix.

    Pre-fix this raised the bare ``asyncio.run() cannot be called from a
    running event loop`` (and leaked a never-awaited coroutine); the point is
    that a future caller gets told exactly what to do instead of a mystery,
    and that the refused coroutine is closed — not left to raise
    ``RuntimeWarning: coroutine 'extract_entities' was never awaited``.
    """
    _user_id, memory_id = await _seed_memory(store)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with sync_session() as db, pytest.raises(RuntimeError, match=r"asyncio\.to_thread"):
            build_memory_graph_sync(db, str(memory_id))
        gc.collect()
    leaks = [str(record.message) for record in caught if "never awaited" in str(record.message)]
    assert not leaks, f"the refused coroutine was never awaited: {leaks}"


async def test_the_write_returns_while_a_slow_build_is_still_in_flight(store, monkeypatch):
    """R27(p2): the write path never waits for the extraction.

    The fake build blocks on an Event this test controls, so the assertion is
    not "the write was fast" but "the build was STILL RUNNING when the write
    returned" — the exact shape a slow (free-tier) provider produces.
    """
    _user_id, memory_id = await _seed_memory(store)
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def _slow_build(_db, _memory_id):
        started.set()
        release.wait(5)
        finished.set()

    monkeypatch.setattr("app.graph.builder.build_memory_graph_sync", _slow_build)

    async with store.sessions() as db:
        memory = await db.get(Memory, memory_id)
        assert await write_back.index_new_memory(memory) is True

    assert not finished.is_set(), "the write path waited for the graph build"
    assert await asyncio.to_thread(started.wait, 5), "the build never started"
    release.set()
    await _settle_graph_builds()
    assert finished.is_set(), "the released build never landed"


async def test_no_loop_context_runs_the_build_inline(store):
    """R27(p2): without a running loop (CLI / script / worker) nothing is scheduled.

    A worker thread has no loop to schedule on, so the helper keeps its
    historical synchronous shape: when the thread returns, the graph is built
    and nothing is left pending.
    """
    _user_id, memory_id = await _seed_memory(store)

    await asyncio.to_thread(write_back.safe_enqueue_graph_build, memory_id)

    assert not write_back._pending_graph_builds, (
        "the no-loop path scheduled a task instead of running the build"
    )
    metadata, entities, links, relations = _graph_state(memory_id)
    assert metadata.get("graph_extracted_at"), "the inline build never ran"
    assert entities >= 1 and links >= 1 and relations >= 1


async def test_graph_failure_is_loud_and_never_fails_the_write(store, monkeypatch, caplog):
    """A failed background graph build keeps the write alive and reaches the operator.

    The write path does not await the build (R27(p2)), so the ERROR must come
    from the task's own done callback — and it must carry the traceback.
    """
    _user_id, memory_id = await _seed_memory(store)

    def boom(*_args, **_kwargs):
        raise RuntimeError("graph store down")

    monkeypatch.setattr("app.graph.builder.build_memory_graph_sync", boom)
    async with store.sessions() as db:
        memory = await db.get(Memory, memory_id)
        with caplog.at_level(logging.ERROR, logger="app.retrieval.memory.write_back"):
            assert await write_back.index_new_memory(memory) is True  # never fails the write
            await _settle_graph_builds()

    loud = [
        record for record in caplog.records
        if record.levelno >= logging.ERROR and str(memory_id) in record.getMessage()
    ]
    assert loud, f"the graph failure was not loud:\n{caplog.text}"
    for record in loud:
        assert record.exc_info is not None, "the graph failure was logged without its traceback"
        assert "graph store down" in str(record.exc_info[1]), "the traceback lost the cause"


async def test_concurrent_writes_never_exceed_the_llm_concurrency_limit(store, monkeypatch):
    """N>limit memories written at once: the shared gate caps the provider burst.

    The write path SCHEDULES one full extraction per written memory
    (the write no longer awaits it, R27(p2)) and extraction used to call
    ``chat.completions.create`` outside ``LLM_MAX_CONCURRENCY`` entirely
    (P2/T9 fix round 1, I1: the gate only wrapped ``complete()`` /
    ``complete_stream()``). The gate now lives on the shared client's wrapper,
    so the fake provider is wrapped the way the app builds its real client —
    an unwrapped fake would bypass the very policy under test. The write path
    is exercised for real, and the counting create must never see more calls
    in flight than the gate allows.
    """
    limit = 2
    monkeypatch.setattr(settings, "LLM_MAX_CONCURRENCY", limit)
    monkeypatch.setattr(llm_client, "_llm_semaphore", None, raising=False)

    state = {"inflight": 0, "peak": 0, "calls": 0}
    counter_lock = threading.Lock()

    async def _create(**_kwargs):
        with counter_lock:
            state["calls"] += 1
            state["inflight"] += 1
            state["peak"] = max(state["peak"], state["inflight"])
        try:
            await asyncio.sleep(0.05)  # hold the slot long enough to overlap
        finally:
            with counter_lock:
                state["inflight"] -= 1
        payload = {
            "entities": [
                {"name": "Project Atlas", "type": "project"},
                {"name": "lantern walk", "type": "event"},
            ],
            "relations": [
                {"source": "Project Atlas", "target": "lantern walk", "relation": "related_to"},
            ],
        }
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
        )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )
    monkeypatch.setattr(
        "app.graph.extraction._get_client",
        lambda: llm_client.ResilientAsyncOpenAI(fake_client),
    )

    seeds = [await _seed_memory(store) for _ in range(limit * 3)]

    async def _write(_user_id, memory_id):
        async with store.sessions() as db:
            memory = await db.get(Memory, memory_id)
            await write_back.index_new_memory(memory)

    await asyncio.gather(*(_write(user_id, memory_id) for user_id, memory_id in seeds))
    # The writes no longer await the builds (R27(p2)): drain the scheduled
    # tasks so the peak/landing assertions describe the completed work.
    await _settle_graph_builds()

    assert state["peak"] <= limit, (
        f"extraction burst the provider: {state['peak']} concurrent creates "
        f"with LLM_MAX_CONCURRENCY={limit}"
    )
    assert state["calls"] >= limit, "the gate starved the provider: extraction never ran"
    for _user_id, memory_id in seeds:
        metadata, _entities, _links, _relations = _graph_state(memory_id)
        assert metadata.get("graph_extracted_at"), f"memory {memory_id} never got its graph"
