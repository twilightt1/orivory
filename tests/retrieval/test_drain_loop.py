"""P3 — the background drain loop: one claimer, both dialects, no boot blocker.

Isolated by construction (the ``tests/retrieval/test_chunk_index.py`` pattern):
a real embedded Qdrant on a private ``tmp_path`` folder — closed in teardown,
so its folder lock never leaks into another test — and a private per-test
SQLite file monkeypatched in as the module engines / sessionmakers, so this
suite can never read or write whatever ``DATABASE_URL`` is ambient.

Embeddings are deterministic unit vectors and the intents are applied by the
REAL drain against the REAL store (only the embedder and the boot timer are
pinned), so neither the loop nor the vector write face is mocked.
"""
from __future__ import annotations

import asyncio
import gc
import hashlib
import logging
import math
import random
import uuid

import pytest_asyncio
from sqlalchemy import create_engine, event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from app import database
from app import models as _models  # noqa: F401 — register every table on Base
from app.config import settings
from app.database import Base
from app.models.index_outbox import IndexGeneration, IndexOutbox
from app.models.memory import Memory
from app.models.user import User
from app.retrieval import vector_backend
from app.retrieval.embedding_fingerprint import canonical_fingerprint, fingerprint_generation
from app.retrieval.memory import drain_loop, outbox, vector_store
from app.retrieval.memory.vector_store import COLLECTION_NAME

DIM = 8
DRIFT_DB = "drain.sqlite"
# The embedding contract this suite pins (the ambient settings must not decide
# it: the same tests have to hold on a 384-dim lite install and a 1536-dim
# OpenAI one).
FINGERPRINT = {
    "model_id": "test-model",
    "model_revision": "revision-1",
    "dim": DIM,
    "provider": "test",
}


def _fingerprint() -> dict:
    return dict(FINGERPRINT)


def _expected_token() -> str:
    return fingerprint_generation(canonical_fingerprint(FINGERPRINT))


def _vector_for(text: str) -> list[float]:
    """Unit vector for a text: same text -> same vector, every run."""
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    raw = [rng.uniform(-1.0, 1.0) for _ in range(DIM)]
    norm = math.sqrt(sum(value * value for value in raw))
    return [value / norm for value in raw]


async def _fake_embed(texts: list[str]) -> list[list[float]]:
    return [_vector_for(text) for text in texts]


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def env(tmp_path, monkeypatch):
    """Real embedded Qdrant + private SQLite outbox, both on ``tmp_path``."""
    folder = tmp_path / "qdrant"
    folder.mkdir()
    monkeypatch.setattr(settings, "QDRANT_MODE", "local")
    monkeypatch.setattr(settings, "QDRANT_LOCAL_PATH", str(folder))

    url = f"sqlite+aiosqlite:///{tmp_path / DRIFT_DB}"
    engine = create_async_engine(
        url, connect_args={"check_same_thread": False}, poolclass=NullPool
    )
    event.listen(engine.sync_engine, "connect", database._configure_sqlite_connection)
    sync_engine = create_engine(
        url.replace("+aiosqlite", ""), connect_args={"check_same_thread": False}
    )
    event.listen(sync_engine, "connect", database._configure_sqlite_connection)
    sessions = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    monkeypatch.setattr(database, "AsyncSessionLocal", sessions)
    monkeypatch.setattr(outbox, "AsyncSessionLocal", sessions)
    monkeypatch.setattr(
        database,
        "_get_sync_sessionmaker",
        lambda: sessionmaker(bind=sync_engine, expire_on_commit=False, autoflush=False),
    )
    monkeypatch.setattr(vector_store, "embed_texts", _fake_embed)
    # Every guard must see the same contract: embedder binds the fingerprint at
    # import, so patch the alias each module actually calls.
    from app.retrieval import embedder as embedder_module
    from app.retrieval import embedding_fingerprint as fingerprint_module

    for module in (fingerprint_module, embedder_module, vector_store):
        monkeypatch.setattr(module, "current_fingerprint", _fingerprint)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield sessions
    finally:
        await vector_backend.close_clients()
        await engine.dispose()
        sync_engine.dispose()


@pytest_asyncio.fixture
async def owner(env) -> uuid.UUID:
    user_id = uuid.uuid4()
    async with env() as db:
        db.add(
            User(
                id=user_id,
                email=f"{user_id.hex}@test.invalid",
                hashed_password="x",
                display_name="Owner",
                is_verified=True,
                is_active=True,
            )
        )
        await db.commit()
    return user_id


async def _activate_memory_manifest(env, fingerprint: str | None = None) -> None:
    """Point the MEMORY manifest at the served generation (one active row)."""
    async with env() as db:
        db.add(
            IndexGeneration(
                id=uuid.uuid4().hex,
                kind=outbox.KIND_MEMORY,
                generation=outbox.TARGET_GENERATION,
                fingerprint=fingerprint or _expected_token(),
                is_active=True,
            )
        )
        await db.commit()


async def _enqueue(env, owner: uuid.UUID, count: int) -> list[uuid.UUID]:
    """``count`` committed memories, each with its pending upsert intent."""
    ids: list[uuid.UUID] = []
    async with env() as db:
        for i in range(count):
            memory = Memory(id=uuid.uuid4(), user_id=owner, content=f"memory {i}", tags=[])
            db.add(memory)
            outbox.bump_revision(memory)
            await outbox.enqueue_upsert(db, memory)
            ids.append(memory.id)
        await db.commit()
    return ids


async def _outbox_rows() -> list[IndexOutbox]:
    """Read the outbox through a fresh session (immune to snapshot staleness)."""
    async with database.AsyncSessionLocal() as session:
        return list(
            (await session.execute(select(IndexOutbox).order_by(IndexOutbox.seq))).scalars().all()
        )


async def _statuses() -> list[str]:
    return [row.status for row in await _outbox_rows()]


async def _until(predicate, *, timeout: float = 1.0) -> None:
    """Poll ``predicate()`` (sync or async) until it holds, or fail the test."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        result = predicate()
        if await result if asyncio.iscoroutine(result) else result:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"condition never held within {timeout}s")


class _LogCapture:
    """Records ``drain_loop.log`` calls (level, event, kwargs)."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict]] = []

    def info(self, event: str, **kw) -> None:
        self.events.append(("info", event, kw))

    def warning(self, event: str, **kw) -> None:
        self.events.append(("warning", event, kw))


# ── the loop runs, drains deep, and keeps going ─────────────────────────────


async def test_the_loop_applies_pending_intents_within_an_interval(env, owner, monkeypatch):
    """Three pending intents land in the real store, driven by the loop alone."""
    monkeypatch.setattr(settings, "OUTBOX_DRAIN_INTERVAL_SECONDS", 0.05)
    await _activate_memory_manifest(env)
    ids = await _enqueue(env, owner, 3)

    task = await drain_loop.start_drain_loop()
    assert task is not None
    try:
        await _until(lambda: _done(ids))
    finally:
        await drain_loop.stop_drain_loop(task)

    assert await _statuses() == ["done"] * 3
    client = vector_backend.get_async_client()
    assert (await client.count(COLLECTION_NAME)).count == 3  # the vectors are real


async def _done(ids: list[uuid.UUID]) -> bool:
    rows = {row.entity_id: row.status for row in await _outbox_rows()}
    return all(rows.get(memory_id.hex) == "done" for memory_id in ids)


async def test_a_productive_batch_drains_deep_without_waiting_the_interval(monkeypatch):
    """``applied > 0`` means more work is likely: the next batch goes immediately."""
    reports = [1, 0]
    calls: list[int] = []

    async def fake_drain(*, batch_size):
        calls.append(batch_size)
        return {
            "claimed": 0,
            "applied": reports.pop(0) if reports else 0,
            "skipped": 0,
            "blocked": 0,
            "failed": 0,
        }

    monkeypatch.setattr(drain_loop, "drain_pending", fake_drain)
    stop = asyncio.Event()
    task = asyncio.create_task(
        drain_loop.run_drain_loop(interval=30.0, batch_size=7, stop=stop)
    )
    try:
        await _until(lambda: len(calls) >= 2)
        assert not task.done()
    finally:
        stop.set()
        await asyncio.wait_for(task, timeout=5)
    assert calls == [7, 7]  # the batch size is forwarded, and no 30s wait between them


async def test_a_drain_exception_is_logged_and_the_loop_keeps_going(env, owner, monkeypatch):
    """The loop never raises out of itself: warn, then keep draining."""
    calls = {"n": 0}
    real_drain = outbox.drain_pending

    async def flaky(*, batch_size):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("outbox table is gone")
        return await real_drain(batch_size=batch_size)

    monkeypatch.setattr(drain_loop, "drain_pending", flaky)
    captured = _LogCapture()
    monkeypatch.setattr(drain_loop, "log", captured)
    await _activate_memory_manifest(env)
    ids = await _enqueue(env, owner, 1)

    stop = asyncio.Event()
    task = asyncio.create_task(
        drain_loop.run_drain_loop(interval=0.05, batch_size=50, stop=stop)
    )
    try:
        await _until(lambda: _done(ids))
        assert not task.done() and not task.cancelled()  # the failure did not kill it
    finally:
        stop.set()
        await asyncio.wait_for(task, timeout=5)

    assert captured.events[0][:2] == ("warning", "outbox drain failed")
    assert "outbox table is gone" in captured.events[0][2]["error"]
    # …and the round after it still reported its counts.
    assert ("info", "outbox drain") in [(level, event) for level, event, _ in captured.events]


async def test_stop_ends_the_task_within_the_bound(env, monkeypatch, caplog):
    """A clean stop: no pending task, no "Task was destroyed" from the loop."""
    monkeypatch.setattr(settings, "OUTBOX_DRAIN_INTERVAL_SECONDS", 0.05)
    caplog.set_level(logging.DEBUG, logger="asyncio")

    task = await drain_loop.start_drain_loop()
    assert task is not None
    loop = asyncio.get_running_loop()
    started = loop.time()
    await drain_loop.stop_drain_loop(task)

    assert loop.time() - started <= 5
    assert task.done() and not task.cancelled()
    assert [t for t in asyncio.all_tasks() if t.get_name() == "outbox-drain"] == []
    gc.collect()  # a destroyed-but-pending task would log here, on its finalizer
    await asyncio.sleep(0)
    assert "was destroyed" not in caplog.text


async def test_stop_cancels_a_drain_that_overruns_the_bound(monkeypatch):
    """A hung drain must not hold the shutdown past the 5s bound."""
    stuck = asyncio.Event()

    async def hung(*, batch_size):
        await stuck.wait()  # a vector call that never comes back

    monkeypatch.setattr(drain_loop, "drain_pending", hung)
    monkeypatch.setattr(drain_loop, "_STOP_TIMEOUT_SECONDS", 0.05)
    captured = _LogCapture()
    monkeypatch.setattr(drain_loop, "log", captured)
    monkeypatch.setattr(settings, "OUTBOX_DRAIN_INTERVAL_SECONDS", 0.05)

    task = await drain_loop.start_drain_loop()
    assert task is not None
    await asyncio.sleep(0.02)  # let it enter the hung batch
    await drain_loop.stop_drain_loop(task)

    assert task.cancelled()
    assert ("warning", "Outbox drain loop did not stop in time; cancelling") in [
        (level, event) for level, event, _ in captured.events
    ]


# ── both dialects (ruling R4: no SQLite-only gate) ──────────────────────────


async def test_the_loop_runs_on_a_postgres_shaped_url(env, owner, monkeypatch):
    """The boot drain's ``startswith("sqlite")`` gate is gone: P3 drains everywhere."""
    assert drain_loop._should_drain() is True  # the sqlite-shaped URL of this suite
    monkeypatch.setattr(
        settings, "DATABASE_URL", "postgresql+asyncpg://user:pw@db/orivory"
    )
    assert drain_loop._should_drain() is True  # …and a Postgres-shaped one

    monkeypatch.setattr(settings, "OUTBOX_DRAIN_INTERVAL_SECONDS", 0.05)
    await _activate_memory_manifest(env)
    ids = await _enqueue(env, owner, 1)

    task = await drain_loop.start_drain_loop()
    assert task is not None
    try:
        await _until(lambda: _done(ids))
    finally:
        await drain_loop.stop_drain_loop(task)
    assert await _statuses() == ["done"]


async def test_a_disabled_drain_starts_no_task(env, monkeypatch):
    """``OUTBOX_DRAIN_ENABLED=false``: the lifespan creates no task at all."""
    monkeypatch.setattr(settings, "OUTBOX_DRAIN_ENABLED", False)
    assert await drain_loop.start_drain_loop() is None
    await drain_loop.stop_drain_loop(None)  # the lifespan's finally stays a no-op


# ── one claimer (ruling R1), across the loops a process may see ─────────────


def test_drain_once_is_single_flight_across_event_loops(monkeypatch):
    """Concurrent claimers serialize; a fresh event loop drains again.

    ``asyncio.Lock`` binds to the first loop that contends it, so the second
    ``asyncio.run`` here would die with "bound to a different event loop" if
    the module lock outlived its loop (every test, every re-entered runner).
    """
    live = 0
    peak = 0

    async def slow_drain(*, batch_size):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.05)
        live -= 1
        return {"claimed": 0, "applied": 0, "skipped": 0, "blocked": 0, "failed": 0}

    monkeypatch.setattr(drain_loop, "drain_pending", slow_drain)

    async def two_claimers():
        await asyncio.gather(
            drain_loop.drain_once(batch_size=1), drain_loop.drain_once(batch_size=1)
        )

    asyncio.run(two_claimers())
    asyncio.run(two_claimers())  # a second loop still gets exactly one claimer
    assert peak == 1
