"""Outbox: same-transaction intent, idempotent unique key, stale-skip, blocked-on-mismatch.

Isolated by construction: the fixture builds a private SQLite file on the
test's ``tmp_path`` and monkeypatches it in as the module engine / async
sessionmaker / sync sessionmaker, so this suite can never read or write
whatever ``DATABASE_URL`` is ambient. (It used to DELETE every row of
``index_outbox`` / ``index_generations`` / ``memories`` / ``users`` in the
ambient DB — run against a developer's real lite install that destroyed
memories.) Real file, real sessions, real drain: only the engines are swapped
(pattern: ``tests/lite/test_sqlite_schema_v2.py``).
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import create_engine, event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

import app.main as main
from app import database
from app.config import settings
from app.database import Base, sync_session
from app.models.index_outbox import IndexGeneration, IndexOutbox
from app.models.memory import Memory
from app.models.user import User
from app.retrieval.embedder import EmbeddingDimensionMismatch
from app.retrieval.memory import outbox
from app.retrieval.memory.vector_store import COLLECTION_NAME

OUTBOX_DB = "outbox.sqlite"


def _sync_engine(url: str):
    """Sync twin of the temp engine (Celery/CLI face), same file + pragmas."""
    eng = create_engine(url.replace("+aiosqlite", ""), connect_args={"check_same_thread": False})
    event.listen(eng, "connect", database._configure_sqlite_connection)
    return eng


def _open_engines(url: str, monkeypatch):
    """Fresh async+sync engines for ``url``, bound as the app's own.

    Called by the fixture and by the restart test — a restart is exactly this:
    new engines over the same committed SQLite file.
    """
    eng = create_async_engine(url, poolclass=NullPool)
    event.listen(eng.sync_engine, "connect", database._configure_sqlite_connection)
    sessions = async_sessionmaker(eng, class_=AsyncSession, expire_on_commit=False, autoflush=False)
    sync_eng = _sync_engine(url)
    monkeypatch.setattr(database, "engine", eng)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    monkeypatch.setattr(database, "AsyncSessionLocal", sessions)
    monkeypatch.setattr(outbox, "AsyncSessionLocal", sessions)  # the drain's own sessionmaker
    monkeypatch.setattr(
        database, "_get_sync_sessionmaker",
        lambda: sessionmaker(bind=sync_eng, expire_on_commit=False, autoflush=False),
    )
    return eng, sync_eng, sessions


@pytest_asyncio.fixture
async def db(tmp_path, monkeypatch):
    """A private per-test SQLite file — nothing here can reach an ambient DB."""
    url = f"sqlite+aiosqlite:///{tmp_path / OUTBOX_DB}"
    eng, sync_eng, sessions = _open_engines(url, monkeypatch)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with sessions() as session:
        yield session
    await eng.dispose()
    sync_eng.dispose()


@pytest_asyncio.fixture
async def owner(db) -> uuid.UUID:
    """A user row — ``memories.user_id`` is a FK."""
    uid = uuid.uuid4()
    db.add(User(id=uid, email=f"{uid.hex}@test.invalid", hashed_password="x",
                display_name="Owner", is_verified=True, is_active=True))
    await db.commit()
    return uid


def _memory(owner: uuid.UUID, content: str = "x") -> Memory:
    return Memory(id=uuid.uuid4(), user_id=owner, content=content, tags=[])


async def _outbox_rows() -> list[IndexOutbox]:
    """Read the outbox through a fresh session (immune to snapshot staleness)."""
    async with database.AsyncSessionLocal() as session:
        return list((await session.execute(select(IndexOutbox).order_by(IndexOutbox.seq))).scalars().all())


async def _memory_row(memory_id: uuid.UUID) -> Memory | None:
    async with database.AsyncSessionLocal() as session:
        return await session.get(Memory, memory_id)


# ── isolation: the module can only ever touch its own temp file ─────────────


async def test_module_engine_points_at_a_private_temp_file(db, tmp_path):
    assert database.engine.url.database == str(tmp_path / OUTBOX_DB)
    # The drain's sessionmaker is the patched one, bound to that same engine.
    assert outbox.AsyncSessionLocal is database.AsyncSessionLocal
    assert outbox.AsyncSessionLocal.kw["bind"] is database.engine
    async with outbox.AsyncSessionLocal() as session:
        assert session.get_bind() is database.engine.sync_engine


# ── enqueue: same transaction, idempotent key ───────────────────────────────


async def test_enqueue_rolls_back_with_the_row(db, owner):
    memory = _memory(owner)
    db.add(memory)
    outbox.bump_revision(memory)
    await outbox.enqueue_upsert(db, memory)
    await db.rollback()

    assert await _outbox_rows() == []
    assert await _memory_row(memory.id) is None


async def test_enqueue_without_a_bumped_revision_fails_loudly(db, owner):
    """A guessed revision is a silently lost index write — never guess."""
    memory = _memory(owner)  # unsaved: the revision column default lands at flush
    db.add(memory)
    with pytest.raises(ValueError, match=str(memory.id)) as excinfo:
        await outbox.enqueue_upsert(db, memory)
    assert "bump_revision" in str(excinfo.value)


def test_sync_enqueue_without_a_bumped_revision_fails_loudly(db, owner):
    memory = _memory(owner)
    with sync_session() as sync_db, pytest.raises(ValueError, match=str(memory.id)):
        outbox.enqueue_upsert_sync(sync_db, memory)


async def test_upsert_intent_is_idempotent_per_revision(db, owner):
    memory = _memory(owner)
    db.add(memory)
    revision = outbox.bump_revision(memory)
    await outbox.enqueue_upsert(db, memory)
    await outbox.enqueue_upsert(db, memory)  # replay of the same logical write
    await db.commit()

    rows = await _outbox_rows()
    assert len(rows) == 1
    row = rows[0]
    assert (row.kind, row.operation, row.revision) == ("memory", "upsert", revision)
    assert row.entity_id == memory.id.hex
    assert row.tenant_id == owner.hex
    assert row.target_generation == outbox.TARGET_GENERATION
    assert row.status == "pending" and row.attempts == 0

    # A delete at the same revision is a different intent, not a duplicate.
    await outbox.enqueue_delete(db, entity_id=memory.id.hex, tenant_id=owner.hex, revision=revision)
    await outbox.enqueue_delete(db, entity_id=memory.id.hex, tenant_id=owner.hex, revision=revision)
    await db.commit()

    rows = await _outbox_rows()
    assert sorted(row.operation for row in rows) == ["delete", "upsert"]


def test_target_generation_matches_the_seeded_collection():
    # Controller ruling: the fallback spelling IS what the ladder seeds.
    assert outbox.TARGET_GENERATION == COLLECTION_NAME


async def test_target_generation_reads_the_active_generation_row(db, owner):
    memory = _memory(owner)
    db.add(memory)
    outbox.bump_revision(memory)
    db.add(IndexGeneration(id=uuid.uuid4().hex, kind="memory", generation="Orivory_memories_old",
                           fingerprint="f" * 64, is_active=False))
    await outbox.enqueue_upsert(db, memory)
    await db.commit()
    # Inactive manifest rows are ignored: the transitional generation is current.
    assert (await _outbox_rows())[0].target_generation == outbox.TARGET_GENERATION

    db.add(IndexGeneration(id=uuid.uuid4().hex, kind="memory", generation="Orivory_memories_next",
                           fingerprint="f" * 64, is_active=True))
    await db.commit()
    other = _memory(owner, content="y")
    db.add(other)
    outbox.bump_revision(other)
    await db.commit()
    # A new session (a new request) must see the swapped manifest: the read is
    # memoized per session only.
    async with database.AsyncSessionLocal() as fresh:
        await outbox.enqueue_upsert(fresh, other)
        await fresh.commit()

    by_entity = {row.entity_id: row.target_generation for row in await _outbox_rows()}
    assert by_entity[other.id.hex] == "Orivory_memories_next"


def test_sync_enqueue_uses_the_same_intent_key(db, owner):
    memory = _memory(owner)
    with sync_session() as sync_db:
        sync_db.add(memory)
        outbox.bump_revision(memory)
        outbox.enqueue_upsert_sync(sync_db, memory)
        outbox.enqueue_delete_sync(sync_db, entity_id=memory.id.hex, tenant_id=owner.hex, revision=1)
        sync_db.commit()

    with sync_session() as sync_db:
        rows = sync_db.execute(select(IndexOutbox).order_by(IndexOutbox.seq)).scalars().all()
    assert [(row.operation, row.revision, row.entity_id, row.tenant_id) for row in rows] == [
        ("upsert", 1, memory.id.hex, owner.hex),
        ("delete", 1, memory.id.hex, owner.hex),
    ]
    assert {row.target_generation for row in rows} == {outbox.TARGET_GENERATION}


# ── drain: latest SQL state, stale-skip, terminal blocked, backoff ───────────


async def test_drain_skips_stale_revision_and_applies_latest(db, owner, monkeypatch):
    upserts: list[tuple[str, str, int]] = []

    async def record(memory):
        upserts.append((str(memory.id), memory.content, memory.revision))

    monkeypatch.setattr(outbox, "upsert_memory", record)

    memory = _memory(owner, content="v1")
    db.add(memory)
    outbox.bump_revision(memory)
    await outbox.enqueue_upsert(db, memory)
    await db.commit()
    memory.content = "v2"
    outbox.bump_revision(memory)
    await outbox.enqueue_upsert(db, memory)
    await db.commit()

    report = await outbox.drain_pending()

    assert report == {"claimed": 2, "applied": 1, "skipped": 1, "blocked": 0, "failed": 0}
    assert upserts == [(str(memory.id), "v2", 2)]  # the latest SQL state, once
    assert [row.status for row in await _outbox_rows()] == ["done", "done"]


async def test_freshness_drain_prefetches_local_embeddings_with_bounded_concurrency(
    db, owner, monkeypatch
):
    monkeypatch.setattr(settings, "USE_LOCAL_EMBEDDINGS", True)
    monkeypatch.setattr(settings, "EMBED_EXECUTOR_WORKERS", 2)
    active = 0
    max_active = 0
    written: list[tuple[str, list[float] | None]] = []

    async def embed(memory):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return [float(memory.revision)]

    async def upsert(memory, *, embedding=None):
        written.append((str(memory.id), embedding))

    monkeypatch.setattr(outbox, "embed_memory", embed)
    monkeypatch.setattr(outbox, "upsert_memory", upsert)

    foreign_id = uuid.uuid4()
    db.add(
        User(
            id=foreign_id,
            email=f"{foreign_id.hex}@foreign.test.invalid",
            hashed_password="x",
            display_name="Foreign",
            is_verified=True,
            is_active=True,
        )
    )
    await db.commit()

    priority_memories = [_memory(owner, f"parallel-{index}") for index in range(4)]
    foreign_memories = [_memory(foreign_id, name) for name in ("foreign-tagged", "wrong-owner")]
    memories = [*priority_memories, *foreign_memories]
    for memory in memories:
        db.add(memory)
        outbox.bump_revision(memory)
        await outbox.enqueue_upsert(db, memory)
        if memory is foreign_memories[1]:
            queued = (
                await db.execute(select(IndexOutbox).where(IndexOutbox.entity_id == memory.id.hex))
            ).scalar_one()
            queued.tenant_id = owner.hex
    await db.commit()

    report = await outbox.drain_pending(priority_tenant=owner.hex)

    assert report == {"claimed": 6, "applied": 6, "skipped": 0, "blocked": 0, "failed": 0}
    assert max_active == 2
    assert {memory_id for memory_id, _ in written} == {str(memory.id) for memory in memories}
    assert {memory_id for memory_id, embedding in written if embedding is not None} == {
        str(memory.id) for memory in priority_memories
    }


async def test_freshness_drain_falls_back_to_serial_when_prefetch_fails(
    db, owner, monkeypatch
):
    monkeypatch.setattr(settings, "USE_LOCAL_EMBEDDINGS", True)
    monkeypatch.setattr(settings, "EMBED_EXECUTOR_WORKERS", 2)
    written: list[str] = []
    prefetch_attempted = False

    async def fail_prefetch(_rows, *, priority_tenant):
        nonlocal prefetch_attempted
        prefetch_attempted = True
        assert priority_tenant == owner.hex
        raise RuntimeError("snapshot read unavailable")

    async def upsert(memory):
        written.append(str(memory.id))

    monkeypatch.setattr(outbox, "_prefetch_memory_embeddings", fail_prefetch)
    monkeypatch.setattr(outbox, "upsert_memory", upsert)

    memories = [_memory(owner, f"fallback-{index}") for index in range(2)]
    for memory in memories:
        db.add(memory)
        outbox.bump_revision(memory)
        await outbox.enqueue_upsert(db, memory)
    await db.commit()

    report = await outbox.drain_pending(priority_tenant=owner.hex)

    assert prefetch_attempted
    assert report["applied"] == 2
    assert set(written) == {str(memory.id) for memory in memories}


async def test_freshness_drain_retries_failed_prefetch_serially(db, owner, monkeypatch):
    monkeypatch.setattr(settings, "USE_LOCAL_EMBEDDINGS", True)
    monkeypatch.setattr(settings, "EMBED_EXECUTOR_WORKERS", 2)
    attempts: dict[str, int] = {}
    written: list[tuple[str, list[float] | None]] = []

    async def embed(memory):
        memory_id = str(memory.id)
        attempts[memory_id] = attempts.get(memory_id, 0) + 1
        if memory.content == "retry" and attempts[memory_id] == 1:
            raise RuntimeError("transient local model failure")
        return [float(memory.revision)]

    async def upsert(memory, *, embedding=None):
        if embedding is None:
            embedding = await embed(memory)
        written.append((str(memory.id), embedding))

    monkeypatch.setattr(outbox, "embed_memory", embed)
    monkeypatch.setattr(outbox, "upsert_memory", upsert)

    memories = [_memory(owner, content) for content in ("retry", "fast-1", "fast-2")]
    for memory in memories:
        db.add(memory)
        outbox.bump_revision(memory)
        await outbox.enqueue_upsert(db, memory)
    await db.commit()

    report = await outbox.drain_pending(priority_tenant=owner.hex)

    assert report["applied"] == 3
    assert attempts == {
        str(memory.id): (2 if memory.content == "retry" else 1) for memory in memories
    }
    assert {memory_id for memory_id, _ in written} == {str(memory.id) for memory in memories}


async def test_drain_converts_upsert_to_delete_when_the_row_is_gone(db, owner, monkeypatch):
    deletes: list[str] = []

    async def record(memory_id):
        deletes.append(memory_id)
        return True  # the backend confirmed the delete

    monkeypatch.setattr(outbox, "delete_memory", record)

    memory = _memory(owner)
    db.add(memory)
    outbox.bump_revision(memory)
    await outbox.enqueue_upsert(db, memory)
    await db.commit()
    await db.delete(memory)
    await db.commit()

    report = await outbox.drain_pending()

    assert report["applied"] == 1 and report["skipped"] == 0
    assert deletes == [str(memory.id)]
    assert [row.status for row in await _outbox_rows()] == ["done"]


async def test_drain_applies_an_explicit_delete_intent(db, owner, monkeypatch):
    deletes: list[str] = []

    async def record(memory_id):
        deletes.append(memory_id)
        return True  # the backend confirmed the delete

    monkeypatch.setattr(outbox, "delete_memory", record)

    memory = _memory(owner)
    db.add(memory)
    revision = outbox.bump_revision(memory)
    await outbox.enqueue_delete(db, entity_id=memory.id.hex, tenant_id=owner.hex, revision=revision)
    await db.commit()

    report = await outbox.drain_pending()

    assert report["applied"] == 1
    assert deletes == [str(memory.id)]


async def test_drain_keeps_an_unconfirmed_delete_pending(db, owner, monkeypatch):
    """An unconfirmed vector delete is transient, never `done` (Task 2 review)."""
    async def unconfirmed(_memory_id):
        return False

    monkeypatch.setattr(outbox, "delete_memory", unconfirmed)

    memory = _memory(owner)
    db.add(memory)
    revision = outbox.bump_revision(memory)
    await outbox.enqueue_delete(db, entity_id=memory.id.hex, tenant_id=owner.hex, revision=revision)
    await db.commit()

    assert (await outbox.drain_pending())["failed"] == 1
    row = (await _outbox_rows())[0]
    assert row.status == "pending" and row.attempts == 1
    assert row.next_attempt_at is not None
    assert "VectorDeleteUnconfirmed" in row.last_error


async def test_drain_marks_blocked_on_dimension_mismatch(db, owner, monkeypatch):
    async def mismatch(_memory):
        raise EmbeddingDimensionMismatch("contract mismatch: reindex required")

    monkeypatch.setattr(outbox, "upsert_memory", mismatch)

    memory = _memory(owner)
    db.add(memory)
    outbox.bump_revision(memory)
    await outbox.enqueue_upsert(db, memory)
    await db.commit()

    report = await outbox.drain_pending()
    assert report["blocked"] == 1

    row = (await _outbox_rows())[0]
    assert row.status == "blocked"
    assert "contract mismatch" in row.last_error and "EmbeddingDimensionMismatch" in row.last_error
    # Terminal: a blocked intent is never retried.
    assert (await outbox.drain_pending())["claimed"] == 0


async def test_drain_retries_transient_failure_with_backoff(db, owner, monkeypatch):
    async def flaky(_memory):
        raise RuntimeError("chroma is down")

    monkeypatch.setattr(outbox, "upsert_memory", flaky)

    memory = _memory(owner)
    db.add(memory)
    outbox.bump_revision(memory)
    await outbox.enqueue_upsert(db, memory)
    await db.commit()

    report = await outbox.drain_pending()
    assert report["failed"] == 1

    row = (await _outbox_rows())[0]
    assert row.status == "pending" and row.attempts == 1
    assert row.next_attempt_at is not None and "chroma is down" in row.last_error
    # Still inside the backoff window: not claimed.
    assert (await outbox.drain_pending())["claimed"] == 0

    applied: list[str] = []

    async def ok(memory_row):
        applied.append(str(memory_row.id))

    monkeypatch.setattr(outbox, "upsert_memory", ok)
    async with database.AsyncSessionLocal() as session:
        due = await session.get(IndexOutbox, row.seq)
        due.next_attempt_at = datetime.now(UTC) - timedelta(seconds=1)
        await session.commit()

    report = await outbox.drain_pending()
    assert report["applied"] == 1
    assert applied == [str(memory.id)]
    row = (await _outbox_rows())[0]
    assert row.status == "done" and row.attempts == 1  # attempts kept for the audit trail


async def test_backoff_grows_and_is_capped(db, owner, monkeypatch):
    """attempts increments per failure; the wait grows and saturates at 3600s."""
    async def chroma_down(_memory):
        raise RuntimeError("chroma is down")

    monkeypatch.setattr(outbox, "upsert_memory", chroma_down)

    memory = _memory(owner)
    db.add(memory)
    outbox.bump_revision(memory)
    await outbox.enqueue_upsert(db, memory)
    await db.commit()

    deltas: list[float] = []
    for attempt in range(1, 9):
        async with database.AsyncSessionLocal() as session:  # make the intent due again
            row = (await session.execute(
                select(IndexOutbox).order_by(IndexOutbox.seq))).scalars().one()
            row.next_attempt_at = datetime.now(UTC) - timedelta(seconds=1)
            await session.commit()
        before = datetime.now(UTC).replace(tzinfo=None)  # SQLite round-trips naive datetimes
        assert (await outbox.drain_pending())["failed"] == 1
        row = (await _outbox_rows())[0]
        assert row.attempts == attempt  # exactly one increment per failed attempt
        assert row.status == "pending" and row.next_attempt_at is not None
        deltas.append((row.next_attempt_at - before).total_seconds())

    ladder = [60, 120, 240, 480, 960, 1920, 3600, 3600]  # spec §5.2, capped at 3600
    for delta, step in zip(deltas, ladder, strict=True):
        # Window per step: base + uniform(0, min(base*10%, 30)) (spec §5.2). The
        # +2s absorbs the failed drain's own runtime after ``before`` was read;
        # the windows never overlap, so this also proves the ladder grows.
        assert step - 0.01 <= delta <= step + min(step * 0.1, 30) + 2, (delta, step)
    assert deltas[:6] == sorted(deltas[:6]) and len(set(deltas[:6])) == 6  # strictly growing
    assert deltas[5] < min(deltas[6:])  # the cap is a wait, still past the last growth step
    # The two capped waits are the SAME ladder step, so their jittered order is
    # not meaningful — only that both saturate at the 3600s cap.
    assert deltas[6:] == pytest.approx([3600, 3600], abs=32)  # 60*2**6 = 3840 -> capped; abs covers +30 jitter +2s drain

    # The capped intent is still not due: the cap is a wait, not a terminal state.
    assert (await outbox.drain_pending())["claimed"] == 0


async def test_drain_batch_size_bounds_one_run(db, owner, monkeypatch):
    applied: list[str] = []

    async def ok(memory_row):
        applied.append(str(memory_row.id))

    monkeypatch.setattr(outbox, "upsert_memory", ok)

    for i in range(3):
        memory = _memory(owner, content=f"v{i}")
        db.add(memory)
        outbox.bump_revision(memory)
        await outbox.enqueue_upsert(db, memory)
    await db.commit()

    first = await outbox.drain_pending(batch_size=2)
    assert first == {"claimed": 2, "applied": 2, "skipped": 0, "blocked": 0, "failed": 0}
    assert len(applied) == 2
    second = await outbox.drain_pending(batch_size=2)
    assert second["claimed"] == 1


# ── boot drain (P3): one bounded batch, both dialects, never a boot blocker ──


class _LogCapture:
    """Records ``main.log`` calls (level, event, kwargs) for the wiring asserts."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict]] = []

    def info(self, event: str, **kw) -> None:
        self.events.append(("info", event, kw))

    def warning(self, event: str, **kw) -> None:
        self.events.append(("warning", event, kw))

    def reports(self) -> list[dict]:
        return [kw for _level, event, kw in self.events if event == "Index outbox boot drain"]


def _pin_this_files_url(monkeypatch, tmp_path) -> None:
    """``main`` reads the URL at call time; pin it to the test's own SQLite file."""
    monkeypatch.setattr(main.settings, "DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / OUTBOX_DB}")


async def test_pending_intent_survives_a_restart_and_is_applied_on_the_next_boot(
    db, owner, monkeypatch, tmp_path
):
    """Crash between commit and index: the intent survives; the next drain heals it."""
    from app.api.v1 import memories as memories_api
    from app.schemas.Orivory import MemoryCreate

    async def chroma_down(_memory):
        return False  # write-through failed: the durable intent is all that remains

    monkeypatch.setattr(memories_api, "index_new_memory", chroma_down)
    response = await memories_api.create_memory(
        MemoryCreate(content="survives"), SimpleNamespace(id=owner), db
    )
    assert response.indexing == "pending"
    assert [row.status for row in await _outbox_rows()] == ["pending"]

    # Process dies here: its pool goes away, only the committed SQLite file stays.
    await database.engine.dispose()
    eng, sync_eng, _sessions = _open_engines(
        f"sqlite+aiosqlite:///{tmp_path / OUTBOX_DB}", monkeypatch
    )
    indexed: list[tuple[str, str, int]] = []

    async def upsert_ok(memory):
        indexed.append((str(memory.id), memory.content, memory.revision))

    monkeypatch.setattr(outbox, "upsert_memory", upsert_ok)
    try:
        report = await outbox.drain_pending()

        assert report == {"claimed": 1, "applied": 1, "skipped": 0, "blocked": 0, "failed": 0}
        assert indexed == [(str(response.id), "survives", 1)]  # exactly one vector write
        assert [row.status for row in await _outbox_rows()] == ["done"]
    finally:
        await eng.dispose()
        sync_eng.dispose()


async def test_boot_drain_is_bounded_to_one_batch(db, owner, monkeypatch, tmp_path):
    """One batch of 50 per boot: the background loop owns the rest (P3)."""
    indexed: list[str] = []

    async def upsert_ok(memory):
        indexed.append(str(memory.id))

    monkeypatch.setattr(outbox, "upsert_memory", upsert_ok)
    for i in range(120):
        memory = _memory(owner, content=f"v{i}")
        db.add(memory)
        outbox.bump_revision(memory)
        await outbox.enqueue_upsert(db, memory)
    await db.commit()

    _pin_this_files_url(monkeypatch, tmp_path)
    captured = _LogCapture()
    monkeypatch.setattr(main, "log", captured)

    await main._drain_index_outbox_at_boot()

    assert len(indexed) == 50  # a second batch is the loop's, not the boot's
    statuses = [row.status for row in await _outbox_rows()]
    assert statuses.count("done") == 50 and statuses.count("pending") == 70
    assert [report["claimed"] for report in captured.reports()] == [50]


async def test_boot_drain_keeps_intents_pending_when_the_vector_store_is_down(
    db, owner, monkeypatch, tmp_path
):
    """The ruling: a vector outage at boot leaves the intent pending, never blocks boot."""
    async def chroma_down(_memory):
        raise RuntimeError("chroma connection refused")

    monkeypatch.setattr(outbox, "upsert_memory", chroma_down)

    memory = _memory(owner)
    db.add(memory)
    outbox.bump_revision(memory)
    await outbox.enqueue_upsert(db, memory)
    await db.commit()

    _pin_this_files_url(monkeypatch, tmp_path)
    captured = _LogCapture()
    monkeypatch.setattr(main, "log", captured)

    await main._drain_index_outbox_at_boot()  # must return, not raise

    row = (await _outbox_rows())[0]
    assert row.status == "pending" and row.attempts == 1
    assert row.next_attempt_at is not None and "chroma connection refused" in row.last_error
    assert [report["failed"] for report in captured.reports()] == [1]  # the report was logged


async def test_boot_drain_failure_is_logged_not_raised(db, owner, monkeypatch, tmp_path):
    """Even a drain-level failure is a warning: the app must still boot."""
    from sqlalchemy import text

    memory = _memory(owner)
    db.add(memory)
    outbox.bump_revision(memory)
    await outbox.enqueue_upsert(db, memory)
    await db.commit()
    await db.execute(text("DROP TABLE index_outbox"))  # the drain's SELECT can only fail
    await db.commit()

    _pin_this_files_url(monkeypatch, tmp_path)
    captured = _LogCapture()
    monkeypatch.setattr(main, "log", captured)

    await main._drain_index_outbox_at_boot()

    level, event, kw = captured.events[0]
    assert (level, event) == ("warning", "Index outbox boot drain failed")
    assert kw["error"]


async def test_boot_drain_runs_outside_sqlite_deployments(db, owner, monkeypatch, tmp_path):
    """P3: the boot drain's SQLite-only gate is gone — both dialects drain.

    The URL is Postgres-shaped; the drain itself lands on this test's private
    SQLite file, so the code path plainly no longer branches on the dialect.
    """
    memory = _memory(owner)
    db.add(memory)
    outbox.bump_revision(memory)
    await outbox.enqueue_upsert(db, memory)
    await db.commit()

    indexed: list[str] = []

    async def upsert_ok(memory_row):
        indexed.append(str(memory_row.id))

    monkeypatch.setattr(outbox, "upsert_memory", upsert_ok)  # the vector write is not the contract here
    monkeypatch.setattr(main.settings, "DATABASE_URL", "postgresql+asyncpg://user:pw@db/orivory")
    captured = _LogCapture()
    monkeypatch.setattr(main, "log", captured)

    await main._drain_index_outbox_at_boot()

    assert [report["claimed"] for report in captured.reports()] == [1]
    assert indexed == [str(memory.id)]
    row = (await _outbox_rows())[0]
    assert row.status == "done"  # drained, not skipped


async def test_boot_drain_is_a_noop_when_the_drain_is_disabled(db, owner, monkeypatch, tmp_path):
    """``OUTBOX_DRAIN_ENABLED=false``: no boot batch either, logs stay quiet."""
    memory = _memory(owner)
    db.add(memory)
    outbox.bump_revision(memory)
    await outbox.enqueue_upsert(db, memory)
    await db.commit()

    _pin_this_files_url(monkeypatch, tmp_path)
    monkeypatch.setattr(main.settings, "OUTBOX_DRAIN_ENABLED", False)
    captured = _LogCapture()
    monkeypatch.setattr(main, "log", captured)

    await main._drain_index_outbox_at_boot()

    assert captured.events == []
    row = (await _outbox_rows())[0]
    assert row.status == "pending" and row.attempts == 0  # untouched: it never ran


# ── graph metadata writes stay out of the revision/outbox loop ───────────────


async def test_graph_build_neither_bumps_revision_nor_enqueues(db, owner, monkeypatch):
    from app.graph import builder as graph_builder
    from app.graph.extraction import EntityExtractionResult, RelationExtractionResult

    memory = _memory(owner)
    db.add(memory)
    outbox.bump_revision(memory)
    await outbox.enqueue_upsert(db, memory)
    await db.commit()

    async def no_entities(_memory):
        return EntityExtractionResult(entities=[])

    async def no_relations(_memory, _entities):
        return RelationExtractionResult(relations=[])

    monkeypatch.setattr(graph_builder, "extract_entities", no_entities)
    monkeypatch.setattr(graph_builder, "extract_relations", no_relations)

    def build():
        with sync_session() as sync_db:
            return graph_builder.build_memory_graph_sync(sync_db, str(memory.id))

    result = await asyncio.to_thread(build)  # the sync builder runs asyncio.run itself
    assert not result.skipped and result.entities_extracted == 0

    row = await _memory_row(memory.id)
    assert row.revision == 1  # a metadata write is not a content write
    assert row.extra_metadata["graph_extracted_at"]
    intents = await _outbox_rows()
    assert len(intents) == 1  # only the intent the write path itself enqueued
    assert (intents[0].operation, intents[0].revision) == ("upsert", 1)


# ── async write paths enqueue in their own commit ───────────────────────────


async def test_create_memory_enqueues_in_its_commit(db, owner, monkeypatch):
    from app.api.v1 import memories as memories_api
    from app.schemas.Orivory import MemoryCreate

    indexed: list[str] = []

    async def fake_index(memory):
        indexed.append(str(memory.id))
        return True

    monkeypatch.setattr(memories_api, "index_new_memory", fake_index)

    response = await memories_api.create_memory(
        MemoryCreate(content="hello"), SimpleNamespace(id=owner), db
    )

    assert response.revision == 1 and response.indexing == "ready"
    assert indexed == [str(response.id)]
    stored = await _memory_row(response.id)
    assert stored.revision == 1
    intents = await _outbox_rows()
    assert len(intents) == 1
    assert (intents[0].entity_id, intents[0].revision, intents[0].operation) == (
        response.id.hex, 1, "upsert")


async def test_create_memory_reports_pending_when_the_immediate_upsert_failed(db, owner, monkeypatch):
    from app.api.v1 import memories as memories_api
    from app.schemas.Orivory import MemoryCreate

    async def fake_index(_memory):
        return False  # vector store unreachable: the durable intent is the backstop

    monkeypatch.setattr(memories_api, "index_new_memory", fake_index)

    response = await memories_api.create_memory(
        MemoryCreate(content="hello"), SimpleNamespace(id=owner), db
    )

    assert response.indexing == "pending"
    assert len(await _outbox_rows()) == 1


async def test_immediate_index_acks_its_intent_only_when_the_write_landed(db, owner, monkeypatch):
    """F2: a landed fast path flips its own intent to `done`; a failed one stays pending."""
    from app.api.v1 import memories as memories_api
    from app.schemas.Orivory import MemoryCreate

    upserts: list[str] = []

    async def landed(memory):
        upserts.append(memory.content)
        return True

    monkeypatch.setattr(memories_api, "index_new_memory", landed)
    ready = await memories_api.create_memory(
        MemoryCreate(content="indexed now"), SimpleNamespace(id=owner), db
    )
    assert ready.indexing == "ready"

    async def chroma_down(_memory):
        return False  # nothing was written: the intent is the only record of it

    monkeypatch.setattr(memories_api, "index_new_memory", chroma_down)
    pending = await memories_api.create_memory(
        MemoryCreate(content="index later"), SimpleNamespace(id=owner), db
    )
    assert pending.indexing == "pending"

    by_id = {row.entity_id: row for row in await _outbox_rows()}
    assert (by_id[ready.id.hex].status, by_id[ready.id.hex].attempts) == ("done", 0)
    assert by_id[pending.id.hex].status == "pending"

    # The boot drain finds only the un-acked intent: the indexed revision is
    # never re-applied.
    async def drain_upsert(memory):
        upserts.append(memory.content)

    monkeypatch.setattr(outbox, "upsert_memory", drain_upsert)
    assert await outbox.drain_pending() == {
        "claimed": 1, "applied": 1, "skipped": 0, "blocked": 0, "failed": 0,
    }
    assert upserts == ["indexed now", "index later"]

    # ...and it never touches a delete intent at the same entity+revision.
    await outbox.enqueue_delete(db, entity_id=ready.id.hex, tenant_id=owner.hex, revision=1)
    await db.commit()
    await outbox.mark_done(db, entity_id=ready.id, revision=1)

    actions = {row.operation: row.status for row in await _outbox_rows()
               if row.entity_id == ready.id.hex}
    assert actions == {"upsert": "done", "delete": "pending"}


async def test_update_memory_bumps_revision_and_enqueues(db, owner, monkeypatch):
    from app.api.v1 import memories as memories_api
    from app.schemas.Orivory import MemoryUpdate

    async def upsert_ok(_memory):
        return True

    monkeypatch.setattr(memories_api, "safe_upsert_to_index", upsert_ok)

    memory = _memory(owner, content="v1")
    db.add(memory)
    outbox.bump_revision(memory)
    await outbox.enqueue_upsert(db, memory)
    await db.commit()

    updated = await memories_api.update_memory(
        memory.id, MemoryUpdate(title="renamed"), SimpleNamespace(id=owner), db
    )

    assert updated.revision == 2 and updated.indexing == "ready"
    assert updated.title == "renamed"
    intents = await _outbox_rows()
    assert [(row.revision, row.operation) for row in intents] == [(1, "upsert"), (2, "upsert")]


async def test_noop_patch_never_claims_a_pending_intent(db, owner, monkeypatch):
    """A no-op PATCH enqueues nothing: a failed write-through is not 'pending'."""
    from app.api.v1 import memories as memories_api
    from app.schemas.Orivory import MemoryUpdate

    async def upsert_down(_memory):
        return False

    monkeypatch.setattr(memories_api, "safe_upsert_to_index", upsert_down)

    memory = _memory(owner, content="v1")
    db.add(memory)
    outbox.bump_revision(memory)
    await outbox.enqueue_upsert(db, memory)
    await db.commit()

    response = await memories_api.update_memory(
        memory.id, MemoryUpdate(), SimpleNamespace(id=owner), db
    )

    assert response.indexing is None, "no intent exists; 'pending' would be a lie"
    assert response.revision == 1  # the body changed nothing, no bump
    assert len(await _outbox_rows()) == 1  # only the fixture's own intent

    async def upsert_ok(_memory):
        return True

    monkeypatch.setattr(memories_api, "safe_upsert_to_index", upsert_ok)
    again = await memories_api.update_memory(
        memory.id, MemoryUpdate(), SimpleNamespace(id=owner), db
    )
    assert again.indexing == "ready" and again.revision == 1  # write-through landed


async def test_get_memory_does_not_claim_an_index_state(db, owner):
    """A read reports no `indexing`: the only intent is still pending."""
    from app.api.v1 import memories as memories_api

    memory = _memory(owner, content="v1")
    db.add(memory)
    outbox.bump_revision(memory)
    await outbox.enqueue_upsert(db, memory)
    await db.commit()

    fetched = await memories_api.get_memory(memory.id, SimpleNamespace(id=owner), db)

    assert fetched.indexing is None
    assert fetched.revision == 1
    assert (await _outbox_rows())[0].status == "pending"  # not indexed, and we do not claim it


CHATGPT_PAYLOAD = [
    {
        "id": "c1",
        "title": "First",
        "create_time": 1738454400.0,
        "mapping": {
            "u": {"message": {"author": {"role": "user"}, "create_time": 1.0,
                              "content": {"parts": ["hello"]}}},
            "a": {"message": {"author": {"role": "assistant"}, "create_time": 2.0,
                              "content": {"parts": ["hi there"]}}},
        },
    },
    {
        "id": "c2",
        "title": "Second",
        "create_time": 1738454500.0,
        "mapping": {
            "u": {"message": {"author": {"role": "user"}, "create_time": 1.0,
                              "content": {"parts": ["second conv"]}}},
        },
    },
]


async def test_import_enqueues_every_created_row(db, owner, monkeypatch):
    from app.services import import_service

    indexed: list[str] = []

    async def fake_index(memory):
        indexed.append(str(memory.id))

    monkeypatch.setattr(import_service, "index_new_memory", fake_index)

    summary = await import_service.run_import(
        db, owner, json.dumps(CHATGPT_PAYLOAD).encode("utf-8"), "chatgpt", requested_by="test"
    )

    assert summary.created == 2
    async with database.AsyncSessionLocal() as session:
        memories = (await session.execute(select(Memory))).scalars().all()
    assert {row.revision for row in memories} == {1}
    assert len(indexed) == 2
    intents = await _outbox_rows()
    assert {row.entity_id for row in intents} == {row.id.hex for row in memories}
    assert {row.operation for row in intents} == {"upsert"}


async def test_correction_enqueues_the_new_fact(db, owner):
    from app.retrieval.memory.correction import Slot, resolve_correction

    first = await resolve_correction(db, user_id=owner, title="DB", content="Postgres",
                                     slot=Slot.of("proj", "db", "prod"))
    second = await resolve_correction(db, user_id=owner, title="DB", content="SQLite",
                                      slot=Slot.of("proj", "db", "prod"))

    assert second["status"] == "superseded"
    intents = await _outbox_rows()
    assert [(row.entity_id, row.revision) for row in intents] == [
        (first["memory"].id.hex, 1),
        (second["memory"].id.hex, 1),
    ]


# ── R27: drain and mark_done are generation-aware ───────────────────────────


async def _activate_generation(kind: str, generation: str) -> None:
    async with database.AsyncSessionLocal() as session:
        session.add(IndexGeneration(id=uuid.uuid4().hex, kind=kind, generation=generation,
                                    fingerprint="f" * 64, is_active=True))
        await session.commit()


async def _queue_intent(db, *, memory, target_generation: str, revision: int = 1) -> None:
    """A pending intent naming ``target_generation`` (bypassing the manifest)."""
    db.add(IndexOutbox(kind="memory", entity_id=memory.id.hex, tenant_id=str(memory.user_id).replace("-", ""),
                       revision=revision, operation="upsert",
                       target_generation=target_generation, status="pending"))
    await db.commit()


async def test_drain_blocks_an_intent_for_a_superseded_generation(db, owner, monkeypatch):
    """R27: an intent the active pointer no longer names is terminal, never applied.

    Its write is already covered by the migration's backfill; applying it would
    write into a generation the app does not serve."""
    applied: list[str] = []

    async def record(memory):
        applied.append(str(memory.id))

    monkeypatch.setattr(outbox, "upsert_memory", record)

    memory = _memory(owner)
    db.add(memory)
    outbox.bump_revision(memory)
    await db.commit()
    await _activate_generation("memory", "orivory_memories__new")
    await _queue_intent(db, memory=memory, target_generation=outbox.TARGET_GENERATION)

    report = await outbox.drain_pending()

    assert report == {"claimed": 1, "applied": 0, "skipped": 0, "blocked": 1, "failed": 0}
    assert applied == []
    row = (await _outbox_rows())[0]
    assert row.status == "blocked", "terminal: a stale-target intent is not retried"
    assert row.attempts == 0
    assert "generation" in (row.last_error or "")


async def test_drain_applies_an_intent_for_the_active_generation(db, owner, monkeypatch):
    """The control: the predicate must not block the generation in force."""
    applied: list[str] = []

    async def record(memory):
        applied.append(str(memory.id))

    monkeypatch.setattr(outbox, "upsert_memory", record)

    memory = _memory(owner)
    db.add(memory)
    outbox.bump_revision(memory)
    await outbox.enqueue_upsert(db, memory)  # stamped with the transitional fallback
    await db.commit()
    # Activate exactly the generation the intent carries: nothing to block.
    await _activate_generation("memory", outbox.TARGET_GENERATION)

    acked = await outbox.drain_pending()

    assert acked["applied"] == 1 and applied == [str(memory.id)]
    assert (await _outbox_rows())[0].status == "done"


async def test_mark_done_never_acks_an_intent_for_another_generation(db, owner):
    """The ack carries the same predicate: a write into the new generation must
    not close an intent that promised the old one."""
    memory = _memory(owner)
    db.add(memory)
    outbox.bump_revision(memory)
    await db.commit()
    await _activate_generation("memory", "orivory_memories__new")
    await _queue_intent(db, memory=memory, target_generation=outbox.TARGET_GENERATION)

    assert await outbox.mark_done(db, entity_id=memory.id, revision=memory.revision) == 0
    assert (await _outbox_rows())[0].status == "pending"

    # The same intent under the ACTIVE generation is ackable.
    rows = await _outbox_rows()
    async with database.AsyncSessionLocal() as session:
        row = await session.get(IndexOutbox, rows[0].seq)
        row.target_generation = "orivory_memories__new"
        await session.commit()
    assert await outbox.mark_done(db, entity_id=memory.id, revision=memory.revision) == 1
    assert (await _outbox_rows())[0].status == "done"


def test_mark_done_sync_carries_the_same_predicate(db, owner):
    memory = _memory(owner)
    with sync_session() as sync_db:
        sync_db.add(memory)
        outbox.bump_revision(memory)
        sync_db.commit()
    _activate_generation_sync("memory", "orivory_memories__new")
    with sync_session() as sync_db:
        sync_db.add(IndexOutbox(kind="memory", entity_id=memory.id.hex,
                                tenant_id=str(memory.user_id).replace("-", ""), revision=1,
                                operation="upsert", target_generation=outbox.TARGET_GENERATION,
                                status="pending"))
        sync_db.commit()
        assert outbox.mark_done_sync(sync_db, entity_id=memory.id, revision=memory.revision) == 0
        assert sync_db.execute(select(IndexOutbox)).scalars().one().status == "pending"


def _activate_generation_sync(kind: str, generation: str) -> None:
    with sync_session() as sync_db:
        sync_db.add(IndexGeneration(id=uuid.uuid4().hex, kind=kind, generation=generation,
                                    fingerprint="f" * 64, is_active=True))
        sync_db.commit()
