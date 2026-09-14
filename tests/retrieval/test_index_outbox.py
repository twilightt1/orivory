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

from app import database
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


@pytest_asyncio.fixture
async def db(tmp_path, monkeypatch):
    """A private per-test SQLite file — nothing here can reach an ambient DB."""
    url = f"sqlite+aiosqlite:///{tmp_path / OUTBOX_DB}"
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


async def test_update_memory_bumps_revision_and_enqueues(db, owner, monkeypatch):
    from app.api.v1 import memories as memories_api
    from app.schemas.Orivory import MemoryUpdate

    async def upsert_ok(_memory):
        return True

    monkeypatch.setattr(memories_api, "safe_upsert_to_chroma", upsert_ok)

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

    monkeypatch.setattr(memories_api, "safe_upsert_to_chroma", upsert_down)

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

    monkeypatch.setattr(memories_api, "safe_upsert_to_chroma", upsert_ok)
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
