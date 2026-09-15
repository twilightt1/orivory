"""Task 5 — the drain runs CONCURRENTLY with writers: re-check the row (R6/R21).

The applier's snapshot window is reachable in production: ``_apply_memory_intent``
/ ``_apply_chunk_intent`` read the row, then upsert that snapshot to the store.
A writer that commits in between (a delete, or a correction that bumps
``revision``) leaves the store holding a point for a row that is gone or
superseded — and the intent acks the stale snapshot as applied.

Each race here is driven by a DETERMINISTIC hook (R2): the monkeypatched
``upsert_memory`` / ``upsert_chunks`` runs the writer's commit *inside* the
applier's own write call, so the interleaving is exact — no sleeps, no timing
guesses. The suite is isolated the ``tests/retrieval/test_drain_loop.py`` way:
a real embedded Qdrant on a private ``tmp_path`` folder plus a private SQLite
file patched in as the module engines, so it can never read or write the
ambient ``DATABASE_URL``.

What R21 fixes, and what it deliberately does not: a row that vanished while the
snapshot was in flight gets the point deleted (the applier reports ``applied``);
a row whose revision moved on makes the applier STAND DOWN (``skipped``) — the
newer intent owns the point, and this applier neither overwrites nor deletes it
(R6: bounded, one fresh read per applied upsert, never a reconciliation pass).
"""
from __future__ import annotations

import hashlib
import math
import random
import uuid
from types import SimpleNamespace

import pytest_asyncio
from sqlalchemy import create_engine, event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from app import database
from app import models as _models  # noqa: F401 — register every table on Base
from app.config import settings
from app.database import Base, sync_session
from app.models.conversation import Conversation
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.index_outbox import IndexGeneration, IndexOutbox
from app.models.memory import Memory
from app.models.user import User
from app.retrieval import vector_backend, vector_retriever
from app.retrieval.embedding_fingerprint import canonical_fingerprint, fingerprint_generation
from app.retrieval.memory import outbox, vector_store
from app.retrieval.memory.vector_store import COLLECTION_NAME

DIM = 8
RACES_DB = "races.sqlite"
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


def _fake_embed_sync(texts: list[str]) -> list[list[float]]:
    return [_vector_for(text) for text in texts]


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def env(tmp_path, monkeypatch):
    """Real embedded Qdrant + private SQLite outbox, both on ``tmp_path``."""
    folder = tmp_path / "qdrant"
    folder.mkdir()
    monkeypatch.setattr(settings, "QDRANT_MODE", "local")
    monkeypatch.setattr(settings, "QDRANT_LOCAL_PATH", str(folder))

    url = f"sqlite+aiosqlite:///{tmp_path / RACES_DB}"
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
    monkeypatch.setattr(vector_store, "embed_texts_sync", _fake_embed_sync)  # the seeding helper
    monkeypatch.setattr(vector_retriever, "embed_texts", _fake_embed)
    # Every guard must see the same contract: embedder binds the fingerprint at
    # import, so patch the alias each module actually calls.
    from app.retrieval import embedder as embedder_module
    from app.retrieval import embedding_fingerprint as fingerprint_module

    for module in (fingerprint_module, embedder_module, vector_store, vector_retriever):
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
async def world(env) -> SimpleNamespace:
    """An owner, both kinds' active manifests, and one conversation+document."""
    user_id = uuid.uuid4()
    conversation_id, document_id = uuid.uuid4(), uuid.uuid4()
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
        db.add(
            Conversation(
                id=conversation_id, user_id=user_id, title="conv", document_count=1
            )
        )
        db.add(
            Document(
                id=document_id,
                conversation_id=conversation_id,
                filename="f.pdf",
                file_path="uploads/f.pdf",
            )
        )
        for kind, generation in (
            (outbox.KIND_MEMORY, outbox.TARGET_GENERATION),
            (outbox.KIND_CHUNK, outbox.CHUNK_TARGET_GENERATION),
        ):
            db.add(
                IndexGeneration(
                    id=uuid.uuid4().hex,
                    kind=kind,
                    generation=generation,
                    fingerprint=_expected_token(),
                    is_active=True,
                )
            )
        await db.commit()
    return SimpleNamespace(
        user_id=user_id, conversation_id=conversation_id, document_id=document_id
    )


# ── rows, intents, points ───────────────────────────────────────────────────


def _pending_memory(path: str, owner: uuid.UUID) -> uuid.UUID:
    """One committed memory at revision 1 with its pending upsert intent."""
    memory_id = uuid.uuid4()
    with sync_session() as db:
        memory = Memory(id=memory_id, user_id=owner, content=path, tags=[])
        db.add(memory)
        outbox.bump_revision(memory)
        outbox.enqueue_upsert_sync(db, memory)
        db.commit()
    return memory_id


def _store_memory(path: str, owner: uuid.UUID) -> uuid.UUID:
    """A memory row AND its point, already in the store (no pending intent)."""
    memory_id = _pending_memory(path, owner)
    with sync_session() as db:
        memory = db.get(Memory, memory_id)
        vector_store.upsert_memory_sync(memory)
        outbox.mark_done_sync(db, entity_id=memory_id, revision=1)
    return memory_id


def _pending_chunk(world: SimpleNamespace, body: str) -> uuid.UUID:
    """One committed child chunk at revision 1 with its pending upsert intent."""
    chunk_id = uuid.uuid4()
    with sync_session() as db:
        db.add(
            DocumentChunk(
                id=chunk_id,
                document_id=world.document_id,
                content=body,
                chunk_index=0,
                revision=1,
                chunk_metadata={
                    "document_id": str(world.document_id),
                    "conversation_id": str(world.conversation_id),
                    "chunk_type": "child",
                    "child_index": 0,
                    "parent_id": str(uuid.uuid4()),
                },
            )
        )
        outbox.enqueue_chunk_upsert_sync(
            db,
            chunk_id=chunk_id,
            tenant_id=world.user_id,
            revision=1,
            conversation_id=world.conversation_id,
        )
        db.commit()
    return chunk_id


async def _intents() -> list[IndexOutbox]:
    """Read the outbox through a fresh session (immune to snapshot staleness)."""
    async with database.AsyncSessionLocal() as session:
        return list(
            (await session.execute(select(IndexOutbox).order_by(IndexOutbox.seq))).scalars().all()
        )


def _points(generation: str, point_ids: list) -> dict[str, dict]:
    """Persisted payloads read back through the real client (``{}`` when the
    generation was never written to)."""
    client = vector_backend.get_sync_client()
    if not client.collection_exists(generation):
        return {}
    records = client.retrieve(
        collection_name=generation,
        ids=[str(point_id) for point_id in point_ids],
        with_payload=True,
    )
    return {str(record.id): dict(record.payload) for record in records}


def _memory_point(memory_id) -> dict | None:
    return _points(COLLECTION_NAME, [memory_id]).get(str(memory_id))


def _chunk_point(chunk_id) -> dict | None:
    return _points(outbox.CHUNK_TARGET_GENERATION, [chunk_id]).get(str(chunk_id))


# ── (a) the row is deleted while the upsert is in flight ────────────────────


async def test_a_memory_deleted_mid_flight_never_keeps_the_point(env, world, monkeypatch):
    """R21: row gone after the write ⇒ delete the point just written, report ``applied``.

    The row can vanish with no intent of its own (cascade / hard delete); the
    applier's own pre-read already treats "row gone" as "forget" — the race is
    that it vanishes AFTER that read. Without the re-check the point stays in
    the store for a row that is gone, and the intent acks it ``done``.
    """
    memory_id = _pending_memory("doomed", world.user_id)
    real = vector_store.upsert_memory
    raced: list[bool] = []

    async def hook(memory):
        if not raced:  # one-shot: the interleaving happens on the FIRST write only
            with sync_session() as writer:
                row = writer.get(Memory, memory_id)
                assert row is not None
                writer.delete(row)
                writer.commit()
            raced.append(True)
        return await real(memory)  # the applier's stale snapshot write goes out

    monkeypatch.setattr(outbox, "upsert_memory", hook)

    report = await outbox.drain_pending()

    assert raced == [True]  # the interleaving really happened
    assert report == {"claimed": 1, "applied": 1, "skipped": 0, "blocked": 0, "failed": 0}
    assert _memory_point(memory_id) is None  # RED before the fix: the point was left behind
    assert [row.status for row in await _intents()] == ["done"]  # acked, no retry storm


async def test_a_chunk_deleted_mid_flight_never_keeps_the_point(env, world, monkeypatch):
    """Same shape on the chunk face (``_apply_chunk_intent``)."""
    chunk_id = _pending_chunk(world, "doomed chunk")
    real = vector_retriever.upsert_chunks
    raced: list[bool] = []

    async def hook(chunks, *, user_id):
        if not raced:
            with sync_session() as writer:
                row = writer.get(DocumentChunk, chunk_id)
                assert row is not None
                writer.delete(row)
                writer.commit()
            raced.append(True)
        return await real(chunks, user_id=user_id)

    monkeypatch.setattr(outbox, "upsert_chunks", hook)

    report = await outbox.drain_pending()

    assert raced == [True]
    assert report == {"claimed": 1, "applied": 1, "skipped": 0, "blocked": 0, "failed": 0}
    assert _chunk_point(chunk_id) is None  # RED before the fix
    assert [row.status for row in await _intents()] == ["done"]


# ── (b) a correction bumps the revision while the upsert is in flight ───────


async def test_a_memory_correction_mid_flight_stands_the_applier_down(env, world, monkeypatch):
    """R21: revision moved on ⇒ ``skipped``; the newer intent owns the point.

    The correction commits while the snapshot is in flight and enqueues its own
    intent in the same transaction (the codebase's invariant). The applier must
    not claim the superseded snapshot as applied — and must not repair it either:
    the newer intent is what lands the new payload.
    """
    memory_id = _pending_memory("v1", world.user_id)
    real = vector_store.upsert_memory
    raced: list[int] = []

    async def hook(memory):
        if not raced:
            with sync_session() as writer:
                row = writer.get(Memory, memory_id)
                assert row is not None
                row.content = "v2"
                revision = outbox.bump_revision(row)
                outbox.enqueue_upsert_sync(writer, row)
                writer.commit()
            raced.append(revision)
        return await real(memory)  # the applier's superseded snapshot write goes out

    monkeypatch.setattr(outbox, "upsert_memory", hook)

    report = await outbox.drain_pending()

    assert raced == [2]
    # RED before the fix: applied=1, skipped=0 — the stale snapshot was claimed.
    assert report == {"claimed": 1, "applied": 0, "skipped": 1, "blocked": 0, "failed": 0}
    # The superseded intent is acked (not retried), and the newer one is what
    # R21 leans on being there: it is still pending, owning the point.
    assert [row.status for row in await _intents()] == ["done", "pending"]

    # The newer intent owns the point: the next pass lands revision 2, and the
    # superseded payload does not survive it.
    second = await outbox.drain_pending()
    assert second["applied"] == 1, second
    payload = _memory_point(memory_id)
    assert payload["orivory_memory_revision"] == 2
    assert payload["content"] == "v2"
    assert [row.status for row in await _intents()] == ["done", "done"]


async def test_a_chunk_correction_mid_flight_stands_the_applier_down(env, world, monkeypatch):
    """Same shape on the chunk face: ``skipped`` for the superseded revision."""
    chunk_id = _pending_chunk(world, "v1 chunk")
    real = vector_retriever.upsert_chunks
    raced: list[int] = []

    async def hook(chunks, *, user_id):
        if not raced:
            with sync_session() as writer:
                row = writer.get(DocumentChunk, chunk_id)
                assert row is not None
                row.content = "v2 chunk"
                row.revision = 2
                outbox.enqueue_chunk_upsert_sync(
                    writer,
                    chunk_id=chunk_id,
                    tenant_id=world.user_id,
                    revision=2,
                    conversation_id=world.conversation_id,
                )
                writer.commit()
            raced.append(2)
        return await real(chunks, user_id=user_id)

    monkeypatch.setattr(outbox, "upsert_chunks", hook)

    report = await outbox.drain_pending()

    assert raced == [2]
    assert report == {"claimed": 1, "applied": 0, "skipped": 1, "blocked": 0, "failed": 0}
    assert [row.status for row in await _intents()] == ["done", "pending"]

    assert (await outbox.drain_pending())["applied"] == 1
    payload = _chunk_point(chunk_id)
    assert payload["revision"] == 2
    assert payload["content"] == "v2 chunk"
    assert [row.status for row in await _intents()] == ["done", "done"]


# ── R6: the re-check is bounded to upserts ──────────────────────────────────


async def test_a_delete_intent_pays_no_extra_row_read(env, world):
    """R6: a delete intent has no snapshot to go stale — one read, no re-check.

    The count is the honest bound: the applier's own pre-read, and nothing else.
    A re-check bolted onto every intent instead of only the applied upserts
    would still be "correct" on the (a)/(b) races and would fail here.
    """
    memory_id = _store_memory("deleted later", world.user_id)
    assert _memory_point(memory_id) is not None

    async with env() as db:
        await outbox.enqueue_delete(
            db, entity_id=str(memory_id), tenant_id=str(world.user_id), revision=2
        )
        await db.commit()
    with sync_session() as writer:
        row = writer.get(Memory, memory_id)
        assert row is not None
        writer.delete(row)
        writer.commit()

    reads: list[str] = []

    def count_row_reads(_conn, _cursor, statement, _parameters, _context, _many):
        if "FROM memories" in statement:
            reads.append(statement)

    engine = database.engine.sync_engine
    event.listen(engine, "before_cursor_execute", count_row_reads)
    try:
        report = await outbox.drain_pending()
    finally:
        event.remove(engine, "before_cursor_execute", count_row_reads)

    assert report == {"claimed": 1, "applied": 1, "skipped": 0, "blocked": 0, "failed": 0}
    assert _memory_point(memory_id) is None
    assert len(reads) == 1  # the pre-read only: deletes skip the re-check
