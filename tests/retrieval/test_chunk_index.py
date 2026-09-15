"""T3 — chunk generation on Qdrant: payload contract, filter, intents, deletes.

Isolated by construction: a real embedded Qdrant on a private ``tmp_path``
folder (closed in teardown, so its folder lock never leaks into another test)
and a private per-test SQLite file monkeypatched in as the module engines /
sessionmakers (the ``tests/retrieval/test_qdrant_parity.py`` pattern), so this
suite can never read or write whatever ``DATABASE_URL`` is ambient.

Embeddings are deterministic unit vectors: payloads are read back through the
real store and filters are proven by what they return/delete, so neither face
is mocked.
"""
from __future__ import annotations

import hashlib
import math
import random
import uuid
from types import SimpleNamespace

import pytest
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
from app.models.user import User
from app.retrieval import vector_backend, vector_retriever
from app.retrieval.embedding_fingerprint import canonical_fingerprint, fingerprint_generation
from app.retrieval.memory import outbox
from app.retrieval.memory.vector_store import COLLECTION_NAME
from app.retrieval.qdrant_filter import build_chunk_filter

DIM = 8
GEN_DB = "chunks.sqlite"
OTHER_GENERATION = "orivory_chunks__othergen"
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


# ── deterministic embeddings (one rule, both faces) ─────────────────────────


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


async def _fake_embed_query(text: str) -> list[float]:
    return _vector_for(text)


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def env(tmp_path, monkeypatch):
    """Real embedded Qdrant + private SQLite outbox, both on ``tmp_path``."""
    folder = tmp_path / "qdrant"
    folder.mkdir()
    monkeypatch.setattr(settings, "QDRANT_MODE", "local")
    monkeypatch.setattr(settings, "QDRANT_LOCAL_PATH", str(folder))

    url = f"sqlite+aiosqlite:///{tmp_path / GEN_DB}"
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
    monkeypatch.setattr(vector_retriever, "embed_texts", _fake_embed)
    monkeypatch.setattr(vector_retriever, "embed_texts_sync", _fake_embed_sync)
    monkeypatch.setattr(vector_retriever, "embed_query", _fake_embed_query)
    # Every guard must see the same contract: embedder binds the fingerprint at
    # import, so patch the alias each module actually calls.
    from app.retrieval import embedder as embedder_module
    from app.retrieval import embedding_fingerprint as fingerprint_module

    for module in (fingerprint_module, embedder_module, vector_retriever):
        monkeypatch.setattr(module, "current_fingerprint", _fingerprint)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield sessions
    finally:
        await vector_backend.close_clients()
        await engine.dispose()
        sync_engine.dispose()


async def _add_user(sessions, email: str) -> uuid.UUID:
    user_id = uuid.uuid4()
    async with sessions() as db:
        db.add(
            User(
                id=user_id,
                email=email,
                hashed_password="x",
                display_name="Owner",
                is_verified=True,
                is_active=True,
            )
        )
        await db.commit()
    return user_id


async def _activate(sessions, generation: str, fingerprint: str | None = None) -> None:
    """Point the CHUNK manifest at ``generation`` (one active row per kind)."""
    fingerprint = fingerprint or _expected_token()
    async with sessions() as db:
        rows = (
            await db.execute(select(IndexGeneration).where(IndexGeneration.kind == "chunk"))
        ).scalars().all()
        existing = None
        for row in rows:
            row.is_active = False
            if row.generation == generation:
                existing = row
        if existing is not None:
            existing.is_active = True
            existing.fingerprint = fingerprint
        else:
            db.add(
                IndexGeneration(
                    id=uuid.uuid4().hex,
                    kind="chunk",
                    generation=generation,
                    fingerprint=fingerprint,
                    is_active=True,
                )
            )
        await db.commit()


@pytest_asyncio.fixture
async def world(env) -> SimpleNamespace:
    """An owner with one conversation + document, and the chunk manifest row."""
    user_id = await _add_user(env, "owner@test.invalid")
    await _activate(env, outbox.CHUNK_TARGET_GENERATION)
    async with env() as db:
        conversation = Conversation(
            id=uuid.uuid4(), user_id=user_id, title="conv", document_count=1
        )
        document = Document(
            id=uuid.uuid4(),
            conversation_id=conversation.id,
            filename="f.pdf",
            file_path="uploads/f.pdf",
        )
        db.add_all([conversation, document])
        await db.commit()
    return SimpleNamespace(
        user_id=user_id, conversation_id=conversation.id, document_id=document.id
    )


def _chunk(world, *, content: str = "chunk body", index: int = 0, revision: int = 1,
           document_id=None, conversation_id=None) -> DocumentChunk:
    """A child chunk shaped exactly like the pipeline writes them."""
    return DocumentChunk(
        id=uuid.uuid4(),
        document_id=document_id or world.document_id,
        content=content,
        chunk_index=index,
        revision=revision,
        chunk_metadata={
            "document_id": str(document_id or world.document_id),
            "conversation_id": str(conversation_id or world.conversation_id),
            "chunk_type": "child",
            "child_index": index,
            "parent_id": str(uuid.uuid4()),
        },
    )


async def _store(sessions, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    async with sessions() as db:
        for chunk in chunks:
            db.add(chunk)
        await db.commit()
    return chunks


async def _intents() -> list[IndexOutbox]:
    async with database.AsyncSessionLocal() as db:
        return list((await db.execute(select(IndexOutbox).order_by(IndexOutbox.seq))).scalars().all())


def _points(chunk_ids, *, generation: str | None = None) -> dict[str, dict]:
    """Persisted payloads for ``chunk_ids``, read through the real client.

    A generation that was never written to does not exist yet: it holds no
    points.
    """
    client = vector_backend.get_sync_client()
    name = generation or outbox.CHUNK_TARGET_GENERATION
    if not client.collection_exists(name):
        return {}
    records = client.retrieve(
        collection_name=name,
        ids=[str(chunk_id) for chunk_id in chunk_ids],
        with_payload=True,
    )
    return {str(record.id): dict(record.payload) for record in records}


def _present(chunk_id, *, generation: str | None = None) -> bool:
    return bool(_points([chunk_id], generation=generation))


# ── payload contract ────────────────────────────────────────────────────────


async def test_upsert_writes_the_chunk_payload_contract(env, world):
    chunk = _chunk(world, content="contract body", index=3, revision=2)
    await _store(env, [chunk])

    await vector_retriever.upsert_chunks([chunk], user_id=str(world.user_id))

    payload = _points([chunk.id])[str(chunk.id)]
    assert payload["kind"] == "chunk"
    assert payload["user_id"] == str(world.user_id)
    assert payload["conversation_id"] == str(world.conversation_id)
    assert payload["document_id"] == str(world.document_id)
    assert payload["chunk_id"] == str(chunk.id)
    assert payload["revision"] == 2
    assert payload["child_index"] == 3
    assert payload["fingerprint"] == canonical_fingerprint(FINGERPRINT)
    assert payload["parent_id"] == chunk.chunk_metadata["parent_id"]
    assert payload["content"] == "contract body"


async def test_sync_and_async_faces_write_the_same_generation(env, world):
    first = _chunk(world, content="async write")
    second = _chunk(world, content="sync write")
    await _store(env, [first, second])

    await vector_retriever.upsert_chunks([first], user_id=str(world.user_id))
    vector_retriever.upsert_chunks_sync([second], user_id=str(world.user_id))

    assert set(_points([first.id, second.id])) == {str(first.id), str(second.id)}


async def test_upsert_skips_an_empty_batch(env, world):
    assert await vector_retriever.upsert_chunks([], user_id=str(world.user_id)) == 0
    assert vector_retriever.upsert_chunks_sync([], user_id=str(world.user_id)) == 0


async def test_payload_falls_back_to_the_document_conversation_id(env, world):
    """M2: metadata without a conversation_id is scoped by the document row."""
    chunk = _chunk(world, content="no metadata conv")
    chunk.chunk_metadata.pop("conversation_id")
    chunk.document = Document(
        id=world.document_id,
        conversation_id=world.conversation_id,
        filename="f.pdf",
        file_path="uploads/f.pdf",
    )

    await vector_retriever.upsert_chunks([chunk], user_id=str(world.user_id))

    assert _points([chunk.id])[str(chunk.id)]["conversation_id"] == str(world.conversation_id)
    # The point is reachable by the conversation-scoped filter/delete (the whole
    # point of M2: "" made it unreachable by every one of them).
    hits = await vector_retriever.search(
        "no metadata conv", 5, str(world.conversation_id), user_id=str(world.user_id)
    )
    assert [hit["child_id"] for hit in hits] == [str(chunk.id)]


async def test_payload_refuses_a_chunk_with_no_conversation_scope(env, world):
    """M2: no metadata and no document row → never index an ownerless point."""
    chunk = _chunk(world, content="ownerless")
    chunk.chunk_metadata.pop("conversation_id")
    chunk.document = None  # nothing to fall back to either

    with pytest.raises(ValueError, match="conversation_id"):
        await vector_retriever.upsert_chunks([chunk], user_id=str(world.user_id))

    assert _points([chunk.id]) == {}


# ── search: the pinned shape, scoped by tenant + conversation ───────────────


async def test_search_returns_the_pinned_shape(env, world):
    chunk = _chunk(world, content="searchable body")
    await _store(env, [chunk])
    await vector_retriever.upsert_chunks([chunk], user_id=str(world.user_id))

    hits = await vector_retriever.search(
        "searchable body", 5, str(world.conversation_id), user_id=str(world.user_id)
    )

    assert len(hits) == 1
    hit = hits[0]
    assert set(hit) == {"content", "score", "source", "rank", "metadata", "child_id", "parent_id"}
    assert hit["content"] == "searchable body"
    assert hit["source"] == "vector" and hit["rank"] == 0
    assert hit["child_id"] == str(chunk.id)
    assert hit["parent_id"] == chunk.chunk_metadata["parent_id"]
    assert hit["score"] == pytest.approx(1.0, abs=1e-6)  # cosine similarity, never 1 - dist
    assert hit["metadata"]["document_id"] == str(world.document_id)


async def test_search_is_scoped_to_the_conversation_and_the_tenant(env, world):
    stranger = await _add_user(env, "stranger@test.invalid")
    async with env() as db:
        other_conversation = Conversation(id=uuid.uuid4(), user_id=stranger, title="other")
        other_document = Document(
            id=uuid.uuid4(), conversation_id=other_conversation.id, filename="g.pdf",
            file_path="uploads/g.pdf",
        )
        db.add_all([other_conversation, other_document])
        await db.commit()

    mine = _chunk(world, content="mine")
    foreign = _chunk(world, content="foreign conversation",
                     conversation_id=other_conversation.id, document_id=other_document.id)
    await _store(env, [mine, foreign])
    await vector_retriever.upsert_chunks([mine], user_id=str(world.user_id))
    await vector_retriever.upsert_chunks([foreign], user_id=str(stranger))

    found = await vector_retriever.search(
        "mine", 5, str(world.conversation_id), user_id=str(world.user_id)
    )
    assert [hit["child_id"] for hit in found] == [str(mine.id)]

    # A foreign conversation is invisible even with the right tenant ...
    assert await vector_retriever.search(
        "foreign conversation", 5, str(other_conversation.id), user_id=str(world.user_id)
    ) == []
    # ... and a foreign tenant is invisible even with the right conversation.
    assert await vector_retriever.search(
        "mine", 5, str(world.conversation_id), user_id=str(stranger)
    ) == []
    # ... while the foreign point is really there, for its own tenant.
    assert [hit["child_id"] for hit in await vector_retriever.search(
        "foreign conversation", 5, str(other_conversation.id), user_id=str(stranger)
    )] == [str(foreign.id)]


async def test_search_of_a_conversation_with_no_points_is_empty(env, world):
    assert await vector_retriever.search(
        "anything", 5, str(world.conversation_id), user_id=str(world.user_id)
    ) == []


def test_search_requires_a_tenant():
    """I1: keyword-only and defaultless — no caller can drop the boundary."""
    with pytest.raises(TypeError):
        vector_retriever.search("q", 5, "cid")


def test_build_chunk_filter_pins_the_tenant_then_the_conversation():
    chunk_filter = build_chunk_filter("owner", "conv", document_id="doc")

    assert [condition.key for condition in chunk_filter.must] == [
        "user_id", "conversation_id", "document_id"
    ]
    assert chunk_filter.must[0].match.value == "owner"
    assert chunk_filter.must[1].match.value == "conv"
    assert chunk_filter.must[2].match.value == "doc"
    assert [condition.key for condition in build_chunk_filter("owner", "conv").must] == [
        "user_id", "conversation_id"
    ]


def test_build_chunk_filter_refuses_a_missing_or_empty_tenant():
    """I1: the tenant clause is mandatory, never omitted for a wider filter."""
    for missing in (None, "", "   "):
        with pytest.raises(ValueError, match="tenant"):
            build_chunk_filter(missing, "conv")


# ── document / conversation deletes: filtered, tenant-scoped, verified ──────


async def test_delete_document_chunks_removes_only_that_document(env, world):
    async with env() as db:
        second_document = Document(
            id=uuid.uuid4(), conversation_id=world.conversation_id, filename="h.pdf",
            file_path="uploads/h.pdf",
        )
        db.add(second_document)
        await db.commit()

    doomed = _chunk(world, content="doomed")
    survivor = _chunk(world, content="survivor", document_id=second_document.id)
    await _store(env, [doomed, survivor])
    await vector_retriever.upsert_chunks([doomed, survivor], user_id=str(world.user_id))

    assert await vector_retriever.delete_document_chunks(
        str(world.conversation_id), str(world.document_id), user_id=str(world.user_id)
    ) is True

    assert not _present(doomed.id)          # readback: the document is gone
    assert _present(survivor.id)            # ... and its sibling document is not


async def test_delete_conversation_chunks_removes_the_whole_conversation(env, world):
    stranger = await _add_user(env, "stranger2@test.invalid")
    async with env() as db:
        foreign_conversation = Conversation(id=uuid.uuid4(), user_id=stranger, title="other")
        foreign_document = Document(
            id=uuid.uuid4(), conversation_id=foreign_conversation.id, filename="g.pdf",
            file_path="uploads/g.pdf",
        )
        db.add_all([foreign_conversation, foreign_document])
        await db.commit()

    chunks = [_chunk(world, content=f"c{i}") for i in range(3)]
    foreign = _chunk(world, content="foreign", conversation_id=foreign_conversation.id,
                     document_id=foreign_document.id)
    await _store(env, [*chunks, foreign])
    await vector_retriever.upsert_chunks(chunks, user_id=str(world.user_id))
    await vector_retriever.upsert_chunks([foreign], user_id=str(stranger))

    assert await vector_retriever.delete_conversation_chunks(
        str(world.conversation_id), user_id=str(world.user_id)
    ) is True

    assert _points([chunk.id for chunk in chunks]) == {}
    assert _present(foreign.id)  # a foreign conversation is never swept in


async def test_a_foreign_tenant_cannot_delete_by_document(env, world):
    stranger = await _add_user(env, "stranger3@test.invalid")
    chunk = _chunk(world, content="not yours")
    await _store(env, [chunk])
    await vector_retriever.upsert_chunks([chunk], user_id=str(world.user_id))

    await vector_retriever.delete_document_chunks(
        str(world.conversation_id), str(world.document_id), user_id=str(stranger)
    )

    assert _present(chunk.id)


# ── intents: enqueued with the rows, quarantined ids never inserted ─────────


def test_chunk_intents_roll_back_with_the_rows(env, world):
    chunk = _chunk(world)
    with sync_session() as db:
        db.add(chunk)
        outbox.enqueue_chunk_upsert_sync(
            db, chunk_id=chunk.id, tenant_id=world.user_id, revision=1,
            conversation_id=world.conversation_id,
        )
        db.rollback()

    with sync_session() as db:
        assert db.execute(select(IndexOutbox)).scalars().all() == []
        assert db.get(DocumentChunk, chunk.id) is None


def test_chunk_upsert_intent_refuses_a_non_positive_revision(env, world):
    """M1: a revision-0 upsert would dedupe onto an earlier one (lost write)."""
    chunk = _chunk(world)
    with sync_session() as db:
        with pytest.raises(ValueError, match="revision"):
            outbox.enqueue_chunk_upsert_sync(
                db, chunk_id=chunk.id, tenant_id=world.user_id, revision=0,
                conversation_id=world.conversation_id,
            )

    with sync_session() as db:
        assert db.execute(select(IndexOutbox)).scalars().all() == []


async def test_malformed_chunk_ids_are_quarantined_not_fixed(env, world):
    async with env() as db:
        quarantined = await outbox.enqueue_chunk_delete(
            db,
            chunk_ids=[str(world.document_id), "legacy-chunk-1"],
            tenant_id=world.user_id,
        )
        await db.commit()

    assert quarantined == ["legacy-chunk-1"]
    rows = await _intents()
    assert [(row.kind, row.entity_id, row.operation) for row in rows] == [
        ("chunk", world.document_id.hex, "delete")
    ]


def test_sync_upsert_intent_quarantines_a_non_uuid_id(env, world):
    with sync_session() as db:
        quarantined = outbox.enqueue_chunk_upsert_sync(
            db, chunk_id="not-a-uuid", tenant_id=world.user_id, revision=1,
            conversation_id=world.conversation_id,
        )
        db.commit()

    assert quarantined == ["not-a-uuid"]
    with sync_session() as db:
        assert db.execute(select(IndexOutbox)).scalars().all() == []


# ── drain: the chunk kind applies, skips, forgets, verifies ─────────────────


async def test_drain_applies_a_chunk_upsert_from_the_sql_row(env, world):
    chunk = _chunk(world, content="drained chunk")
    with sync_session() as db:
        db.add(chunk)
        outbox.enqueue_chunk_upsert_sync(
            db, chunk_id=chunk.id, tenant_id=world.user_id, revision=1,
            conversation_id=world.conversation_id,
        )
        db.commit()

    report = await outbox.drain_pending()

    assert report == {"claimed": 1, "applied": 1, "skipped": 0, "blocked": 0, "failed": 0}
    assert _points([chunk.id])[str(chunk.id)]["content"] == "drained chunk"
    assert [row.status for row in await _intents()] == ["done"]


async def test_drain_skips_a_stale_chunk_revision(env, world):
    chunk = _chunk(world, content="v1", revision=2)
    with sync_session() as db:
        db.add(chunk)
        outbox.enqueue_chunk_upsert_sync(
            db, chunk_id=chunk.id, tenant_id=world.user_id, revision=1,
            conversation_id=world.conversation_id,
        )
        db.commit()

    assert (await outbox.drain_pending())["skipped"] == 1
    assert not _present(chunk.id)  # an older revision never owns the point

    with sync_session() as db:
        outbox.enqueue_chunk_upsert_sync(
            db, chunk_id=chunk.id, tenant_id=world.user_id, revision=2,
            conversation_id=world.conversation_id,
        )
        db.commit()
    assert (await outbox.drain_pending())["applied"] == 1
    assert _points([chunk.id])[str(chunk.id)]["content"] == "v1"


async def test_drain_forgets_the_point_when_the_chunk_row_is_gone(env, world):
    chunk = _chunk(world, content="soon gone")
    await _store(env, [chunk])
    await vector_retriever.upsert_chunks([chunk], user_id=str(world.user_id))
    assert _present(chunk.id)

    with sync_session() as db:
        outbox.enqueue_chunk_upsert_sync(
            db, chunk_id=chunk.id, tenant_id=world.user_id, revision=1,
            conversation_id=world.conversation_id,
        )
        db.commit()
    async with env() as db:
        await db.delete(chunk)
        await db.commit()

    assert (await outbox.drain_pending())["applied"] == 1
    assert not _present(chunk.id)


async def test_drain_acks_a_chunk_delete_only_after_the_readback(env, world):
    chunk = _chunk(world, content="deleted by intent")
    await _store(env, [chunk])
    await vector_retriever.upsert_chunks([chunk], user_id=str(world.user_id))

    async with env() as db:
        await outbox.enqueue_chunk_delete(
            db, chunk_ids=[chunk.id], tenant_id=world.user_id
        )
        await db.commit()
        await db.delete(chunk)
        await db.commit()

    assert (await outbox.drain_pending())["applied"] == 1
    assert not _present(chunk.id)
    assert [row.status for row in await _intents()] == ["done"]


async def test_an_unconfirmed_chunk_delete_stays_pending(env, world, monkeypatch):
    async def not_confirmed(_chunk_ids):
        return False

    monkeypatch.setattr(outbox, "delete_chunks", not_confirmed)

    chunk = _chunk(world)
    await _store(env, [chunk])
    async with env() as db:
        await outbox.enqueue_chunk_delete(db, chunk_ids=[chunk.id], tenant_id=world.user_id)
        await db.commit()
        await db.delete(chunk)
        await db.commit()

    report = await outbox.drain_pending()

    assert report == {"claimed": 1, "applied": 0, "skipped": 0, "blocked": 0, "failed": 1}
    row = (await _intents())[0]
    assert (row.status, row.attempts) == ("pending", 1)
    assert row.last_error


# ── generation: one chunk collection per generation, not per conversation ───


def test_the_chunk_fallback_generation_is_not_the_memory_one():
    assert outbox.CHUNK_TARGET_GENERATION == f"{COLLECTION_NAME}__chunks"
    assert outbox.CHUNK_TARGET_GENERATION != outbox.TARGET_GENERATION


async def test_a_chunk_generation_pointer_flip_moves_the_writes(env, world):
    chunk = _chunk(world, content="pointer body")
    await _store(env, [chunk])
    await vector_retriever.upsert_chunks([chunk], user_id=str(world.user_id))
    assert _present(chunk.id, generation=outbox.CHUNK_TARGET_GENERATION)

    await _activate(env, OTHER_GENERATION)

    # The pointer moved: the new generation does not even hold the old point ...
    assert not _present(chunk.id, generation=OTHER_GENERATION)
    assert await vector_retriever.search(
        "pointer body", 5, str(world.conversation_id), user_id=str(world.user_id)
    ) == []
    # ... while the previous generation keeps its copy (cutover is a flip).
    assert _present(chunk.id, generation=outbox.CHUNK_TARGET_GENERATION)

    other = _chunk(world, content="after flip")
    await _store(env, [other])
    await vector_retriever.upsert_chunks([other], user_id=str(world.user_id))
    assert _present(other.id, generation=OTHER_GENERATION)
