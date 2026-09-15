"""P3 T2 — no drain-vs-purge orphan window on document reingest.

The P1b review's M6, reachable the moment the drain became a background loop: a
reingest commits its rows + intents and then touches the vector store. A drain
that claims the new children's pending upsert intents in that window writes
those points and acks them ``done`` — and the old document-wide FILTERED purge
then deleted exactly those acked points. When the fast path's own write did not
survive (the outage the pipeline tolerates by design: "the intents stay
pending"), the result was a live SQL row with no point and no pending intent —
unrecoverable.

The interleaving is deterministic (ruling R2): a real ``drain_once`` batch
called from a monkeypatched seam INSIDE the window — after the canonical
commit, before the index work — never a sleep or a timing guess.

Isolation is the ``tests/retrieval/test_chunk_index.py`` pattern: real embedded
Qdrant on a private ``tmp_path`` folder (closed in teardown, so its folder lock
never leaks into another test), a private per-test SQLite file monkeypatched
into the module engines / sessionmakers, deterministic unit-vector embeddings.
The SQL rows, the intents and the points are all real; only the out-of-process
seams (object storage, Redis, BM25, the doc→memory projection) are stubbed.
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import random
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from app import database
from app import models as _models  # noqa: F401 — register every table on Base
from app.config import settings
from app.database import Base, sync_session
from app.ingestion import pipeline
from app.models.conversation import Conversation
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.index_outbox import IndexGeneration, IndexOutbox
from app.models.user import User
from app.retrieval import vector_backend, vector_retriever
from app.retrieval.embedding_fingerprint import canonical_fingerprint, fingerprint_generation
from app.retrieval.memory import drain_loop, outbox

pytestmark = pytest.mark.rag

DIM = 8
FENCE_DB = "purge-fence.sqlite"
# Generous: one window batch must be able to claim every intent one reingest
# enqueues, or the fence test would be measuring the batch size, not the fence.
BATCH = 500
# The embedding contract this suite pins (the ambient settings must not decide
# it: the same tests have to hold on a 384-dim lite install and a 1536-dim one).
FINGERPRINT = {
    "model_id": "test-model",
    "model_revision": "revision-1",
    "dim": DIM,
    "provider": "test",
}


def _fingerprint() -> dict:
    return dict(FINGERPRINT)


def _manifest_token() -> str:
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


# ── fixture: private SQLite + embedded Qdrant + one owned document ──────────


@pytest.fixture
def fence(tmp_path, monkeypatch):
    """A private store, a private DB, and one owner with one document."""
    folder = tmp_path / "qdrant"
    folder.mkdir()
    monkeypatch.setattr(settings, "QDRANT_MODE", "local")
    monkeypatch.setattr(settings, "QDRANT_LOCAL_PATH", str(folder))

    url = f"sqlite+aiosqlite:///{tmp_path / FENCE_DB}"
    engine = create_async_engine(
        url, connect_args={"check_same_thread": False}, poolclass=NullPool
    )
    event.listen(engine.sync_engine, "connect", database._configure_sqlite_connection)
    sessions = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )
    sync_engine = create_engine(
        url.replace("+aiosqlite", ""), connect_args={"check_same_thread": False}
    )
    event.listen(sync_engine, "connect", database._configure_sqlite_connection)
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

    # Out-of-process seams only: object storage, the Redis parent cache, BM25,
    # the retrieval cache, and the doc→memory projection (its own suite).
    monkeypatch.setattr("app.storage.get_object_sync", lambda *a, **k: b"fence bytes")
    monkeypatch.setattr("app.utils.chunker.extract_text", lambda *a, **k: "Fence body. " * 140)
    monkeypatch.setattr("app.retrieval.parent_store.store_parents_sync", lambda *a, **k: None)
    monkeypatch.setattr(
        "app.retrieval.bm25_retriever.bm25_retriever.publish_build_sync", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "app.retrieval.retrieval_cache.invalidate_query_cache_sync", lambda *a, **k: None
    )
    monkeypatch.setattr(pipeline, "_project_document_to_memories", lambda *a, **k: None)

    # A client cached by an earlier test in this process (server mode, another
    # tmp folder) would poison this one: reset the module's client cache first.
    asyncio.run(vector_backend.close_clients())

    Base.metadata.create_all(sync_engine)
    user_id, conversation_id, document_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    with sync_session() as db:
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
        db.add(Conversation(id=conversation_id, user_id=user_id, title="conv", document_count=1))
        db.add(
            Document(
                id=document_id,
                conversation_id=conversation_id,
                filename="fence.pdf",
                file_path="uploads/fence.pdf",
            )
        )
        db.add(
            IndexGeneration(
                id=uuid.uuid4().hex,
                kind="chunk",
                generation=outbox.CHUNK_TARGET_GENERATION,
                fingerprint=_manifest_token(),
                is_active=True,
            )
        )
        db.commit()

    async def _close():
        await vector_backend.close_clients()
        await engine.dispose()

    try:
        yield SimpleNamespace(
            user_id=user_id, conversation_id=conversation_id, document_id=document_id
        )
    finally:
        asyncio.run(_close())
        sync_engine.dispose()


# ── helpers: real SQL, real store ───────────────────────────────────────────


def _row_ids(fence, document_id=None) -> set[str]:
    with sync_session() as db:
        rows = (
            db.execute(
                select(DocumentChunk).where(
                    DocumentChunk.document_id == (document_id or fence.document_id)
                )
            )
            .scalars()
            .all()
        )
        return {str(row.id) for row in rows}


def _child_ids(fence, document_id=None) -> set[str]:
    """The document's CHILD rows — the only ones the chunk index carries."""
    with sync_session() as db:
        rows = (
            db.execute(
                select(DocumentChunk).where(
                    DocumentChunk.document_id == (document_id or fence.document_id)
                )
            )
            .scalars()
            .all()
        )
        return {
            str(row.id)
            for row in rows
            if (row.chunk_metadata or {}).get("chunk_type") == "child"
        }


def _stored_ids() -> set[str]:
    """Every point in the active chunk generation, read through the real client."""
    client = vector_backend.get_sync_client()
    name = outbox.CHUNK_TARGET_GENERATION
    if not client.collection_exists(name):
        return set()
    points, _ = client.scroll(collection_name=name, limit=BATCH, with_payload=False)
    return {str(point.id) for point in points}


def _pending(fence) -> list[str]:
    """The chunk intents still owed to the store — the replayable residue."""
    with sync_session() as db:
        rows = (
            db.execute(
                select(IndexOutbox).where(
                    IndexOutbox.kind == "chunk", IndexOutbox.status == "pending"
                )
            )
            .scalars()
            .all()
        )
        return [f"{row.operation}:{row.entity_id}" for row in rows]


def _drain() -> dict:
    """One real drain batch through the shipped single door (ruling R1)."""
    return asyncio.run(drain_loop.drain_once(batch_size=BATCH))


def _ingest(fence) -> None:
    """Run the real ingestion pipeline on the fence document."""
    with sync_session() as db:
        pipeline._ingest(db, str(fence.document_id))


def _install_window_drain(monkeypatch, observed: dict) -> None:
    """Drive the interleaving: a drain batch in the window (ruling R2).

    The seam is the pipeline's post-commit index step: the canonical commit has
    happened (rows + their intents are durable), the store has not been touched
    yet. That is exactly the window a concurrent drain can land in — and it is
    entered by CALLING the drain there, never by racing a timer.
    """
    real = pipeline._index_document_chunks

    def window_drain(db, *args, **kwargs):
        observed["report"] = _drain()
        observed["present_after_drain"] = _stored_ids()
        return real(db, *args, **kwargs)

    monkeypatch.setattr(pipeline, "_index_document_chunks", window_drain)


# ── the fence (M6): a drain's acked write is never purged away ──────────────


def test_a_window_drain_survives_a_failing_fast_path(fence, monkeypatch):
    """RED on the document-wide purge: live rows, no point, no pending intent.

    The window drain writes and acks the new generation; the fast path's own
    write then fails (the outage ``_ingest`` tolerates) — so the drain's write
    is the ONLY copy of those points. Nothing may take it away: the purge must
    not name a live id.
    """
    _ingest(fence)
    first = _child_ids(fence)
    assert first and first <= _stored_ids()

    observed: dict = {}
    _install_window_drain(monkeypatch, observed)

    def outage(*args, **kwargs):
        raise RuntimeError("vector backend unavailable")

    monkeypatch.setattr(vector_retriever, "upsert_chunks_sync", outage)

    _ingest(fence)  # reingest: best-effort indexing, never raises

    second = _child_ids(fence)
    assert second and first.isdisjoint(second)  # a reingest mints new ids
    report = observed["report"]
    assert report["claimed"] > 0 and report["applied"] == report["claimed"], report
    assert second <= observed["present_after_drain"], "the drain wrote them in the window"

    # Everything is acked: no pending intent is left to replay a lost point.
    assert _pending(fence) == [], "the drain's acks are final"

    present = _stored_ids()
    assert not (second - present), (
        "live rows with no point and no pending intent: the purge deleted points "
        "the drain had already written and acked"
    )
    assert first.isdisjoint(present)  # and the generation that left SQL is gone


def test_a_window_drain_leaves_every_live_row_a_point(fence, monkeypatch):
    """The same interleaving with a healthy fast path: same invariant holds."""
    _ingest(fence)
    first = _child_ids(fence)

    observed: dict = {}
    _install_window_drain(monkeypatch, observed)
    _ingest(fence)

    second = _child_ids(fence)
    assert observed["report"]["applied"] > 0
    assert second <= observed["present_after_drain"]

    present = _stored_ids()
    assert second <= present
    assert first.isdisjoint(present)
    assert set(_row_ids(fence)) >= second  # the rows the points belong to are live


def test_healthy_fast_path_purges_the_ids_that_left_sql(fence):
    """The fast path alone, no drain anywhere (ruling R9).

    The two tests above interleave a drain, which claims the old ids' delete
    intents *before* the fast path runs — so they hold even with the purge call
    gone. This one leaves the window empty: the first generation's points are in
    the store from the first ingest's own write, and the only thing that can
    take them away is the id-scoped purge at the end of ``_index_document_chunks``.
    """
    _ingest(fence)
    first = _child_ids(fence)
    assert first and first <= _stored_ids()

    _ingest(fence)  # healthy: the fast path runs to completion

    second = _child_ids(fence)
    assert second and first.isdisjoint(second)  # a reingest mints new ids
    present = _stored_ids()
    assert second <= present, "the fast path wrote the new generation"
    assert first.isdisjoint(present), (
        "the old generation's rows left SQL but its points are still in the "
        "store: the fast-path purge did not run"
    )


# ── R8: the purge is id-scoped and still tenant-scoped ─────────────────────


def _chunk(fence, *, document_id=None, conversation_id=None, content="chunk body",
           index=0) -> DocumentChunk:
    document_id = document_id or fence.document_id
    conversation_id = conversation_id or fence.conversation_id
    return DocumentChunk(
        id=uuid.uuid4(),
        document_id=document_id,
        content=content,
        chunk_index=index,
        revision=1,
        chunk_metadata={
            "document_id": str(document_id),
            "conversation_id": str(conversation_id),
            "chunk_type": "child",
            "child_index": index,
            "parent_id": str(uuid.uuid4()),
        },
    )


def test_delete_chunks_by_ids_is_id_scoped_and_tenant_scoped(fence):
    """R8: it deletes the NAMED ids, in the caller's own scope, and nothing else.

    A neighbouring live document in the same conversation survives, a foreign
    tenant can never be reached even by naming its ids, and the count readback
    is what reports the deletion.
    """
    stranger = uuid.uuid4()
    stranger_conversation = uuid.uuid4()
    stranger_document = uuid.uuid4()
    neighbour_document = uuid.uuid4()
    with sync_session() as db:
        db.add(
            User(
                id=stranger,
                email=f"{stranger.hex}@test.invalid",
                hashed_password="x",
                display_name="Stranger",
                is_verified=True,
                is_active=True,
            )
        )
        db.add(Conversation(id=stranger_conversation, user_id=stranger, title="other"))
        db.add_all(
            [
                Document(
                    id=stranger_document,
                    conversation_id=stranger_conversation,
                    filename="g.pdf",
                    file_path="uploads/g.pdf",
                ),
                Document(
                    id=neighbour_document,
                    conversation_id=fence.conversation_id,
                    filename="h.pdf",
                    file_path="uploads/h.pdf",
                ),
            ]
        )
        db.commit()

    doomed = [_chunk(fence, index=index) for index in range(2)]
    neighbour = _chunk(fence, document_id=neighbour_document, content="neighbour")
    foreign = _chunk(
        fence, document_id=stranger_document, conversation_id=stranger_conversation,
        content="foreign",
    )
    with sync_session() as db:
        for row in (*doomed, neighbour, foreign):
            db.add(row)
        db.commit()
    vector_retriever.upsert_chunks_sync(doomed, user_id=str(fence.user_id))
    vector_retriever.upsert_chunks_sync([neighbour], user_id=str(fence.user_id))
    vector_retriever.upsert_chunks_sync([foreign], user_id=str(stranger))

    deleted = vector_retriever.delete_chunks_by_ids(
        [str(row.id) for row in (*doomed, foreign)],
        user_id=str(fence.user_id),
        conversation_id=str(fence.conversation_id),
    )

    # The owner's two ids are gone; the neighbour in the same conversation and
    # the foreign tenant (named in the same call!) are untouched.
    assert deleted == 2
    present = _stored_ids()
    assert not ({str(row.id) for row in doomed} & present)
    assert {str(neighbour.id), str(foreign.id)} <= present

    # A caller naming ids outside its own scope deletes nothing.
    vector_retriever.delete_chunks_by_ids(
        [str(neighbour.id)], user_id=str(stranger), conversation_id=str(stranger_conversation)
    )
    assert str(neighbour.id) in _stored_ids()

    # Nothing named: a no-op, never a document-wide sweep.
    assert vector_retriever.delete_chunks_by_ids(
        [], user_id=str(fence.user_id), conversation_id=str(fence.conversation_id)
    ) == 0
    assert str(neighbour.id) in _stored_ids()
