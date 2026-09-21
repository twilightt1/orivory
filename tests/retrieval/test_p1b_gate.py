"""P1b acceptance gate — the plan's §9 P1 row, end to end, over REAL stores.

Everything here runs against a real SQLite file and a real embedded Qdrant
folder; nothing about the vector store or the SQL layer is mocked. The only
substitutions are the seams that are out of process in production:

* the embedder — deterministic unit vectors, because no claim in this file is
  about embedding QUALITY (the mean-vs-CLS ablation and the P0 parity gate own
  that). The embedding CONTRACT (arctic CLS, dim 384) is the real one, so every
  manifest row, generation name and contract guard is exercised for real;
* MinIO/Redis in the ingestion test — object storage and the BM25 parent cache
  are external services; the chunk rows, the durable intents and the vector
  readback are real;
* the erasure outage test, which impersonates a DOWN vector store to prove the
  receipt stays honest — the recovery half of that test then runs for real.

Coverage (ruling R42): memory round trip across a close/reopen; migrate+resume
of the same batch twice; the crash windows at commit, ack and flip; count/ID/
revision coverage after cutover; tenant-injection negatives (search/delete/
presence); correction dirty/old rows; erasure outage/retry/depth/derived;
document reingest+delete orphans; an unconfirmed chunk delete left PENDING (not
acked); the WAL backup restore drill; the rollback's
reconcile of post-cutover writes; the blocked second local owner; no
multiworker fallback; and the "image needs no Chroma" path, proven in a CHILD
interpreter whose ``import chromadb`` raises.

Isolated by construction: a private per-test SQLite file + a private embedded
Qdrant folder, both monkeypatched in as the app's own (the
``tests/migration/test_p1b_migrate.py`` pattern), so this suite can never read
or write whatever ``DATABASE_URL`` is ambient.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import textwrap
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
from qdrant_client import QdrantClient
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
from app.models.memory import Memory, MemorySuppression
from app.models.user import User
from app.retrieval import vector_backend, vector_retriever
from app.retrieval.embedding_fingerprint import (
    ARCTIC_CLS_FINGERPRINT,
    generation_name,
)
from app.retrieval.memory import freshness, namespaces, outbox, vector_store

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPO_ROOT / "scripts" / "migrate_qdrant.py"
ROLLBACK_PATH = REPO_ROOT / "scripts" / "rollback_to_chroma.py"

# The PRODUCTION local contract: arctic XS, CLS pooling, dim 384 (the rollback
# tool rebuilds at the same width, so the stub embedder must produce it).
DIM = int(ARCTIC_CLS_FINGERPRINT["dim"])
DB_NAME = "p1b-gate.db"
FORGOTTEN_SOURCE = "drive://gate-forgotten"


# ── deterministic embeddings (never the claim under test) ───────────────────


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


# ── the migration CLI + the rollback utility, loaded by path ────────────────


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def cli():
    """``scripts/migrate_qdrant.py`` — the offline migration CLI, once."""
    return _load("migrate_qdrant", CLI_PATH)


@pytest.fixture(scope="session")
def rollback_tool(cli):
    """``scripts/rollback_to_chroma.py``; skips when chromadb is gone (T7)."""
    pytest.importorskip("chromadb")
    return _load("rollback_to_chroma", ROLLBACK_PATH)


# ── the store: a private SQLite file + a private embedded Qdrant folder ─────


def _sync_engine(url: str):
    engine = create_engine(
        url.replace("+aiosqlite", ""), connect_args={"check_same_thread": False}
    )
    event.listen(engine, "connect", database._configure_sqlite_connection)
    return engine


def _bind_engines(url: str, monkeypatch):
    """Fresh async+sync engines for ``url``, bound as the app's own.

    Also the "restart" primitive: new engines over the same committed file.
    """
    engine = create_async_engine(
        url, connect_args={"check_same_thread": False}, poolclass=NullPool
    )
    event.listen(engine.sync_engine, "connect", database._configure_sqlite_connection)
    sessions = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )
    sync_engine = _sync_engine(url)
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    monkeypatch.setattr(database, "AsyncSessionLocal", sessions)
    monkeypatch.setattr(outbox, "AsyncSessionLocal", sessions)  # the drain's own
    monkeypatch.setattr(freshness, "AsyncSessionLocal", sessions)  # the R14 barrier's count
    monkeypatch.setattr(
        database,
        "_get_sync_sessionmaker",
        lambda: sessionmaker(bind=sync_engine, expire_on_commit=False, autoflush=False),
    )
    return engine, sync_engine, sessions


@pytest_asyncio.fixture
async def env(tmp_path, monkeypatch, cli):
    """Private stores + the ambient decision (arctic CLS) pinned, nothing else."""
    folder = tmp_path / "qdrant"
    folder.mkdir()
    monkeypatch.setattr(settings, "QDRANT_MODE", "local")
    monkeypatch.setattr(settings, "QDRANT_LOCAL_PATH", str(folder))
    # Nothing listens on port 1, so the CLI's quiesce probe passes (a dev
    # machine may well have something on the default 8000).
    monkeypatch.setattr(settings, "APP_PORT", 1)
    monkeypatch.setattr(settings, "USE_LOCAL_EMBEDDINGS", True)
    monkeypatch.setattr(settings, "LOCAL_EMBED_MODEL", "arctic")
    monkeypatch.setattr(settings, "FS_STORAGE_PATH", str(tmp_path / "uploads"))
    monkeypatch.setattr(settings, "LEGACY_CHROMA_PATH", str(tmp_path / "no-chroma"))

    db_path = tmp_path / DB_NAME
    url = f"sqlite+aiosqlite:///{db_path}"
    monkeypatch.setattr(settings, "DATABASE_URL", url)

    from app.retrieval.memory import retriever as retriever_module

    for module in (vector_store, vector_retriever, retriever_module, cli):
        monkeypatch.setattr(module, "embed_texts", _fake_embed, raising=False)
        monkeypatch.setattr(module, "embed_texts_sync", _fake_embed_sync, raising=False)
        monkeypatch.setattr(module, "embed_query", _fake_embed_query, raising=False)
    # The knowledge-graph build is an LLM call, not part of this gate: the
    # write-back path calls it best-effort and it must not decide anything.
    monkeypatch.setattr("app.graph.builder.build_memory_graph_sync", lambda *a, **k: None)

    engine, sync_engine, sessions = _bind_engines(url, monkeypatch)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield SimpleNamespace(
            tmp_path=tmp_path, db_path=db_path, url=url, qdrant_dir=folder,
            sessions=sessions, sync_engine=sync_engine, cli=cli,
        )
    finally:
        await vector_backend.close_clients()
        await engine.dispose()
        sync_engine.dispose()


def _user(db, email: str) -> User:
    return User(id=uuid.uuid4(), email=email, hashed_password="x",
                display_name=email.split("@")[0], is_verified=True, is_active=True)


def _memory(user_id, content: str, **kwargs) -> Memory:
    kwargs.setdefault("captured_at", datetime.now(UTC))
    return Memory(id=uuid.uuid4(), user_id=user_id, content=content, tags=[], **kwargs)


@pytest_asyncio.fixture
async def world(env):
    """Alice (current/superseded/dirty/forgotten) + Bob (two current), real chunks.

    Seeded through SQL, like the operator's live data; the ladder has run, so
    the two P1b generation rows exist INACTIVE (the expand never moves the
    pointer: F1) and the CLI is what flips them.
    """
    await database.bootstrap_sqlite()
    async with env.sessions() as db:
        alice, bob = _user(db, "alice@gate.invalid"), _user(db, "bob@gate.invalid")
        db.add_all([alice, bob])
        await db.flush()
        alice_conv = Conversation(id=uuid.uuid4(), user_id=alice.id, document_count=1)
        bob_conv = Conversation(id=uuid.uuid4(), user_id=bob.id, document_count=1)
        db.add_all([alice_conv, bob_conv])
        alice_doc = Document(id=uuid.uuid4(), conversation_id=alice_conv.id, filename="gate.txt",
                             file_path="uploads/gate.txt", chunk_count=2)
        bob_doc = Document(id=uuid.uuid4(), conversation_id=bob_conv.id, filename="bob.txt",
                           file_path="uploads/bob.txt", chunk_count=1)
        db.add_all([alice_doc, bob_doc])

        def chunk(content: str, document, index: int) -> DocumentChunk:
            row = DocumentChunk(
                id=uuid.uuid4(), document_id=document.id, content=content, chunk_index=index,
                revision=1,
                chunk_metadata={
                    "document_id": str(document.id),
                    "conversation_id": str(document.conversation_id),
                    "filename": document.filename, "chunk_type": "child", "child_index": index,
                },
            )
            db.add(row)
            return row

        alice_chunks = [chunk("gate chunk 0", alice_doc, 0), chunk("gate chunk 1", alice_doc, 1)]
        bob_chunks = [chunk("bob chunk 0", bob_doc, 0)]

        alice_current = _memory(alice.id, "alice current")
        alice_superseded = _memory(alice.id, "alice old fact",
                                   extra_metadata={"cm_superseded_by": "later",
                                                   "cm_subject": "alice"})
        alice_dirty = _memory(alice.id, "alice dirty",
                              extra_metadata={"cm_derived_dirty": True})
        alice_forgotten = _memory(alice.id, "alice forgotten", source_type="file_upload",
                                  source_ref=FORGOTTEN_SOURCE)
        bob_current = _memory(bob.id, "bob current")
        bob_extra = _memory(bob.id, "bob extra")
        db.add_all([alice_current, alice_superseded, alice_dirty, alice_forgotten,
                    bob_current, bob_extra])
        db.add(MemorySuppression(id=uuid.uuid4().hex, user_id=alice.id,
                                 source_ref=FORGOTTEN_SOURCE, reason="user_forget"))
        await db.commit()
        rows = SimpleNamespace(
            alice=alice, bob=bob, alice_conv=alice_conv, bob_conv=bob_conv,
            alice_doc=alice_doc, bob_doc=bob_doc,
            alice_chunks=alice_chunks, bob_chunks=bob_chunks,
            alice_current=alice_current, alice_superseded=alice_superseded,
            alice_dirty=alice_dirty, alice_forgotten=alice_forgotten,
            bob_current=bob_current, bob_extra=bob_extra,
        )
    return SimpleNamespace(**vars(env), **vars(rows))


@pytest_asyncio.fixture
async def live(world):
    """The finished install: both kinds backfilled, then cut over.

    This is the state every "after cutover" claim in this file is made in.
    """
    world.cli.backfill(kind="memory", batch=4)
    world.cli.backfill(kind="chunk", batch=4)
    world.cli.cutover(yes=True)
    return world


# ── small readers (all through the real store) ──────────────────────────────


def _client():
    return vector_backend.get_sync_client()


def _scroll_ids(generation: str) -> set[str]:
    points, _ = _client().scroll(collection_name=generation, limit=1024, offset=None,
                                 with_payload=False, with_vectors=False)
    return {str(point.id) for point in points}


def _payloads(generation: str) -> dict[str, dict]:
    points, _ = _client().scroll(collection_name=generation, limit=1024, offset=None,
                                 with_payload=True, with_vectors=False)
    return {str(point.id): dict(point.payload or {}) for point in points}


async def _manifest(env) -> dict[str, IndexGeneration]:
    async with env.sessions() as db:
        rows = (await db.execute(select(IndexGeneration))).scalars().all()
        return {row.kind: row for row in rows if row.is_active}


async def _intents(env) -> list[IndexOutbox]:
    async with env.sessions() as db:
        return list((await db.execute(
            select(IndexOutbox).order_by(IndexOutbox.seq))).scalars().all())


async def _restart(env, monkeypatch):
    """The process died: its engines go, its committed file stays."""
    await vector_backend.close_clients()
    await database.engine.dispose()
    env.sync_engine.dispose()
    engine, sync_engine, sessions = _bind_engines(env.url, monkeypatch)
    env.sessions, env.sync_engine = sessions, sync_engine  # type: ignore[attr-defined]
    return engine


async def _create_memory(owner_id, content: str, *, title: str | None = None):
    """A real API write: row + durable intent + the best-effort fast path."""
    from app.api.v1 import memories as memories_api
    from app.schemas.Orivory import MemoryCreate

    async with database.AsyncSessionLocal() as db:
        return await memories_api.create_memory(
            MemoryCreate(content=content, title=title), SimpleNamespace(id=owner_id), db
        )


# ── 1. the memory round trip, across a close/reopen ─────────────────────────


async def test_memory_round_trip_survives_close_and_reopen(live, monkeypatch):
    """Add → update → search → presence → delete, then close the store and
    reopen it: the committed state is what the next process reads."""
    from app.api.v1 import memories as memories_api
    from app.schemas.Orivory import MemoryUpdate

    created = await _create_memory(live.alice.id, "gate round trip", title="Round trip")
    generation = generation_name("memory")
    assert created.indexing == "ready"  # the fast path indexed it in-band
    assert str(created.id) in _scroll_ids(generation)

    async with database.AsyncSessionLocal() as db:
        updated = await memories_api.update_memory(
            created.id, MemoryUpdate(title="Round trip v2"), SimpleNamespace(id=live.alice.id), db
        )
    assert updated.revision == 2

    async with database.AsyncSessionLocal() as db:
        row = (await db.execute(select(Memory).where(Memory.id == created.id))).scalars().one()
        query = _vector_for(vector_store._memory_to_document(row))
    hits = await vector_store.search_memories(query, user_id=str(live.alice.id), top_k=3)
    assert hits[0]["memory_id"] == str(created.id)
    assert hits[0]["score"] == pytest.approx(1.0)  # cosine similarity, best first
    assert hits[0]["metadata"]["orivory_memory_revision"] == 2  # the update landed
    assert hits[0]["metadata"]["orivory_embed_generation"] == live.cli.fingerprint_token()
    assert await vector_store.get_memory_ids_present([str(created.id)]) == {str(created.id)}

    # Close the store, kill the process's engines, come back up.
    engine = await _restart(live, monkeypatch)
    try:
        assert str(created.id) in _scroll_ids(generation)
        absent = str(uuid.uuid4())
        assert await vector_store.get_memory_ids_present([str(created.id), absent]) == {
            str(created.id)}
        hits = await vector_store.search_memories(query, user_id=str(live.alice.id), top_k=3)
        assert hits[0]["memory_id"] == str(created.id)

        async with database.AsyncSessionLocal() as db:
            await memories_api.delete_memory(created.id, SimpleNamespace(id=live.alice.id), db)
        assert await vector_store.get_memory_ids_present([str(created.id)]) == set()
        assert str(created.id) not in _scroll_ids(generation)
        # The delete is durable work the outbox owes, not a hope: drain acks it.
        await outbox.drain_pending()
        by_id = [row for row in await _intents(live) if row.entity_id == created.id.hex]
        assert by_id and {row.status for row in by_id} == {"done"}
    finally:
        await engine.dispose()


# ── 2. migrate + resume of the same batch twice ─────────────────────────────


async def test_migrate_resume_twice_changes_no_id_and_duplicates_nothing(world):
    """The same batch, migrated twice: same ids, same payloads, no doubles."""
    generation = generation_name("memory")
    first = world.cli.backfill(kind="memory", batch=2)
    assert first["complete"] is True and first["upserted"] == 3  # 3 eligible rows
    ids_after_first = _scroll_ids(generation)
    payloads = _payloads(generation)
    assert len(ids_after_first) == 3 == int(_client().count(generation).count)

    # Resume: the checkpoint says where the last batch stopped, so nothing is
    # re-walked, re-embedded or re-written.
    resumed = world.cli.backfill(kind="memory", batch=2, resume=True)
    assert resumed["resumed_from"] == first["last_id"]
    assert (resumed["upserted"], resumed["batches"], resumed["gc_deleted"]) == (0, 0, 0)
    assert _scroll_ids(generation) == ids_after_first
    assert _payloads(generation) == payloads

    # A second FULL run (no resume) re-writes the same three points — the point
    # id is the memory id, so the set is unchanged and the count never doubles.
    again = world.cli.backfill(kind="memory", batch=2)
    assert again["upserted"] == 3
    assert _scroll_ids(generation) == ids_after_first
    assert int(_client().count(generation).count) == 3
    assert _payloads(generation) == payloads


# ── 3. crash window: commit → die → restart → drain ─────────────────────────


async def test_crash_after_the_commit_before_the_drain_replays_on_restart(live, monkeypatch):
    """The row and its intent are committed while the store is down; the
    restart's drain is what puts the vector there — exactly once."""
    from app.api.v1 import memories as memories_api

    async def store_down(_memory) -> bool:
        return False  # the fast path did not index: the intent is the only proof

    monkeypatch.setattr(memories_api, "index_new_memory", store_down)
    created = await _create_memory(live.alice.id, "survives the crash")

    assert created.indexing == "pending"
    assert str(created.id) not in _scroll_ids(generation_name("memory"))
    assert [(row.operation, row.status, row.revision) for row in await _intents(live)
            if row.entity_id == created.id.hex] == [("upsert", "pending", 1)]

    # The process dies; only the committed SQLite file and the Qdrant folder stay.
    engine = await _restart(live, monkeypatch)
    try:
        report = await outbox.drain_pending()
        assert report["applied"] == 1
        assert str(created.id) in _scroll_ids(generation_name("memory"))
        assert [row.status for row in await _intents(live)
                if row.entity_id == created.id.hex] == ["done"]
    finally:
        await engine.dispose()


async def test_crash_after_the_vector_write_before_the_ack_replays_safely(live, monkeypatch):
    """Ack window: the vector landed, the ack never committed. The replay must
    leave exactly one point carrying the latest revision."""
    from app.api.v1 import memories as memories_api

    async def store_down(_memory) -> bool:
        return False

    monkeypatch.setattr(memories_api, "index_new_memory", store_down)
    created = await _create_memory(live.alice.id, "lost ack")

    first = await outbox.drain_pending()
    generation = generation_name("memory")
    ids_after_first = _scroll_ids(generation)
    assert first["applied"] == 1 and str(created.id) in ids_after_first
    assert int(_client().count(generation).count) == len(ids_after_first)

    # The ack was lost with the process: the intent is pending again.
    async with live.sessions() as db:
        row = (await db.execute(select(IndexOutbox).where(
            IndexOutbox.entity_id == created.id.hex))).scalars().one()
        row.status, row.attempts, row.next_attempt_at = "pending", 0, None
        await db.commit()

    second = await outbox.drain_pending()
    assert second["applied"] == 1  # re-applied, not double-written
    assert _scroll_ids(generation) == ids_after_first  # the same points, no doubles
    assert int(_client().count(generation).count) == len(ids_after_first)
    assert [row.status for row in await _intents(live)
            if row.entity_id == created.id.hex] == ["done"]


# ── 4. crash window: the flip committed, the marker never landed ────────────


async def test_crash_between_the_flip_and_the_marker_still_knows_the_origin(live, rollback_tool):
    """The cutover is the commit point; the marker is written after it. With the
    marker gone, the store's own pointer + the expand record still answer."""
    marker = Path(f"{live.db_path}.p1b-rollback-marker.json")
    assert marker.is_file(), "cutover writes the marker"

    # The store's own truth survives the crash window...
    active = await _manifest(live)
    assert active["memory"].generation == generation_name("memory")
    assert outbox.active_generation_sync()[0] == generation_name("memory")

    marker.unlink()  # the crash: the flip committed, the marker did not

    origin = rollback_tool.rollback_from(live.db_path)
    assert origin["source"] == "expand-record", origin
    assert origin["active"]["memory"] == vector_store.COLLECTION_NAME

    # And the rollback's own precondition still holds without the marker, so a
    # real rollback can run (the full rebuild is test 13's job).
    evidence = rollback_tool.mean_evidence(live.db_path)
    assert evidence["mean"] is True, evidence
    assert rollback_tool.mean_token() not in {
        row.fingerprint for row in (await _manifest(live)).values()}


# ── 5. count / ID / revision coverage after cutover ─────────────────────────


async def test_count_ids_and_revisions_are_covered_after_cutover(live):
    """The audit says green AND the runtime's own readback agrees with SQL."""
    report = live.cli.verify(kind="memory")
    assert report["ok"] is True, report["findings"]
    assert (report["sql_eligible"], report["points"]) == (3, 3)
    assert live.cli.verify(kind="chunk")["ok"] is True

    async with live.sessions() as db:
        expected = {
            str(m.id): (int(m.revision), str(m.user_id))
            for m in (await db.execute(select(Memory))).scalars().all()
        }
    payloads = _payloads(generation_name("memory"))
    eligible = {str(live.alice_current.id), str(live.bob_current.id), str(live.bob_extra.id)}
    assert set(payloads) == eligible  # the superseded/dirty/suppressed rows carry no point
    for point_id, payload in payloads.items():
        revision, tenant = expected[point_id]
        assert payload["orivory_memory_revision"] == revision
        assert payload["user_id"] == tenant
        assert payload["memory_id"] == point_id

    # The runtime's presence readback (the erasure verification seam) agrees.
    assert await vector_store.get_memory_ids_present(sorted(eligible)) == eligible
    assert await vector_store.get_memory_ids_present([str(live.alice_superseded.id)]) == set()


# ── 6. tenant injections: search, delete, presence ──────────────────────────


async def test_tenant_injection_is_refused_and_cannot_reach_another_owner(live):
    """One collection, two owners: the tenant clause is the boundary, and a
    caller's filter — on the memory OR the chunk path — cannot widen it."""
    from app.retrieval.qdrant_filter import build_filter

    # (a) search: a where-clause that names the tenant is REFUSED outright.
    with pytest.raises(ValueError, match="authenticated principal"):
        build_filter(str(live.alice.id), {"user_id": str(live.bob.id)},
                     namespace=namespaces.PERSONAL)
    with pytest.raises(ValueError, match="authenticated principal"):
        await vector_store.search_memories(
            _vector_for("bob current"), user_id=str(live.alice.id), top_k=5,
            where={"user_id": str(live.bob.id)},
        )

    # (b) search: the query that matches Bob's point best still returns only
    # Alice's points when Alice asks — and only Bob's when Bob asks.
    query = _vector_for("bob current")
    as_alice = await vector_store.search_memories(query, user_id=str(live.alice.id), top_k=10)
    assert str(live.bob_current.id) not in {hit["memory_id"] for hit in as_alice}
    assert {hit["memory_id"] for hit in as_alice} == {
        str(live.alice_current.id)}  # the only eligible row of hers in a top_k=10
    as_bob = await vector_store.search_memories(query, user_id=str(live.bob.id), top_k=10)
    assert as_bob[0]["memory_id"] == str(live.bob_current.id)
    assert {hit["memory_id"] for hit in as_bob} == {
        str(live.bob_current.id), str(live.bob_extra.id)}

    # (c) chunk scope: a foreign tenant cannot read (presence) or delete the
    # owner's conversation, and the points are still there afterwards.
    generation = generation_name("chunk")
    owner_chunks = {str(chunk.id) for chunk in live.alice_chunks}
    assert owner_chunks <= _scroll_ids(generation)
    assert await vector_retriever.search(
        "gate chunk 0", 5, str(live.alice_conv.id), user_id=str(live.bob.id)) == []
    assert await vector_retriever.delete_conversation_chunks(
        str(live.alice_conv.id), user_id=str(live.bob.id)) is True  # deleted nothing
    assert owner_chunks <= _scroll_ids(generation)  # ...and nothing is gone


# ── 7. correction: old + dirty rows leave context AND rerank ────────────────


async def _recall(env, owner, query: str, *, rerank_input=None):
    """The REAL recall pipeline over the real store: only the reranker seam is
    stubbed (it records what it was handed)."""
    from app.retrieval import reranker as reranker_module
    from app.retrieval.memory import retriever as retriever_module
    from app.retrieval.memory.retriever import MemoryRetriever

    async def _rewrite(query, context=None):
        return {"rewritten_query": query, "entities": [], "reasoning": None,
                "_fallback_used": False}

    async def _rerank(_query, chunks, *, top_n=None):
        if rerank_input is not None:
            rerank_input.extend(chunk["memory_id"] for chunk in chunks)
        return chunks

    async with env.sessions() as db:
        retriever = MemoryRetriever(db, owner, semantic_rerank=True)
        rewrite_before = retriever_module.rewrite_query
        retriever_module.rewrite_query = _rewrite
        reranker_module.rerank, rerank_before = _rerank, reranker_module.rerank
        try:
            return await retriever.recall(query)
        finally:
            retriever_module.rewrite_query = rewrite_before
            reranker_module.rerank = rerank_before


async def test_correction_old_and_dirty_rows_leave_context_and_rerank(live):
    """A correction after cutover: the superseded row and its dirty derived view
    stay in the store, and are still dropped before the reranker sees them."""
    from app.retrieval.memory.context import fetch_personal_context
    from app.retrieval.memory.correction import (
        CM_DERIVED_FROM,
        Slot,
        resolve_correction,
        state_of,
    )

    async with live.sessions() as db:
        first = await resolve_correction(db, user_id=live.alice.id, title="DB",
                                         content="Postgres", slot=Slot.of("proj", "db", "prod"))
        old = first["memory"]
        derived = _memory(live.alice.id, "derived view", title="Summary",
                          extra_metadata={CM_DERIVED_FROM: [str(old.id)]})
        db.add(derived)
        await db.commit()
        await vector_store.upsert_memory(old)
        await vector_store.upsert_memory(derived)
        old_id, derived_id = old.id, derived.id

    async with live.sessions() as db:
        second = await resolve_correction(db, user_id=live.alice.id, title="DB",
                                          content="SQLite", slot=Slot.of("proj", "db", "prod"))
        new = second["memory"]
        assert second["status"] == "superseded"
        assert second["superseded"] == [str(old_id)]
        assert second["dirtied"] == [str(derived_id)]
        new_id = new.id

    # The correction enqueues its own upsert; the drain is what lands it.
    await outbox.drain_pending()
    async with live.sessions() as db:
        rows = [await db.get(Memory, memory_id) for memory_id in (old_id, derived_id)]
    assert tuple(state_of(row) for row in rows) == ("superseded", "dirty")

    # The store still holds the stale vectors — the read path is the filter.
    payloads = _payloads(generation_name("memory"))
    stale = {str(old_id), str(derived_id), str(new_id)} - set(payloads)
    assert not stale, f"vectors missing from the store: {sorted(stale)}"
    assert payloads[str(old_id)]["visibility_state"] == "superseded"
    # A dirtying is a SQL-side mark: the payload still says what the row was when
    # its vector was written — which is exactly why the read path filters on SQL
    # (below) instead of trusting `visibility_state`.
    assert payloads[str(derived_id)]["visibility_state"] == "current"

    # The personal context serves the corrected fact only ...
    async with live.sessions() as db:
        context_ids = [row.id for row in await fetch_personal_context(db, live.alice.id)]
    assert new_id in context_ids
    assert old_id not in context_ids and derived_id not in context_ids

    # ... and the reranker is never handed the stale candidates.
    reranker_input: list[str] = []
    response = await _recall(live, live.alice.id, "Postgres", rerank_input=reranker_input)
    returned = {str(result.id) for result in response.results}
    assert str(old_id) not in returned and str(derived_id) not in returned
    assert str(old_id) not in reranker_input and str(derived_id) not in reranker_input


# ── 8. erasure: outage → retry → verified, over the real store ──────────────


async def test_erasure_outage_then_drain_finishes_the_purge(live, monkeypatch):
    """While the store is down the receipt must NOT claim `completed`; the
    recovery half then runs for real and the vector is gone."""
    from app.models.erasure_receipt import ErasureReceipt
    from app.services import erasure_service
    from app.services.erasure_service import erase_memories

    async with live.sessions() as db:
        target = _memory(live.alice.id, "forget me")
        db.add(target)
        await db.commit()
        await vector_store.upsert_memory(target)
    assert str(target.id) in _scroll_ids(generation_name("memory"))

    real_purge, real_present = (erasure_service.safe_delete_from_index,
                               erasure_service._vector_present_ids)

    async def purge_down(_memory_id) -> bool:
        return False  # the purge did not land

    async def present_check_down(_memory_ids):
        raise ConnectionError("qdrant down")

    monkeypatch.setattr(erasure_service, "safe_delete_from_index", purge_down)
    monkeypatch.setattr(erasure_service, "_vector_present_ids", present_check_down)

    async with live.sessions() as db:
        receipt = await erase_memories(db, live.alice.id, [target.id], requested_by="rest_api")

    entry = receipt.detail["targets"][0]
    assert receipt.status == "completed_unverified"
    assert (entry["vector_state"], entry["vector_residual_checked"]) == ("pending", False)
    assert receipt.detail["index_pending"] == 1
    assert str(target.id) in _scroll_ids(generation_name("memory"))  # still there

    # Recovery: the store is back; the owed delete intent is what finishes it.
    monkeypatch.setattr(erasure_service, "safe_delete_from_index", real_purge)
    monkeypatch.setattr(erasure_service, "_vector_present_ids", real_present)
    report = await outbox.drain_pending()
    assert report["applied"] >= 1
    assert str(target.id) not in _scroll_ids(generation_name("memory"))
    assert "delete" in {row.operation for row in await _intents(live)}
    assert {row.status for row in await _intents(live) if row.entity_id == target.id.hex} == {
        "done"}

    # The same receipt is updatable once the owed work landed (re-verification).
    async with live.sessions() as db:
        refreshed = await db.get(ErasureReceipt, receipt.id)
        assert refreshed.status == "completed_unverified"
        refreshed.status = "completed"
        refreshed.detail = {**refreshed.detail, "verification": "verified", "index_pending": 0}
        await db.commit()
    async with live.sessions() as db:
        again = await db.get(ErasureReceipt, receipt.id)
    assert (again.status, again.detail["verification"], again.detail["index_pending"]) == (
        "completed", "verified", 0)


async def test_erasure_closure_depth_cycle_and_derived_vectors_all_go(live):
    """A deep closure with a cycle plus a derived dependent: every vector of the
    affected set is gone from the REAL store, and every intent is acked."""
    from app.retrieval.memory.correction import CM_DERIVED_FROM
    from app.services.erasure_service import erase_memories

    async with live.sessions() as db:
        root = _memory(live.alice.id, "closure root")
        chain = [_memory(live.alice.id, f"closure c{i}") for i in range(4)]
        parent = root
        for node in chain:
            node.parent_id = parent.id
            parent = node
        derived = _memory(live.alice.id, "closure derived view",
                          extra_metadata={CM_DERIVED_FROM: [str(root.id)]})
        db.add_all([root, *chain, derived])
        await db.flush()
        root.parent_id = chain[-1].id  # c3 -> root back-edge: the walk must terminate
        await db.commit()
        for row in (root, *chain, derived):
            await vector_store.upsert_memory(row)

    generation = generation_name("memory")
    affected = {root.id, *(node.id for node in chain), derived.id}
    assert {str(i) for i in affected} <= _scroll_ids(generation)

    async with live.sessions() as db:
        receipt = await erase_memories(db, live.alice.id, [root.id], requested_by="rest_api")

    entry = receipt.detail["targets"][0]
    assert entry["status"] == "deleted"
    assert {uuid.UUID(i) for i in entry["affected_memory_ids"]} == affected - {
        root.id, derived.id}  # the derived dependent is reported in its own list
    assert uuid.UUID(entry["derived_memory_ids"][0]) == derived.id
    assert entry["traversal_depth"] == len(chain)
    assert entry["vector_state"] == "verified" and entry["index_pending"] == len(affected)
    assert affected.isdisjoint({uuid.UUID(i) for i in _scroll_ids(generation)})

    await outbox.drain_pending()
    assert {row.status for row in await _intents(live) if row.kind == "memory"} == {"done"}


# ── 9. document reingest + delete leave no orphan vector ────────────────────


async def test_document_reingest_and_delete_leave_no_orphan_vectors(live, monkeypatch):
    """The real ingestion pipeline against the real stores: a reingest mints new
    ids and the old points go; the delete cascade takes the rest."""
    from app.ingestion import pipeline
    from app.models.document_chunk import DocumentChunk
    from app.services import document_service

    # Out-of-process seams only: object storage, the Redis parent cache, BM25.
    monkeypatch.setattr("app.storage.get_object_sync", lambda *a, **k: b"gate bytes")
    monkeypatch.setattr("app.utils.chunker.extract_text", lambda *a, **k: "Gate body. " * 140)
    monkeypatch.setattr("app.retrieval.parent_store.store_parents_sync", lambda *a, **k: None)
    monkeypatch.setattr("app.retrieval.bm25_retriever.bm25_retriever.publish_build_sync",
                        lambda *a, **k: None)
    monkeypatch.setattr("app.retrieval.retrieval_cache.invalidate_query_cache_sync",
                        lambda *a, **k: None)

    async def _remove_object(*_a, **_k):
        return None

    monkeypatch.setattr("app.storage.remove_object", _remove_object)
    monkeypatch.setattr("app.retrieval.retrieval_cache.invalidate_query_cache", _remove_object)
    # The BM25 rebuild reads the SQLite chunk metadata with a Postgres-shaped
    # expression: the BM25/Redis seam is stubbed like the rest of them here.
    monkeypatch.setattr("app.retrieval.bm25_retriever.bm25_retriever.publish_rebuild_async",
                        _remove_object)

    generation = generation_name("chunk")

    def _document_children() -> set[str]:
        """The document's CHILD rows — the only ones the chunk index carries."""
        with sync_session() as db:
            rows = db.execute(
                select(DocumentChunk).where(DocumentChunk.document_id == live.alice_doc.id)
            ).scalars().all()
            return {str(row.id) for row in rows
                    if (row.chunk_metadata or {}).get("chunk_type") == "child"}

    # First ingest: the rows and their points.
    with sync_session() as db:
        pipeline._ingest(db, str(live.alice_doc.id))
    first_ids = _document_children()
    assert first_ids and first_ids <= _scroll_ids(generation)

    # Reingest: the chunker mints new ids; the old points must not survive.
    with sync_session() as db:
        pipeline._ingest(db, str(live.alice_doc.id))
    second_ids = _document_children()
    assert second_ids and not (first_ids & second_ids)
    await outbox.drain_pending()  # the durable delete intents own the old points
    present = _scroll_ids(generation)
    assert second_ids <= present
    assert first_ids.isdisjoint(present), "a reingest left an orphan point"

    # Delete the document through the real service: rows, intents and vectors.
    async with live.sessions() as db:
        document = await db.get(Document, live.alice_doc.id)
        conversation = await db.get(Conversation, live.alice_conv.id)
        await document_service.delete_document(db, document, conversation)
    await outbox.drain_pending()
    present = _scroll_ids(generation)
    assert (first_ids | second_ids).isdisjoint(present)
    assert {str(chunk.id) for chunk in live.bob_chunks} <= present, "Bob's chunks stay"
    orphans = {
        point_id for point_id, payload in _payloads(generation).items()
        if payload.get("document_id") == str(live.alice_doc.id)
    }
    assert orphans == set(), f"the delete left orphan points: {sorted(orphans)}"

    # The conversation cascade scopes by tenant as well.
    assert await vector_retriever.delete_conversation_chunks(
        str(live.alice_conv.id), user_id=str(live.alice.id)) is True
    assert {str(chunk.id) for chunk in live.bob_chunks} <= _scroll_ids(generation)


# ── 10. a chunk delete that is not confirmed is never acked ─────────────────


async def test_unconfirmed_chunk_delete_leaves_the_intent_pending(live, monkeypatch):
    """The durable delete intent is the only record that the point still has to
    go: while the store does not confirm it, the intent must stay pending (the
    drain's backoff owns it) — acking it would strand the vector forever, with
    nothing left in SQL to find it by."""
    generation = generation_name("chunk")
    chunk_id = live.alice_chunks[0].id
    assert str(chunk_id) in _scroll_ids(generation)

    # The delete path's own enqueue, in its own transaction.
    async with live.sessions() as db:
        assert await outbox.enqueue_chunk_delete(
            db, chunk_ids=[chunk_id], tenant_id=live.alice.id) == []
        await db.commit()

    real_delete = outbox.delete_chunks

    async def unconfirmed(_chunk_ids) -> bool:
        return False  # the store took the call and never confirmed the delete

    monkeypatch.setattr(outbox, "delete_chunks", unconfirmed)
    report = await outbox.drain_pending()
    assert (report["claimed"], report["applied"], report["failed"]) == (1, 0, 1)

    row = next(row for row in await _intents(live) if row.entity_id == chunk_id.hex)
    assert (row.operation, row.status) == ("delete", "pending")  # NOT acked
    assert row.attempts == 1 and row.next_attempt_at is not None
    assert row.last_error and row.last_error.startswith("VectorDeleteUnconfirmed"), \
        row.last_error
    assert str(chunk_id) in _scroll_ids(generation)  # the point is still there

    # The store confirms on the retry: the SAME intent is what finishes it.
    monkeypatch.setattr(outbox, "delete_chunks", real_delete)
    async with live.sessions() as db:  # the backoff elapsed: same intent, due again
        due = (await db.execute(select(IndexOutbox).where(
            IndexOutbox.entity_id == chunk_id.hex))).scalars().one()
        due.next_attempt_at = None
        await db.commit()
    report = await outbox.drain_pending()
    assert report["applied"] == 1
    assert str(chunk_id) not in _scroll_ids(generation)
    assert [row.status for row in await _intents(live)
            if row.entity_id == chunk_id.hex] == ["done"]


# ── 11. backup: the WAL is in the snapshot, the drill restores it ───────────


async def test_backup_takes_the_wal_and_the_drill_restores_into_a_new_dir(live, tmp_path):
    """A write that only ever lived in the WAL must be IN the backup, and the
    restore drill must report ready from a fresh target directory."""
    # NOTHING checkpoints here — that is the pin. The backup must be the thing
    # that takes the WAL (product side: checkpoint then VACUUM INTO), and for the
    # WAL to still be holding the write something has to keep the file open:
    # SQLite checkpoints and deletes the -wal when the LAST connection closes, so
    # a second connection (a live app, a reader, a -wal left by a kill) is
    # exactly the state this claim is about.
    holder = sqlite3.connect(live.db_path)
    # A read first: an idle connection only takes its WAL handle on first access,
    # and without one SQLite still checkpoints-and-deletes the -wal when the
    # writer's own connection closes.
    holder.execute("SELECT count(*) FROM memories").fetchall()
    try:
        late = await _create_memory(live.alice.id, "written after the last checkpoint")

        # Prove the write is WAL-resident (a vacuous pin is worse than none)…
        wal = Path(f"{live.db_path}-wal")
        assert wal.exists() and wal.stat().st_size > 0, "the WAL is empty: pin is vacuous"
        # …and that the main database file ALONE does not carry it: a backup that
        # copied just that file would come back without the row asserted below.
        main_only = tmp_path / "main-only.db"
        shutil.copy(live.db_path, main_only)
        connection = sqlite3.connect(str(main_only))
        try:
            assert connection.execute(
                "SELECT count(*) FROM memories WHERE id = ?", (late.id.hex,)
            ).fetchone()[0] == 0, "the late write already reached the main file"
        finally:
            connection.close()

        dest = tmp_path / "backups"
        dest.mkdir()
        report = live.cli.backup(dest_dir=dest)
    finally:
        holder.close()
    snapshot = dest / f"{DB_NAME}.pre-p1b.bak"
    assert report["db_backup"] == str(snapshot) and snapshot.exists()
    assert Path(f"{snapshot}.manifest.json").exists()

    target = tmp_path / "restored"
    drill = live.cli.restore_drill(backup_dir=dest, target=target)
    assert drill["ok"] is True, drill["checks"]
    restored = target / f"{DB_NAME}.pre-p1b.bak"
    assert restored.exists()
    connection = sqlite3.connect(f"file:{restored}?mode=ro", uri=True)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == database.SQLITE_SCHEMA_VERSION
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute(
            "SELECT count(*) FROM memories WHERE id = ?", (late.id.hex,)
        ).fetchone()[0] == 1, "the WAL-resident write is in the snapshot"
        assert connection.execute(
            "SELECT count(*) FROM memory_suppressions WHERE source_ref = ?", (FORGOTTEN_SOURCE,)
        ).fetchone()[0] == 1  # the ledger travels with the backup
    finally:
        connection.close()
    assert drill["checks"]["ledger"]["ok"] is True
    assert drill["checks"]["absence"]["ok"] is True


# ── 12. rollback reconciles the writes made AFTER the cutover ───────────────


async def test_rollback_reconciles_writes_made_after_cutover(
    live, rollback_tool, tmp_path, monkeypatch
):
    """Add / edit / correct / forget / reingest — all after the flip — and then
    the pre-P1b store the old binary reads must carry exactly the live set."""
    import chromadb

    from app.api.v1 import memories as memories_api
    from app.ingestion import pipeline
    from app.retrieval.memory.correction import Slot, resolve_correction
    from app.schemas.Orivory import MemoryUpdate
    from app.services.erasure_service import erase_memories

    added = await _create_memory(live.alice.id, "written after cutover", title="Post flip")
    async with database.AsyncSessionLocal() as db:
        await memories_api.update_memory(
            added.id, MemoryUpdate(title="Edited after cutover"), SimpleNamespace(id=live.alice.id), db
        )
    async with database.AsyncSessionLocal() as db:
        kept = await resolve_correction(db, user_id=live.alice.id, title="DB", content="Postgres",
                                        slot=Slot.of("proj", "db", "prod"))
        corrected = await resolve_correction(db, user_id=live.alice.id, title="DB", content="SQLite",
                                             slot=Slot.of("proj", "db", "prod"))
    assert corrected["status"] == "superseded"

    async with database.AsyncSessionLocal() as db:
        forgotten = _memory(live.alice.id, "forgotten after cutover")
        db.add(forgotten)
        await db.commit()
        await vector_store.upsert_memory(forgotten)
        await erase_memories(db, live.alice.id, [forgotten.id], requested_by="rest_api")

    monkeypatch.setattr("app.storage.get_object_sync", lambda *a, **k: b"gate bytes")
    monkeypatch.setattr("app.utils.chunker.extract_text", lambda *a, **k: "Reingested body. " * 140)
    monkeypatch.setattr("app.retrieval.parent_store.store_parents_sync", lambda *a, **k: None)
    monkeypatch.setattr("app.retrieval.bm25_retriever.bm25_retriever.publish_build_sync",
                        lambda *a, **k: None)
    monkeypatch.setattr("app.retrieval.retrieval_cache.invalidate_query_cache_sync",
                        lambda *a, **k: None)
    with sync_session() as db:
        pipeline._ingest(db, str(live.alice_doc.id))
    with sync_session() as db:
        rows = db.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == live.alice_doc.id)
        ).scalars().all()
    current_rows = {str(row.id) for row in rows}
    reingested = {str(row.id) for row in rows
                  if (row.chunk_metadata or {}).get("chunk_type") == "child"}

    chroma_path = tmp_path / "chroma-rollback"
    report = rollback_tool.rollback(db_path=live.db_path, chroma_path=chroma_path,
                                    embed_passages=_fake_embed_sync)
    assert report["ok"] is True and report["ready"] is True, report.get("gates")

    client = chromadb.PersistentClient(path=str(chroma_path))
    memory_ids = set(client.get_collection(vector_store.COLLECTION_NAME).get(include=[])["ids"])
    # The writes after the cutover are IN, the erase/correction losers are OUT.
    assert str(added.id) in memory_ids
    assert str(corrected["memory"].id) in memory_ids
    assert str(kept["memory"].id) not in memory_ids  # superseded: history, not recall
    assert str(forgotten.id) not in memory_ids
    document = client.get_collection(vector_store.COLLECTION_NAME).get(
        ids=[str(added.id)], include=["documents"])["documents"][0]
    assert "Edited after cutover" in document  # the edit, read back

    chunks = set(client.get_collection(f"rag_conv_{live.alice_conv.id}").get(include=[])["ids"])
    # The pre-P1b per-conversation collection indexed parents AND children (the
    # old pipeline's whole row set), so the rebuild carries exactly the current
    # rows — no id from the superseded generation survives.
    assert reingested and reingested <= chunks, "the rebuild carries the post-cutover reingest"
    assert chunks == current_rows, "stale chunk ids survive the rebuild"
    assert {str(chunk.id) for chunk in live.bob_chunks}.isdisjoint(chunks)


# ── 13. one owner per folder; no multiworker fallback ───────────────────────


async def test_a_second_local_owner_is_blocked(live):
    """The embedded folder has exactly ONE owner: a second client dies on the
    lock (a real qdrant-client property, not a mocked one)."""
    with pytest.raises(RuntimeError, match="already accessed"):
        QdrantClient(path=str(live.qdrant_dir))


async def test_no_multiworker_fallback_for_the_local_owner(env, monkeypatch):
    """A multi-process launcher is REFUSED — never silently served by a second
    client on the same folder, and never rescued by a server-mode fallback."""
    from app import main

    assert vector_backend.is_local_mode() is True
    main._refuse_multi_owner_local_qdrant()  # one process: fine

    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    with pytest.raises(RuntimeError, match="single process"):
        main._refuse_multi_owner_local_qdrant()

    monkeypatch.delenv("WEB_CONCURRENCY")
    monkeypatch.setenv("UVICORN_WORKERS", "2")
    with pytest.raises(RuntimeError, match="2 processes"):
        main._refuse_multi_owner_local_qdrant()

    monkeypatch.delenv("UVICORN_WORKERS")
    monkeypatch.setattr(sys, "argv", ["uvicorn", "app.main:app", "--workers=3"])
    with pytest.raises(RuntimeError, match="3 processes"):
        main._refuse_multi_owner_local_qdrant()

    # The refusal is not a mode switch: with a Qdrant URL in reach it still owns
    # the folder (local mode is the decision, not the reachable server).
    monkeypatch.setattr(settings, "QDRANT_URL", "http://qdrant.internal:6333")
    with pytest.raises(RuntimeError):
        main._refuse_multi_owner_local_qdrant()


# ── 14. the image needs no Chroma ───────────────────────────────────────────

CHILD_SCRIPT = '''
import asyncio
import os
import sys
import uuid

os.environ["USE_LOCAL_EMBEDDINGS"] = "true"
os.environ["LOCAL_EMBED_MODEL"] = "arctic"
os.environ["DATABASE_URL"] = os.environ["GATE_CHILD_DB"]
os.environ["QDRANT_MODE"] = "local"
os.environ["QDRANT_LOCAL_PATH"] = os.environ["GATE_CHILD_QDRANT"]


class _NoChroma:
    """`import chromadb` fails exactly like an image that never had it."""

    def find_spec(self, name, path=None, target=None):
        if name == "chromadb" or name.startswith("chromadb."):
            raise ImportError(f"{name} is not installed in this image")
        return None


sys.meta_path.insert(0, _NoChroma())

import app.main  # noqa: E402,F401 — the app's whole import surface
from app.services import health_service  # noqa: E402,F401 — the readiness surface
from app.retrieval import embedder  # noqa: E402,F401 — the embedding surface
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import NullPool  # noqa: E402

from app import database  # noqa: E402
from app.models.memory import Memory  # noqa: E402
from app.models.user import User  # noqa: E402
from app.retrieval import vector_backend  # noqa: E402
from app.retrieval.memory import outbox, vector_store  # noqa: E402

DIM = 384


def _blocker_is_real() -> None:
    """The premise: `import chromadb` really raises in this interpreter."""
    try:
        import chromadb  # noqa: F401
    except ImportError:
        return
    raise AssertionError("the import blocker did not fire — the premise is void")


def _vector(text: str) -> list[float]:
    import hashlib
    import math
    import random

    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    raw = [rng.uniform(-1.0, 1.0) for _ in range(DIM)]
    norm = math.sqrt(sum(value * value for value in raw))
    return [value / norm for value in raw]


async def _fake_embed(texts):
    return [_vector(text) for text in texts]


def _fake_embed_sync(texts):
    """The sync twin: the async fake handed to ``embed_texts_sync`` would be a
    never-awaited coroutine — truthy, and silently unembedded."""
    return [_vector(text) for text in texts]


async def main() -> None:
    _blocker_is_real()
    assert "chromadb" not in sys.modules
    from app.retrieval.embedding_fingerprint import ARCTIC_CLS_FINGERPRINT
    assert embedder.current_fingerprint() == ARCTIC_CLS_FINGERPRINT

    engine = create_async_engine(os.environ["DATABASE_URL"],
                                 connect_args={"check_same_thread": False}, poolclass=NullPool)
    sessions = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False,
                                  autoflush=False)
    database.engine = engine
    database.AsyncSessionLocal = sessions
    outbox.AsyncSessionLocal = sessions
    vector_store.embed_texts = _fake_embed
    vector_store.embed_texts_sync = _fake_embed_sync

    await database.bootstrap_sqlite()  # fresh image: schema + the active rows

    async with sessions() as db:
        user = User(id=uuid.uuid4(), email="child@gate.invalid", hashed_password="x",
                    display_name="Child", is_verified=True, is_active=True)
        db.add(user)
        await db.commit()
        row = Memory(id=uuid.uuid4(), user_id=user.id, content="child ingest", tags=[])
        db.add(row)
        await db.commit()

        await vector_store.upsert_memory(row)  # real write, real collection
        hits = await vector_store.search_memories(
            _vector(vector_store._memory_to_document(row)), user_id=str(user.id), top_k=5)
        assert [hit["memory_id"] for hit in hits] == [str(row.id)], hits

    await vector_backend.close_clients()
    await engine.dispose()
    print("NO-CHROMA-OK")


asyncio.run(main())
'''


def test_the_image_imports_ingests_and_recalls_without_chroma(tmp_path):
    """A CHILD interpreter whose ``import chromadb`` raises: the app imports,
    the image boots, and the ingest+recall path works end to end."""
    script = tmp_path / "child_no_chroma.py"
    script.write_text(textwrap.dedent(CHILD_SCRIPT))
    qdrant = tmp_path / "child-qdrant"
    qdrant.mkdir()
    environment = {
        **os.environ,
        "GATE_CHILD_DB": f"sqlite+aiosqlite:///{tmp_path / 'child.db'}",
        "GATE_CHILD_QDRANT": str(qdrant),
        "DATABASE_URL": f"sqlite+aiosqlite:///{tmp_path / 'child.db'}",
        "PYTHONPATH": str(REPO_ROOT),
    }
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, env=environment,
        cwd=str(REPO_ROOT), timeout=180,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "NO-CHROMA-OK" in result.stdout
