"""Task 5 — the offline P1b migration CLI against a real store (no mocks).

Real SQLite file, real embedded Qdrant folder, real keyset pagination, real
scroll-based verify/GC. Only the embedder is stubbed (deterministic unit
vectors): the durability, pagination, checkpoint and audit claims are the
point of the suite and they must run against the actual stores.

Contract under test (brief + rulings R24-R29):

- ``inventory --out``: read-only counts per tenant/kind + quarantine lists.
- ``backup --dir``: VACUUM INTO snapshot + checksum manifest, resumable.
- ``backfill --kind --batch [--resume]``: keyset by primary key (never
  OFFSET), checkpoint per acked batch, GC of points whose SQL row is not
  eligible, no re-embed of acked rows after a crash.
- the expand (the ladder v3 step and ``backfill``'s ``_expand``) only WRITES the
  two generation rows; the pointer stays where it was until ``cutover`` flips
  it. A lite install's old pointer names the masked-mean contract, so the read
  path's guard raises (loud, never an empty generation). Where there is NO
  active row (Postgres, where P1a never seeded ``index_generations``) the read
  path falls back to the transitional generation name and answers empty
  results until the flip — the pointer move is still the only thing that
  starts serving the built generation (F1).
- ``verify --kind``: full read-side audit — count, ID set, revision,
  fingerprint, tenant, absence of excluded rows; empty is valid only with the
  manifest row present.
- ``cutover --yes``: refuses without ``--yes`` and unless BOTH kinds verify
  green; one transaction flips the pointer for both kinds and blocks stale
  generation intents; rollback marker written.
- quiesce: refuses while ``migrate.lock`` is held by a live pid, or while the
  app's listen port answers.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import socket
import sqlite3
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
from qdrant_client import models as qm
from sqlalchemy import create_engine, event, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from app import database
from app import models as _models  # noqa: F401 — register every table on Base
from app.config import settings
from app.database import Base
from app.models.conversation import Conversation
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.index_outbox import IndexGeneration, IndexOutbox
from app.models.memory import Memory, MemorySuppression
from app.models.user import User
from app.retrieval import vector_backend
from app.retrieval.embedding_fingerprint import (
    canonical_fingerprint,
    fingerprint_generation,
    generation_name,
)
from app.retrieval.memory import outbox, vector_store

DB_NAME = "migrate.db"
DIM = 8
# The embedding contract this suite pins (never the ambient one: the same
# tests run on a 384-dim lite install and a 1536-dim OpenAI one).
FINGERPRINT = {
    "model_id": "test-model",
    "model_revision": "revision-1",
    "dim": DIM,
    "provider": "test",
}


def _fingerprint() -> dict:
    return dict(FINGERPRINT)


def _token() -> str:
    return fingerprint_generation(canonical_fingerprint(FINGERPRINT))


def _vector_for(text: str) -> list[float]:
    """Unit vector for a text: same text -> same vector, every run."""
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    raw = [rng.uniform(-1.0, 1.0) for _ in range(DIM)]
    norm = math.sqrt(sum(value * value for value in raw))
    return [value / norm for value in raw]


def _fake_embed_sync(texts: list[str]) -> list[list[float]]:
    return [_vector_for(text) for text in texts]


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def env(tmp_path, monkeypatch, migrate_cli):
    """Private SQLite + private embedded Qdrant, pointed at by settings."""
    folder = tmp_path / "qdrant"
    folder.mkdir()
    monkeypatch.setattr(settings, "QDRANT_MODE", "local")
    monkeypatch.setattr(settings, "QDRANT_LOCAL_PATH", str(folder))
    # Nothing listens on port 1, so the quiesce probe passes unless a test says
    # otherwise (a dev machine may well have something on the default 8000).
    monkeypatch.setattr(settings, "APP_PORT", 1)
    monkeypatch.setattr(settings, "FS_STORAGE_PATH", str(tmp_path / "uploads"))
    monkeypatch.setattr(settings, "LEGACY_CHROMA_PATH", str(tmp_path / "no-chroma"))

    db_path = tmp_path / DB_NAME
    url = f"sqlite+aiosqlite:///{db_path}"
    monkeypatch.setattr(settings, "DATABASE_URL", url)
    engine = create_async_engine(url, connect_args={"check_same_thread": False}, poolclass=NullPool)
    event.listen(engine.sync_engine, "connect", database._configure_sqlite_connection)
    sync_engine = create_engine(
        url.replace("+aiosqlite", ""), connect_args={"check_same_thread": False}
    )
    event.listen(sync_engine, "connect", database._configure_sqlite_connection)
    sessions = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False, autoflush=False)
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    monkeypatch.setattr(database, "AsyncSessionLocal", sessions)
    monkeypatch.setattr(outbox, "AsyncSessionLocal", sessions)
    monkeypatch.setattr(
        database, "_get_sync_sessionmaker",
        lambda: sessionmaker(bind=sync_engine, expire_on_commit=False, autoflush=False),
    )

    # Every guard must see the same contract: the fingerprint and the embedder
    # are bound at import in several modules, so patch each reader.
    from app.retrieval import embedder as embedder_module
    from app.retrieval import embedding_fingerprint as fingerprint_module
    from app.retrieval import vector_retriever

    for module in (fingerprint_module, embedder_module, vector_store, vector_retriever, migrate_cli):
        monkeypatch.setattr(module, "current_fingerprint", _fingerprint)
    monkeypatch.setattr(migrate_cli, "embed_texts_sync", _fake_embed_sync)
    monkeypatch.setattr(vector_store, "embed_texts_sync", _fake_embed_sync)
    monkeypatch.setattr(vector_retriever, "embed_texts_sync", _fake_embed_sync)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield SimpleNamespace(
            sessions=sessions, db_path=db_path, url=url, tmp_path=tmp_path,
            qdrant_dir=folder, cli=migrate_cli,
        )
    finally:
        await vector_backend.close_clients()
        await engine.dispose()
        sync_engine.dispose()


async def _ladder(env) -> None:
    """The SQLite expand step (ladder v3): two real generation rows, INACTIVE."""
    await database.bootstrap_sqlite()


async def _add_user(sessions, email: str) -> User:
    user = User(id=uuid.uuid4(), email=email, hashed_password="x",
                display_name="Owner", is_verified=True, is_active=True)
    async with sessions() as db:
        db.add(user)
        await db.commit()
    return user


async def _add_memory(sessions, user_id, content: str, *, metadata: dict | None = None) -> Memory:
    memory = Memory(id=uuid.uuid4(), user_id=user_id, content=content, tags=[],
                    extra_metadata=metadata or {})
    async with sessions() as db:
        db.add(memory)
        await db.commit()
    return memory


async def _add_chunk(sessions, document, content: str, index: int) -> DocumentChunk:
    chunk = DocumentChunk(
        id=uuid.uuid4(), document_id=document.id, content=content, chunk_index=index,
        revision=1, chunk_metadata={"conversation_id": str(document.conversation_id)},
    )
    async with sessions() as db:
        db.add(chunk)
        await db.commit()
    return chunk


async def _add_document(sessions, conversation) -> Document:
    document = Document(id=uuid.uuid4(), conversation_id=conversation.id,
                        filename="doc.txt", file_path="/tmp/doc.txt", chunk_count=2)
    async with sessions() as db:
        db.add(document)
        await db.commit()
    return document


async def _add_conversation(sessions, user_id) -> Conversation:
    conversation = Conversation(id=uuid.uuid4(), user_id=user_id, document_count=1)
    async with sessions() as db:
        db.add(conversation)
        await db.commit()
    return conversation


async def _ownership(sessions) -> SimpleNamespace:
    """Two owners, one conversation + document each."""
    alice = await _add_user(sessions, "alice@test.invalid")
    bob = await _add_user(sessions, "bob@test.invalid")
    alice_conv = await _add_conversation(sessions, alice.id)
    bob_conv = await _add_conversation(sessions, bob.id)
    alice_doc = await _add_document(sessions, alice_conv)
    bob_doc = await _add_document(sessions, bob_conv)
    return SimpleNamespace(alice=alice, bob=bob, alice_conv=alice_conv,
                           bob_conv=bob_conv, alice_doc=alice_doc, bob_doc=bob_doc)


def _client():
    return vector_backend.get_sync_client()


def _scroll_ids(generation: str) -> set[str]:
    client = _client()
    points, _ = client.scroll(collection_name=generation, limit=512, offset=None,
                              with_payload=False, with_vectors=False)
    return {str(point.id) for point in points}


def _raw_upsert(generation: str, point_id: str, payload: dict, text: str = "junk") -> None:
    _client().upsert(
        collection_name=generation,
        points=[qm.PointStruct(id=point_id, vector=_vector_for(text), payload=payload)],
    )


async def _manifest_rows(env) -> list[IndexGeneration]:
    async with env.sessions() as db:
        return list((await db.execute(select(IndexGeneration))).scalars().all())


async def _outbox_rows(env) -> list[IndexOutbox]:
    async with env.sessions() as db:
        return list((await db.execute(select(IndexOutbox).order_by(IndexOutbox.seq))).scalars().all())


def _checkpoint_db(sqlite_path: Path) -> None:
    """Flush WAL content into the main file (read-only inventory needs it)."""
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()


def _file_state(path: Path) -> dict:
    state = {}
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(f"{path}{suffix}")
        state[suffix] = (
            hashlib.sha256(candidate.read_bytes()).hexdigest() if candidate.exists() else None
        )
    return state


# ── the brief's fixture, indexed through the real store ─────────────────────


@pytest_asyncio.fixture
async def world(env) -> SimpleNamespace:
    """Alice: current (indexed) + superseded + dirty. Bob: two current (indexed).

    Chunks: alice 2 rows, bob 1 row — all indexed. Plus one orphan point and
    one malformed-id point in the memory collection, and the superseded row's
    stale point (the runtime writes once and never rewrites on correction).
    """
    await _ladder(env)
    own = await _ownership(env.sessions)
    memory_generation = generation_name("memory")
    chunk_generation = generation_name("chunk")

    alice_current = await _add_memory(env.sessions, own.alice.id, "alice current")
    alice_superseded = await _add_memory(
        env.sessions, own.alice.id, "alice old fact",
        metadata={"cm_superseded_by": "later", "cm_subject": "alice", "cm_attribute": "city"},
    )
    alice_dirty = await _add_memory(
        env.sessions, own.alice.id, "alice dirty derived", metadata={"cm_derived_dirty": True}
    )
    bob_current = await _add_memory(env.sessions, own.bob.id, "bob current")
    bob_extra = await _add_memory(env.sessions, own.bob.id, "bob extra")

    alice_chunks = [await _add_chunk(env.sessions, own.alice_doc, f"alice chunk {i}", i)
                    for i in range(2)]
    bob_chunks = [await _add_chunk(env.sessions, own.bob_doc, "bob chunk 0", 0)]

    # Build the target the way the operator does: the CLI's own keyset
    # backfill, which writes into ``generation_name(kind)`` by NAME (F1: the
    # expand never moves the pointer, so the runtime would still index into the
    # transitional generation). Then inject the points the audit must own — the
    # superseded row's (the runtime writes once and never rewrites on
    # correction), an orphan and a malformed id.
    env.cli.backfill(kind="memory", batch=2)
    env.cli.backfill(kind="chunk", batch=2)
    document = vector_store._memory_to_document(alice_superseded)
    _client().upsert(
        collection_name=memory_generation,
        points=[vector_store._point(alice_superseded, _vector_for(document), document)],
    )

    orphan_id = str(uuid.uuid4())
    malformed_id = str(uuid.uuid4())
    # No fingerprint on the orphan: it is also an unknown-contract point.
    _raw_upsert(memory_generation, orphan_id,
                {"kind": "memory", "memory_id": orphan_id, "user_id": str(own.alice.id)})
    _raw_upsert(memory_generation, malformed_id,
                {"kind": "memory", "memory_id": "not-a-uuid", "user_id": str(own.alice.id),
                 "orivory_embed_fingerprint": canonical_fingerprint(FINGERPRINT),
                 "orivory_embed_generation": _token(), "orivory_memory_revision": 1})

    memories = [alice_current, alice_superseded, alice_dirty, bob_current, bob_extra]
    return SimpleNamespace(
        **vars(own), env=env, memory_generation=memory_generation,
        chunk_generation=chunk_generation, alice_current=alice_current,
        alice_superseded=alice_superseded, alice_dirty=alice_dirty,
        bob_current=bob_current, bob_extra=bob_extra, memories=memories,
        alice_chunks=alice_chunks, bob_chunks=bob_chunks,
        orphan_id=orphan_id, malformed_id=malformed_id,
    )


# ── inventory ───────────────────────────────────────────────────────────────


async def test_inventory_reports_counts_per_tenant_and_quarantine_lists(world, tmp_path):
    cli = world.env.cli
    _checkpoint_db(world.env.db_path)
    before_db = _file_state(world.env.db_path)
    before_points = _scroll_ids(world.memory_generation)

    out = tmp_path / "inventory.json"
    report = cli.inventory(out=out)

    memory = report["kinds"]["memory"]
    assert memory["sql"] == {
        "total": 5, "eligible": 3, "superseded": 1, "dirty": 1, "suppressed": 0, "unowned": 0,
    }
    alice = memory["per_tenant"][str(world.alice.id)]
    bob = memory["per_tenant"][str(world.bob.id)]
    assert (alice["eligible"], bob["eligible"]) == (1, 2)
    assert alice["points"] == 4  # current + superseded + orphan + malformed
    assert bob["points"] == 2

    audit = memory["collections"][world.memory_generation]
    assert audit["exists"] is True
    assert audit["points"] == 6
    assert audit["missing"] == []
    assert audit["stale"] == []
    assert audit["orphan"] == [world.orphan_id]
    assert audit["excluded_present"] == [str(world.alice_superseded.id)]
    assert audit["malformed_id"] == [world.malformed_id]
    assert audit["unknown_fingerprint"] == [world.orphan_id]

    chunk = report["kinds"]["chunk"]
    assert chunk["sql"]["total"] == 3 and chunk["sql"]["eligible"] == 3
    assert chunk["collections"][world.chunk_generation]["points"] == 3
    assert chunk["collections"][world.chunk_generation]["missing"] == []

    assert memory["correction_chains"] == [
        {"memory_id": str(world.alice_superseded.id), "superseded_by": "later", "supersedes": []}
    ]

    # The report is the artifact the operator reads.
    assert json.loads(out.read_text())["kinds"]["memory"]["sql"]["eligible"] == 3

    # Strictly read-only (R28): the database file is byte-identical and every
    # point is untouched. SQLite re-creates the empty -wal/-shm sidecars for ANY
    # WAL-mode connection (read-only included), so the WAL is compared by
    # content: absent and empty are the same "no committed frame" state.
    after_db = _file_state(world.env.db_path)
    assert after_db[""] == before_db[""], "inventory must not write the database file"
    empty = hashlib.sha256(b"").hexdigest()
    assert (after_db["-wal"] in (None, empty)) == (before_db["-wal"] in (None, empty)), (
        "inventory must not add a committed frame to the WAL")
    assert _scroll_ids(world.memory_generation) == before_points


async def test_inventory_never_creates_a_missing_collection(env, tmp_path):
    """Read-only means no ``ensure_collection``: a missing target stays missing."""
    await _ladder(env)
    target = generation_name("memory")
    assert _client().collection_exists(target) is False

    report = env.cli.inventory(out=tmp_path / "inv.json")

    assert report["kinds"]["memory"]["collections"][target]["exists"] is False
    assert report["kinds"]["memory"]["collections"][target]["points"] == 0
    assert _client().collection_exists(target) is False


# ── backup ──────────────────────────────────────────────────────────────────


async def test_backup_snapshot_manifest_and_resume_does_not_overwrite(world, tmp_path):
    cli = world.env.cli
    dest = tmp_path / "backups"
    dest.mkdir()
    uploads = world.env.tmp_path / "uploads"
    uploads.mkdir(exist_ok=True)
    (uploads / "note.txt").write_text("kept")

    report = cli.backup(dest_dir=dest)

    snapshot = dest / f"{DB_NAME}.pre-p1b.bak"
    manifest_path = Path(f"{snapshot}.manifest.json")
    assert snapshot.exists() and manifest_path.exists()
    assert report["reused"] is False

    manifest = json.loads(manifest_path.read_text())
    assert manifest["db"]["sha256"] == hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert manifest["db"]["user_version"] == database.SQLITE_SCHEMA_VERSION
    assert {entry["path"] for entry in manifest["files"]} >= {"uploads/note.txt"}
    assert manifest["missing"] == ["chroma"]

    # The snapshot is a real, readable database carrying the fixture data.
    conn = sqlite3.connect(snapshot)
    try:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 5
    finally:
        conn.close()

    before = _file_state(snapshot)
    stats = snapshot.stat()
    again = cli.backup(dest_dir=dest)
    assert again["reused"] is True
    assert _file_state(snapshot) == before
    assert snapshot.stat().st_mtime_ns == stats.st_mtime_ns


async def test_backup_refuses_an_empty_existing_snapshot(world, tmp_path):
    dest = tmp_path / "backups"
    dest.mkdir()
    (dest / f"{DB_NAME}.pre-p1b.bak").write_bytes(b"")
    with pytest.raises(world.env.cli.MigrationRefused, match="empty or unreadable"):
        world.env.cli.backup(dest_dir=dest)


async def test_backup_treats_a_blank_source_path_as_absent(world, tmp_path, monkeypatch):
    """F3: a blank setting must never mean ``Path(".")``.

    ``Path("")`` passes ``is_dir()``, so the old code copied the whole working
    directory (sha256-ing every file) into the backup when a path was unset.
    """
    cli = world.env.cli
    monkeypatch.setattr(settings, "FS_STORAGE_PATH", "   ")
    monkeypatch.setattr(settings, "LEGACY_CHROMA_PATH", "")

    report = cli.backup(dest_dir=tmp_path / "backups")

    assert report["missing"] == ["uploads", "chroma"]
    manifest = json.loads(Path(report["manifest"]).read_text())
    assert manifest["files"] == [], "a blank path copies nothing"
    assert not (Path(report["dir"]) / "uploads").exists()


async def test_backup_refuses_a_destination_inside_a_source_tree(world, tmp_path):
    """F3: ``--dir`` inside a copied tree would recurse into its own output."""
    cli = world.env.cli
    uploads = world.env.tmp_path / "uploads"
    uploads.mkdir(exist_ok=True)
    (uploads / "note.txt").write_text("kept")
    inside = uploads / "backups"

    with pytest.raises(cli.MigrationRefused, match="inside the uploads source tree"):
        cli.backup(dest_dir=inside)
    assert not inside.exists(), "the refusal happens before anything is written"


async def test_backup_refuses_a_destination_that_owns_a_source_tree(world):
    """The other half of F3: ``--dir`` that is a PARENT of a copied tree.

    Each tree is copied to ``{dir}/{label}``, so ``--dir /data`` with
    ``FS_STORAGE_PATH=/data/uploads`` copies the uploads tree onto its own
    source — refused before anything is written, exactly like the inside case.
    """
    cli = world.env.cli
    uploads = world.env.tmp_path / "uploads"
    uploads.mkdir(exist_ok=True)
    (uploads / "note.txt").write_text("kept")
    before = sorted(p.name for p in world.env.tmp_path.iterdir())

    with pytest.raises(cli.MigrationRefused, match="CONTAINS the uploads source tree"):
        cli.backup(dest_dir=world.env.tmp_path)
    # The refusal happens before anything is written: the directory (which
    # already holds the DB + its ladder milestone backup) is untouched.
    assert sorted(p.name for p in world.env.tmp_path.iterdir()) == before


# ── backfill ────────────────────────────────────────────────────────────────


async def test_backfill_keyset_is_idempotent_and_checkpoint_advances(world, monkeypatch):
    cli = world.env.cli
    seen: list[list[str]] = []
    real = _fake_embed_sync

    def recording(texts: list[str]) -> list[list[float]]:
        seen.append(list(texts))
        return real(texts)

    monkeypatch.setattr(cli, "embed_texts_sync", recording)

    first = cli.backfill(kind="memory", batch=1)

    entry = json.loads(cli.checkpoint_path().read_text())["checkpoints"][
        f"{world.memory_generation}|memory"]
    assert entry["generation"] == world.memory_generation
    assert entry["kind"] == "memory"
    assert entry["last_id"] == str(max(memory.id for memory in world.memories))
    assert entry["tenant"] in {str(world.alice.id), str(world.bob.id)}
    assert first["upserted"] == 3
    assert first["batches"] == 5  # every row is visited, eligible or not
    assert first["excluded"] + first["skipped_unowned"] == 2
    assert first["gc_deleted"] == 3  # superseded point + orphan + malformed
    ids_after_first = _scroll_ids(world.memory_generation)
    assert ids_after_first == {str(memory.id) for memory in
                               (world.alice_current, world.bob_current, world.bob_extra)}

    second = cli.backfill(kind="memory", batch=1)

    assert second["upserted"] == 3 and second["rows_per_sec"] > 0
    assert _scroll_ids(world.memory_generation) == ids_after_first
    assert len(seen) == 6, "two runs, one embed per eligible batch"
    assert sorted(seen[:3]) == sorted(seen[3:]), "the same rows, no duplicates"


async def test_backfill_resumes_after_a_crash_without_re_embedding(world, monkeypatch):
    cli = world.env.cli
    calls: list[list[str]] = []
    real = _fake_embed_sync

    def crashing(texts: list[str]) -> list[list[float]]:
        if len(calls) >= 2:
            raise RuntimeError("simulated crash after the 2nd acked batch")
        calls.append(list(texts))
        return real(texts)

    monkeypatch.setattr(cli, "embed_texts_sync", crashing)
    with pytest.raises(RuntimeError, match="simulated crash"):
        cli.backfill(kind="memory", batch=1)

    assert len(calls) == 2, "the crash lands after the 2nd acked batch"
    acked = {text for batch in calls for text in batch}
    checkpoint = json.loads(cli.checkpoint_path().read_text())[
        "checkpoints"][f"{world.memory_generation}|memory"]
    assert uuid.UUID(checkpoint["last_id"]) in {memory.id for memory in world.memories}

    monkeypatch.setattr(cli, "embed_texts_sync", recording(calls, real))
    resumed = cli.backfill(kind="memory", batch=1, resume=True)

    assert resumed["resumed_from"] == checkpoint["last_id"]
    assert resumed["upserted"] == 1
    new_texts = {text for batch in calls[2:] for text in batch}
    assert acked & new_texts == set(), "acked rows are never re-embedded"
    assert _scroll_ids(world.memory_generation) == {
        str(memory.id) for memory in (world.alice_current, world.bob_current, world.bob_extra)}


def recording(calls: list[list[str]], real):
    def _record(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return real(texts)
    return _record


async def test_backfill_batch_bounds_and_reports_throughput(world):
    report = world.env.cli.backfill(kind="chunk", batch=2)
    assert report["kind"] == "chunk"
    assert report["batches"] == 2  # 3 rows, batch 2
    assert report["upserted"] == 3
    assert report["elapsed_s"] >= 0 and report["rows_per_sec"] >= 0


async def test_backfill_gc_deletes_points_for_ineligible_rows(world):
    cli = world.env.cli
    # A deleted row whose point survived (the SQL cascade happened, Qdrant did not).
    orphan = str(uuid.uuid4())
    _raw_upsert(world.memory_generation, orphan,
                {"kind": "memory", "memory_id": orphan, "user_id": str(world.alice.id),
                 "orivory_embed_fingerprint": canonical_fingerprint(FINGERPRINT),
                 "orivory_embed_generation": _token(), "orivory_memory_revision": 1})

    report = cli.backfill(kind="memory", batch=2)

    ids = _scroll_ids(world.memory_generation)
    assert report["gc_deleted"] >= 2
    assert orphan not in ids
    assert world.orphan_id not in ids
    assert str(world.alice_superseded.id) not in ids
    assert ids == {str(world.alice_current.id), str(world.bob_current.id),
                   str(world.bob_extra.id)}

    chunks = cli.backfill(kind="chunk", batch=2)
    assert chunks["upserted"] == 3
    assert _scroll_ids(world.chunk_generation) == {
        str(chunk.id) for chunk in (*world.alice_chunks, *world.bob_chunks)}


async def test_backfill_skips_chunks_without_an_owner(world):
    cli = world.env.cli
    async with world.env.sessions() as db:
        chunk = DocumentChunk(id=uuid.uuid4(), document_id=world.bob_doc.id,
                              content="unowned", chunk_index=9, revision=1, chunk_metadata={})
        db.add(chunk)
        await db.commit()
    # Give it a point anyway: the migration owns the stale point.
    _raw_upsert(world.chunk_generation, str(chunk.id),
                {"kind": "chunk", "chunk_id": str(chunk.id), "user_id": str(world.bob.id),
                 "conversation_id": "", "document_id": str(world.bob_doc.id), "revision": 1,
                 "fingerprint": canonical_fingerprint(FINGERPRINT), "content": "unowned"})

    report = cli.backfill(kind="chunk", batch=10)

    assert report["skipped_unowned"] == 1
    assert str(chunk.id) not in _scroll_ids(world.chunk_generation)


# ── verify ──────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def built(world) -> SimpleNamespace:
    """Both kinds backfilled and GC'd: the state ``verify`` must accept."""
    world.env.cli.backfill(kind="memory", batch=2)
    world.env.cli.backfill(kind="chunk", batch=2)
    return world


async def test_verify_accepts_a_fully_backfilled_generation(built):
    report = built.env.cli.verify(kind="memory")
    assert report["ok"] is True, report["findings"]
    assert report["generation"] == built.memory_generation
    assert report["sql_eligible"] == 3 and report["points"] == 3
    assert report["findings"] == {"missing": [], "stale_revision": [], "fingerprint": [],
                                  "absence": [], "tenant_mismatch": [], "malformed_id": []}
    assert built.env.cli.verify(kind="chunk")["ok"] is True


async def test_verify_flags_a_missing_point_even_beyond_one_scroll_page(env):
    """A full scan, not a sample: every page is audited."""
    cli = env.cli
    await _ladder(env)
    own = await _ownership(env.sessions)
    for index in range(300):
        await _add_memory(env.sessions, own.alice.id, f"bulk memory {index:03d}")
    cli.backfill(kind="memory", batch=64)
    generation = generation_name("memory")
    ids = sorted(_scroll_ids(generation))
    assert len(ids) == 300

    _client().delete(collection_name=generation,
                     points_selector=qm.PointIdsList(points=[ids[-1]]))
    report = cli.verify(kind="memory")

    assert report["ok"] is False
    assert report["findings"]["missing"] == [ids[-1]]
    assert report["points"] == 299, "the audit counts every point it scanned"


async def test_verify_flags_revision_fingerprint_and_absence(built):
    cli = built.env.cli
    generation = built.memory_generation
    # A newer SQL revision that was never indexed.
    async with built.env.sessions() as db:
        await db.execute(update(Memory).where(Memory.id == built.bob_current.id).values(revision=2))
        await db.commit()
    # A point carrying the wrong embedding contract (same dim, different token).
    _raw_upsert(generation, str(built.alice_current.id),
                {"kind": "memory", "memory_id": str(built.alice_current.id),
                 "user_id": str(built.alice.id), "orivory_memory_revision": 1,
                 "orivory_embed_fingerprint": canonical_fingerprint(FINGERPRINT),
                 "orivory_embed_generation": "deadbeef"})
    # A superseded row whose stale point is still present (GC did not run).
    _raw_upsert(generation, str(built.alice_superseded.id),
                {"kind": "memory", "memory_id": str(built.alice_superseded.id),
                 "user_id": str(built.alice.id), "orivory_memory_revision": 1,
                 "orivory_embed_fingerprint": canonical_fingerprint(FINGERPRINT),
                 "orivory_embed_generation": _token()})

    report = cli.verify(kind="memory")

    assert report["ok"] is False
    assert report["findings"]["stale_revision"] == [str(built.bob_current.id)]
    assert report["findings"]["fingerprint"] == [str(built.alice_current.id)]
    assert report["findings"]["absence"] == [str(built.alice_superseded.id)]

    # A dirty row's point is a residual the migration must own too (R29b).
    async with built.env.sessions() as db:
        await db.execute(update(Memory).where(Memory.id == built.bob_current.id).values(
            revision=1, extra_metadata={"cm_derived_dirty": True}))
        await db.commit()
    again = cli.verify(kind="memory")
    assert str(built.bob_current.id) in again["findings"]["absence"]


async def test_verify_requires_the_manifest_row_for_an_empty_dataset(env):
    cli = env.cli
    target = generation_name("memory")

    report = cli.verify(kind="memory")
    assert report["ok"] is False
    assert report["no_manifest_row"] is True

    await _ladder(env)
    empty = cli.verify(kind="memory")

    assert empty["ok"] is True, empty["findings"]
    assert empty["sql_eligible"] == 0 and empty["points"] == 0
    assert empty["generation"] == target


async def test_verify_flags_a_suppressed_projection(built):
    """A suppressed source identity must not keep a servable vector (R28)."""
    cli = built.env.cli
    async with built.env.sessions() as db:
        memory = await db.get(Memory, built.alice_current.id)
        memory.source_type = "file_upload"
        memory.source_ref = "doc-1"
        db.add(MemorySuppression(id=uuid.uuid4().hex, user_id=memory.user_id,
                                 source_ref="doc-1", reason="forgotten"))
        await db.commit()

    report = cli.verify(kind="memory")

    assert report["ok"] is False
    assert report["findings"]["absence"] == [str(built.alice_current.id)]
    # And the migration owns the stale point: the next backfill GCs it.
    assert cli.backfill(kind="memory", batch=10)["gc_deleted"] == 1


async def test_verify_flags_a_point_whose_payload_tenant_drifted(built):
    """F5: the payload ``user_id`` is the read path's filter (the security
    boundary) — a drifted point is invisible to its owner and visible to
    somebody else, so the audit must name it."""
    cli = built.env.cli
    generation = built.memory_generation
    _raw_upsert(generation, str(built.bob_current.id), {
        "kind": "memory", "memory_id": str(built.bob_current.id),
        "user_id": str(built.alice.id), "orivory_memory_revision": 1,
        "orivory_embed_fingerprint": canonical_fingerprint(FINGERPRINT),
        "orivory_embed_generation": _token()})

    report = cli.verify(kind="memory")

    assert report["ok"] is False
    assert report["findings"]["tenant_mismatch"] == [str(built.bob_current.id)]
    # The inventory report carries the same finding for the operator...
    audit = cli.inventory()["kinds"]["memory"]["collections"][generation]
    assert audit["tenant_mismatch"] == [str(built.bob_current.id)]
    # ...and the flip is gated on the audit, so this point blocks the cutover.
    with pytest.raises(cli.VerifyFailed, match="tenant_mismatch"):
        cli.cutover(yes=True)


# ── cutover ─────────────────────────────────────────────────────────────────


async def _pending_intent(env, *, kind: str, target_generation: str, entity_id: str) -> None:
    async with env.sessions() as db:
        db.add(IndexOutbox(
            kind=kind, entity_id=entity_id, tenant_id=uuid.uuid4().hex, revision=1,
            operation="upsert", target_generation=target_generation, status="pending",
        ))
        await db.commit()


async def _drift_pointer_back(env) -> None:
    """Model the pre-flip world: the transitional rows are the active ones."""
    async with env.sessions() as db:
        await db.execute(update(IndexGeneration).values(is_active=False))
        for kind, name in (("memory", vector_store.COLLECTION_NAME),
                           ("chunk", outbox.CHUNK_TARGET_GENERATION)):
            db.add(IndexGeneration(id=uuid.uuid4().hex, kind=kind, generation=name,
                                   fingerprint=env.cli.fingerprint_token(), is_active=True))
        await db.commit()


async def test_the_expand_writes_the_new_rows_inactive_until_cutover_flips_them(env):
    """F1: the expand only writes; ``cutover`` is the flip.

    Nothing here is active after the expand, so the runtime keeps its old
    pointer (the transitional fallback) while the migration window is open.
    """
    cli = env.cli
    await _ladder(env)

    rows = await _manifest_rows(env)
    assert {row.generation for row in rows} == {
        generation_name("memory"), generation_name("chunk")}
    assert not any(row.is_active for row in rows), "the expand never moves the pointer"
    assert outbox.active_generation_sync()[0] == vector_store.COLLECTION_NAME, (
        "the runtime still serves the old generation")

    report = cli.backfill(kind="memory", batch=10)
    assert report["generation"] == generation_name("memory"), "written by NAME, not by the pointer"
    assert cli.verify(kind="memory")["ok"] is True

    cli.cutover(yes=True)

    rows = await _manifest_rows(env)
    active = {row.kind: row.generation for row in rows if row.is_active}
    assert active == {"memory": generation_name("memory"), "chunk": generation_name("chunk")}
    assert sum(row.is_active for row in rows) == 2, "exactly one active row per kind"
    cli.cutover(yes=True)  # idempotent: the flip never doubles a pointer
    assert sum(row.is_active for row in (await _manifest_rows(env))) == 2


async def test_cutover_refuses_without_yes_and_when_verify_fails(built):
    cli = built.env.cli
    with pytest.raises(cli.MigrationRefused, match="--yes"):
        cli.cutover(yes=False)

    # One missing point is enough to refuse the flip (both kinds must be green).
    _client().delete(collection_name=built.chunk_generation,
                     points_selector=qm.PointIdsList(points=[str(built.bob_chunks[0].id)]))
    with pytest.raises(cli.MigrationRefused, match="verify"):
        cli.cutover(yes=True)
    assert not cli.rollback_marker_path().exists(), "a refused cutover writes no marker"


async def test_cutover_flips_both_kinds_blocks_stale_intents_and_writes_a_marker(built):
    cli = built.env.cli
    await _drift_pointer_back(built.env)
    await _pending_intent(built.env, kind="memory",
                          target_generation=vector_store.COLLECTION_NAME,
                          entity_id=str(built.alice_current.id))
    await _pending_intent(built.env, kind="chunk",
                          target_generation=outbox.CHUNK_TARGET_GENERATION,
                          entity_id=str(built.alice_chunks[0].id))
    await _pending_intent(built.env, kind="memory",
                          target_generation=built.memory_generation,
                          entity_id=str(built.bob_current.id))

    report = cli.cutover(yes=True)

    rows = await _manifest_rows(built.env)
    active = {row.kind: row for row in rows if row.is_active}
    assert set(active) == {"memory", "chunk"}
    assert active["memory"].generation == built.memory_generation
    assert active["chunk"].generation == built.chunk_generation
    assert all(row.fingerprint == cli.fingerprint_token() for row in rows)
    assert report["previous"]["memory"] == vector_store.COLLECTION_NAME
    assert report["active"]["memory"] == built.memory_generation
    assert report["blocked_intents"] == 2
    assert report["blocked"]["memory"] == 1 and report["blocked"]["chunk"] == 1

    # The pointer the runtime reads is the flipped one, for both kinds.
    assert outbox.active_generation_sync() == (built.memory_generation, cli.fingerprint_token())
    assert outbox.active_generation_sync(kind="chunk") == (
        built.chunk_generation, cli.fingerprint_token())

    intents = await _outbox_rows(built.env)
    by_target = {row.target_generation: row for row in intents}
    assert by_target[vector_store.COLLECTION_NAME].status == "blocked"
    assert "superseded" in (by_target[vector_store.COLLECTION_NAME].last_error or "")
    assert by_target[outbox.CHUNK_TARGET_GENERATION].status == "blocked"
    assert by_target[built.memory_generation].status == "pending"

    marker = json.loads(cli.rollback_marker_path().read_text())
    assert marker["previous"]["memory"] == vector_store.COLLECTION_NAME
    assert marker["active"] == report["active"]
    # The origin of the migration (the pointer before the expand moved it) is
    # what a rollback tool restores: recorded once, at expand time.
    assert "pre_migration" in marker and marker["pre_migration"]["memory"]


async def test_cutover_is_idempotent(built):
    cli = built.env.cli
    first = cli.cutover(yes=True)
    second = cli.cutover(yes=True)
    assert second["active"] == first["active"]
    rows = [row for row in await _manifest_rows(built.env) if row.is_active]
    assert len(rows) == 2 and {row.kind for row in rows} == {"memory", "chunk"}


# ── quiesce ─────────────────────────────────────────────────────────────────


async def test_quiesce_refuses_while_a_live_lock_file_is_held(world):
    cli = world.env.cli
    cli.lock_path().write_text(f"{os.getpid()}\n")
    with pytest.raises(cli.MigrationRefused, match="pid"):
        cli.backfill(kind="memory", batch=1)
    assert cli.lock_path().exists(), "a held lock is never stolen"


async def test_quiesce_refuses_while_the_app_listens(world, monkeypatch):
    cli = world.env.cli
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    try:
        monkeypatch.setattr(settings, "APP_PORT", server.getsockname()[1])
        with pytest.raises(cli.MigrationRefused, match="listening"):
            cli.cutover(yes=True)
    finally:
        server.close()


async def test_a_stale_lock_is_replaced_and_the_run_completes(world):
    cli = world.env.cli
    cli.lock_path().write_text("999999999\n")  # no such pid
    report = cli.backfill(kind="memory", batch=10)
    assert report["upserted"] == 3
    assert not cli.lock_path().exists(), "the lock is released after the run"


async def test_cli_exit_codes_follow_the_reports(world, tmp_path):
    """The argv surface the operator types (and the runbook prints).

    0 = the command did what it says; 1 = it ran and a gate failed (findings);
    2 = it refused before doing anything.
    """
    cli = world.env.cli
    out = tmp_path / "inv.json"
    assert cli.main(["inventory", "--out", str(out)]) == 0
    assert out.exists()
    assert cli.main(["backfill", "--kind", "memory", "--batch", "2"]) == 0
    assert cli.main(["backfill", "--kind", "chunk", "--batch", "2"]) == 0
    assert cli.main(["verify", "--kind", "memory"]) == 0
    assert cli.main(["verify", "--kind", "chunk"]) == 0
    assert cli.main(["cutover"]) == 2, "cutover without --yes is refused, not run"
    assert cli.main(["cutover", "--yes"]) == 0

    # A finding is exit 1, not a refusal: verify, and a cutover its own audit
    # would refuse, must be distinguishable in a shell from a misuse (2).
    _client().delete(collection_name=world.memory_generation,
                     points_selector=qm.PointIdsList(points=[str(world.bob_current.id)]))
    assert cli.main(["verify", "--kind", "memory"]) == 1
    assert cli.main(["cutover", "--yes"]) == 1


async def test_an_aborted_backfill_exits_non_zero(world, monkeypatch):
    """A batch the store did not ack is not success (``complete: false``)."""
    cli = world.env.cli
    real_client = _client()

    class Unacked:
        def __getattr__(self, name):
            return getattr(real_client, name)

        def upsert(self, **kwargs):
            return SimpleNamespace(status="failed")

    monkeypatch.setattr(cli, "_client", lambda: Unacked())

    report = cli.backfill(kind="memory", batch=1)
    assert report["complete"] is False and report["errors"]
    assert cli.main(["backfill", "--kind", "memory", "--batch", "1"]) == 1
