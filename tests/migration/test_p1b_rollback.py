"""Task 6 — the isolated Chroma rollback rebuild and the restore drill.

Real SQLite file, real (temp) Chroma directory, real ID/tenant reconciliation
against the pre-P1b contract. Only the embedder is injected (deterministic
384-dim unit vectors): the artifact under test is the STORE the pre-P1b binary
reads plus the SQL stamp that lets the old ladder accept the file — not the
model weights.

Skipped wholesale once Task 7 drops ``chromadb`` from the runtime env
(``pytest.importorskip``): the rollback path then runs from its own venv built
from ``requirements-rollback.txt``, which this suite is not executed from.

Contract under test (brief + rulings R30-R33):

- the rebuilt memory collection is the pre-P1b ``Orivory_memories`` and the
  chunk collections are the per-conversation ``rag_conv_<id>`` names;
- the eligible set is the live reader's set: current rows only — superseded,
  dirty, suppressed and unowned rows must NOT be resurrected;
- the recorded contract is the LEGACY MEAN fingerprint (never CLS);
- the emitted DB is stamped ``user_version = 2`` with exactly one active
  generation row per kind, so the old ladder accepts the file;
- the tool reads the LIVE database (post-cutover writes survive) and modifies
  neither it nor the pre-P1b snapshot beside it;
- the fixture gates (tenant isolation, correction, forget) run BEFORE the tool
  reports ready, and a failing gate never reports ready;
- ``verify --restore-drill`` restores into a NEW directory and refuses to
  report ready without a matching ledger record.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, delete, event, select, text, update
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

pytest.importorskip("chromadb")

import chromadb

from app import database
from app import models as _models  # noqa: F401 — register every table
from app.config import settings
from app.database import Base
from app.models.conversation import Conversation
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.erasure_receipt import ErasureReceipt
from app.models.index_outbox import IndexGeneration
from app.models.memory import Memory, MemorySuppression
from app.models.user import User
from app.retrieval.embedding_fingerprint import (
    ARCTIC_CLS_FINGERPRINT,
    LEGACY_MEAN_FINGERPRINT,
    canonical_fingerprint,
    fingerprint_generation,
    generation_name,
)
from app.retrieval.memory import outbox, vector_store

DB_NAME = "rollback.db"
DIM = int(LEGACY_MEAN_FINGERPRINT["dim"])
MEAN_CANONICAL = canonical_fingerprint(LEGACY_MEAN_FINGERPRINT)
MEAN_TOKEN = fingerprint_generation(LEGACY_MEAN_FINGERPRINT)
CLS_TOKEN = fingerprint_generation(ARCTIC_CLS_FINGERPRINT)
MEMORY_COLLECTION = vector_store.COLLECTION_NAME
FORGOTTEN_SOURCE = "drive://forgotten-doc"


def _vector(text: str) -> list[float]:
    """Unit vector for a text: same text -> same vector, every run."""
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    raw = [((seed >> (i % 61)) & 0xFF) / 255.0 - 0.5 for i in range(DIM)]
    norm = sum(value * value for value in raw) ** 0.5
    return [value / norm for value in raw]


def _fake_embed(texts: list[str]) -> list[list[float]]:
    return [_vector(text) for text in texts]


def _file_state(path: Path) -> dict:
    state = {}
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(f"{path}{suffix}")
        state[suffix] = (
            hashlib.sha256(candidate.read_bytes()).hexdigest() if candidate.exists() else None
        )
    return state


def _empty() -> str:
    return hashlib.sha256(b"").hexdigest()


class _Sqlite:
    """A sync SQLAlchemy session over the test's own SQLite file."""

    def __init__(self, path: Path):
        self.engine = create_engine(
            f"sqlite:///{path}", connect_args={"check_same_thread": False}, poolclass=NullPool
        )
        event.listen(self.engine, "connect", database._configure_sqlite_connection)
        self.sessions = sessionmaker(bind=self.engine, expire_on_commit=False, autoflush=False)

    def __enter__(self) -> Session:
        self.session = self.sessions()
        return self.session

    def __exit__(self, *_exc) -> None:
        self.session.close()
        self.engine.dispose()


# ── the fixture world: a real post-cutover SQLite file ──────────────────────


@pytest.fixture
def env(tmp_path, monkeypatch, migrate_cli):
    """Alice (current/superseded/dirty/forgotten) + Bob (current), real chunks.

    Post-cutover state: schema v3, the two P1b generations ACTIVE at the CLS
    contract, the P1a transitional ``Orivory_memories`` row retired — exactly
    what the migration CLI's ``cutover`` leaves behind. A pre-P1b snapshot sits
    beside the live file, and one memory is written AFTER it is taken.
    """
    db_path = tmp_path / DB_NAME
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "note.txt").write_text("kept")
    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setattr(settings, "APP_PORT", 1)
    monkeypatch.setattr(settings, "FS_STORAGE_PATH", str(uploads))
    monkeypatch.setattr(settings, "CHROMA_LOCAL_PATH", str(tmp_path / "no-chroma"))

    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    event.listen(engine, "connect", database._configure_sqlite_connection)
    Base.metadata.create_all(engine)
    engine.dispose()

    world = SimpleNamespace(tmp_path=tmp_path, db_path=db_path, uploads=uploads)
    with _Sqlite(db_path) as db:
        alice = User(id=uuid.uuid4(), email="alice@test.invalid", hashed_password="x",
                     display_name="Alice", is_verified=True, is_active=True)
        bob = User(id=uuid.uuid4(), email="bob@test.invalid", hashed_password="x",
                   display_name="Bob", is_verified=True, is_active=True)
        db.add_all([alice, bob])
        db.flush()  # FK order: the ledger rows below quote these users
        alice_conv = Conversation(id=uuid.uuid4(), user_id=alice.id, document_count=1)
        bob_conv = Conversation(id=uuid.uuid4(), user_id=bob.id, document_count=1)
        db.add_all([alice_conv, bob_conv])
        alice_doc = Document(id=uuid.uuid4(), conversation_id=alice_conv.id,
                             filename="alice.txt", file_path="/tmp/alice.txt", chunk_count=2)
        bob_doc = Document(id=uuid.uuid4(), conversation_id=bob_conv.id,
                           filename="bob.txt", file_path="/tmp/bob.txt", chunk_count=1)
        db.add_all([alice_doc, bob_doc])

        def chunk(content: str, document, index: int, **meta) -> DocumentChunk:
            row = DocumentChunk(
                id=uuid.uuid4(), document_id=document.id, content=content, chunk_index=index,
                revision=1,
                chunk_metadata={
                    "document_id": str(document.id),
                    "conversation_id": str(document.conversation_id),
                    "filename": document.filename,
                    "chunk_type": "child",
                    "child_index": index,
                    **meta,
                },
            )
            db.add(row)
            return row

        alice_chunks = [chunk("alice chunk a", alice_doc, 0), chunk("alice chunk b", alice_doc, 1)]
        bob_chunks = [chunk("bob chunk a", bob_doc, 0)]
        # No resolvable conversation: the drain's own rule makes it unindexable.
        unowned = DocumentChunk(
            id=uuid.uuid4(), document_id=alice_doc.id, content="unowned", chunk_index=9,
            revision=1, chunk_metadata={"document_id": str(alice_doc.id)},
        )
        db.add(unowned)

        def memory(user, content: str, **kw) -> Memory:
            row = Memory(id=uuid.uuid4(), user_id=user.id, content=content, tags=[],
                         extra_metadata=kw.pop("extra_metadata", {}), **kw)
            db.add(row)
            return row

        alice_current = memory(alice, "alice current")
        alice_superseded = memory(
            alice, "alice old fact",
            extra_metadata={"cm_superseded_by": "later", "cm_subject": "alice"},
        )
        alice_dirty = memory(alice, "alice dirty", extra_metadata={"cm_derived_dirty": True})
        alice_forgotten = memory(alice, "alice forgotten", source_ref=FORGOTTEN_SOURCE)
        bob_current = memory(bob, "bob current")
        bob_extra = memory(bob, "bob extra")
        db.add(MemorySuppression(id=uuid.uuid4().hex, user_id=alice.id,
                                 source_ref=FORGOTTEN_SOURCE, reason="user_forget"))
        db.add(ErasureReceipt(id=uuid.uuid4(), user_id=alice.id,
                              requested_memory_ids=[str(uuid.uuid4())], status="completed",
                              detail={"deleted": 1}))
        # The post-cutover manifest: both P1b rows ACTIVE at CLS, the P1a
        # transitional row retired at the legacy mean contract.
        db.add_all([
            IndexGeneration(id=uuid.uuid4().hex, kind="memory",
                            generation=generation_name("memory"), fingerprint=CLS_TOKEN,
                            is_active=True),
            IndexGeneration(id=uuid.uuid4().hex, kind="chunk",
                            generation=generation_name("chunk"), fingerprint=CLS_TOKEN,
                            is_active=True),
            IndexGeneration(id=uuid.uuid4().hex, kind="memory", generation=MEMORY_COLLECTION,
                            fingerprint=MEAN_TOKEN, is_active=False),
        ])
        db.commit()
        world.alice, world.bob = alice, bob
        world.alice_conv, world.bob_conv = alice_conv, bob_conv
        world.alice_doc, world.bob_doc = alice_doc, bob_doc
        world.alice_chunks, world.bob_chunks, world.unowned = alice_chunks, bob_chunks, unowned
        world.alice_current, world.alice_superseded = alice_current, alice_superseded
        world.alice_dirty, world.alice_forgotten = alice_dirty, alice_forgotten
        world.bob_current, world.bob_extra = bob_current, bob_extra
        db.execute(text("PRAGMA user_version = 3"))

    # The pre-P1b milestone snapshot beside the live file (written by the ladder
    # in a real install) plus the marker the cutover wrote after the flip.
    world.snapshot = Path(f"{db_path}.pre-p1b.bak")
    source = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    snapshot_conn = sqlite3.connect(world.snapshot)
    try:
        source.backup(snapshot_conn)
    finally:
        snapshot_conn.close()
        source.close()  # never leave a handle on the live file
    world.marker = Path(f"{db_path}.p1b-rollback-marker.json")
    world.marker.write_text(json.dumps({
        "cutover_at": "2026-09-15T00:00:00+00:00",
        "previous": {"memory": MEAN_TOKEN, "chunk": None},
        "active": {"memory": generation_name("memory"), "chunk": generation_name("chunk")},
    }))
    return world


def _live_write(env, content: str) -> Memory:
    """A write that lands AFTER the snapshot: the rollback must still see it."""
    with _Sqlite(env.db_path) as db:
        row = Memory(id=uuid.uuid4(), user_id=env.alice.id, content=content, tags=[])
        db.add(row)
        db.commit()
        return SimpleNamespace(id=row.id)


def _open_chroma(path: Path):
    return chromadb.PersistentClient(path=str(path))


def _ids(collection) -> set[str]:
    return set(collection.get(include=[])["ids"])


# ── the rollback rebuild ────────────────────────────────────────────────────


def test_rebuild_is_the_eligible_set_at_the_legacy_mean_contract(env, rollback_cli, tmp_path):
    chroma_path = tmp_path / "chroma-rollback"
    report = rollback_cli.rollback(
        db_path=env.db_path, chroma_path=chroma_path, embed_passages=_fake_embed
    )

    assert report["ok"] is True and report["ready"] is True
    assert report["db"]["user_version"] == 2
    assert report["manifest"] == {"memory": MEMORY_COLLECTION,
                                  "chunk": outbox.CHUNK_TARGET_GENERATION}

    client = _open_chroma(chroma_path)
    memory = client.get_collection(MEMORY_COLLECTION)
    # Eligible: current rows only — never the superseded, dirty, suppressed or
    # unowned ones, and never a resurrected point.
    assert _ids(memory) == {str(env.alice_current.id), str(env.bob_current.id),
                            str(env.bob_extra.id)}
    for excluded in (env.alice_superseded, env.alice_dirty, env.alice_forgotten):
        assert str(excluded.id) not in _ids(memory)

    # The recorded contract is the LEGACY MEAN one — the reason the old binary
    # can read these vectors at all.
    assert memory.metadata["orivory_embed_fingerprint"] == MEAN_CANONICAL
    assert memory.metadata["orivory_embed_generation"] == MEAN_TOKEN
    assert memory.metadata["orivory_embed_dim"] == DIM
    assert memory.metadata["orivory_embed_backend"] == "local-arctic", (
        "the stamp describes the CONTRACT, not the ambient settings")
    assert MEAN_CANONICAL != canonical_fingerprint(ARCTIC_CLS_FINGERPRINT)

    # Per-conversation chunk collections, under the names the old binary reads.
    alice_name = f"rag_conv_{env.alice_conv.id}"
    bob_name = f"rag_conv_{env.bob_conv.id}"
    assert _ids(client.get_collection(alice_name)) == {str(c.id) for c in env.alice_chunks}
    assert _ids(client.get_collection(bob_name)) == {str(c.id) for c in env.bob_chunks}
    assert str(env.unowned.id) not in _ids(client.get_collection(alice_name))
    assert client.get_collection(alice_name).metadata["orivory_embed_fingerprint"] == MEAN_CANONICAL

    # One collection per conversation that has eligible rows, nothing else.
    assert {c.name for c in client.list_collections()} == {MEMORY_COLLECTION, alice_name, bob_name}

    assert report["counts"]["memory"]["eligible"] == 3
    assert report["counts"]["memory"]["excluded"] == {
        "superseded": 1, "dirty": 1, "suppressed": 1, "unowned": 0,
    }


def test_emitted_db_is_readable_by_the_pre_p1b_ladder(env, rollback_cli, tmp_path):
    report = rollback_cli.rollback(
        db_path=env.db_path, chroma_path=tmp_path / "chroma-rollback",
        embed_passages=_fake_embed,
    )
    emitted = Path(report["db"]["emitted"])
    assert emitted.exists() and emitted != env.db_path

    conn = sqlite3.connect(f"file:{emitted}?mode=ro", uri=True)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        # v2 shape the old ladder expects: the revision columns and the ledger.
        columns = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
        assert "revision" in columns
        assert conn.execute(
            "SELECT count(*) FROM memory_suppressions WHERE source_ref = ?", (FORGOTTEN_SOURCE,)
        ).fetchone()[0] == 1
        # Exactly ONE active row per kind, naming the store we just built.
        rows = conn.execute(
            "SELECT kind, generation, fingerprint FROM index_generations WHERE is_active = 1"
        ).fetchall()
        assert sorted(rows) == sorted([
            ("memory", MEMORY_COLLECTION, MEAN_TOKEN),
            ("chunk", outbox.CHUNK_TARGET_GENERATION, MEAN_TOKEN),
        ])
        assert conn.execute(
            "SELECT count(*) FROM index_generations WHERE fingerprint = ?", (CLS_TOKEN,)
        ).fetchone()[0] == 0
        # The old ladder's own insert-if-absent is a no-op: the row is there.
        assert conn.execute(
            "SELECT count(*) FROM index_generations WHERE kind = 'memory' AND generation = ?",
            (MEMORY_COLLECTION,),
        ).fetchone()[0] == 1
    finally:
        conn.close()

    # The live reader's rule still holds in the copy: a suppressed source stays
    # blocked (the ledger travels with the store, so a re-import is refused).
    with _Sqlite(emitted) as db:
        assert db.execute(select(IndexGeneration.is_active)).scalars().all().count(True) == 2


def test_rollback_reads_the_live_database_and_never_touches_the_snapshot(env, rollback_cli, tmp_path):
    late = _live_write(env, "written after the snapshot")
    before_db, before_snapshot = _file_state(env.db_path), _file_state(env.snapshot)
    before_marker = env.marker.read_bytes()

    report = rollback_cli.rollback(
        db_path=env.db_path, chroma_path=tmp_path / "chroma-rollback",
        embed_passages=_fake_embed,
    )

    client = _open_chroma(Path(report["chroma"]["path"]))
    assert str(late.id) in _ids(client.get_collection(MEMORY_COLLECTION)), (
        "the rollback sources the LIVE database, not the pre-P1b snapshot")
    assert str(late.id) not in _snapshot_ids(env.snapshot), (
        "the fixture must write that row AFTER the snapshot for this to mean anything")

    after_db, after_snapshot = _file_state(env.db_path), _file_state(env.snapshot)
    assert after_db[""] == before_db[""], "the live database must not be written"
    assert (after_db["-wal"] in (None, _empty())) == (before_db["-wal"] in (None, _empty()))
    # Snapshot: byte-identical, like the live file. SQLite itself leaves an
    # EMPTY -wal/-shm behind for ANY read-only open (the assertion above reads
    # it too), so only the database file and the committed-frame state count.
    assert after_snapshot[""] == before_snapshot[""]
    assert (after_snapshot["-wal"] in (None, _empty())) == (
        before_snapshot["-wal"] in (None, _empty())), "the snapshot gained a committed frame"
    assert env.marker.read_bytes() == before_marker


def test_rollback_reports_what_it_is_rolling_back_from(env, rollback_cli, tmp_path):
    report = rollback_cli.rollback(
        db_path=env.db_path, chroma_path=tmp_path / "chroma-rollback",
        embed_passages=_fake_embed,
    )
    assert report["rollback_from"]["source"] == "marker"
    assert report["rollback_from"]["active"] == {
        "memory": generation_name("memory"), "chunk": generation_name("chunk")}
    assert report["rollback_from"]["cutover_at"] == "2026-09-15T00:00:00+00:00"

    # Marker missing: the pre-migration pointer recorded at expand time is the
    # documented fallback. A second rollback needs its own emitted copy (the
    # default path is taken by the first run and is never overwritten).
    env.marker.unlink()
    Path(f"{env.db_path}.p1b-expand-record.json").write_text(json.dumps(
        {"recorded_at": "2026-09-14T00:00:00+00:00", "previous": {"memory": MEAN_TOKEN, "chunk": None}}
    ))
    again = rollback_cli.rollback(
        db_path=env.db_path, chroma_path=tmp_path / "chroma-rollback-2",
        out_db=tmp_path / "second.db", embed_passages=_fake_embed,
    )
    assert again["rollback_from"]["source"] == "expand-record"
    assert again["rollback_from"]["active"] == {"memory": MEAN_TOKEN, "chunk": None}


def test_rollback_refuses_half_states_and_unknown_contracts(env, rollback_cli, tmp_path):
    taken = tmp_path / "taken"
    taken.mkdir()
    (taken / "chroma.sqlite3").write_bytes(b"partial")
    with pytest.raises(rollback_cli.RollbackRefused):
        rollback_cli.rollback(db_path=env.db_path, chroma_path=taken, embed_passages=_fake_embed)

    with pytest.raises(rollback_cli.RollbackRefused):
        rollback_cli.rollback(db_path=env.db_path, chroma_path=tmp_path / "c1",
                              fingerprint="cls", embed_passages=_fake_embed)

    with pytest.raises(rollback_cli.RollbackRefused):
        rollback_cli.rollback(db_path=tmp_path / "missing.db", chroma_path=tmp_path / "c2",
                              embed_passages=_fake_embed)


def test_a_chroma_path_that_is_a_file_is_refused_instead_of_tracebacking(
    env, rollback_cli, tmp_path, capsys
):
    """M1: a FILE at --chroma-path is a refusal (exit 2), not a NotADirectoryError."""
    file_path = tmp_path / "not-a-store"
    file_path.write_text("")
    assert rollback_cli.main(
        ["--db", str(env.db_path), "--chroma-path", str(file_path)]
    ) == 2
    assert "is a file" in capsys.readouterr().err


def test_a_server_deployment_is_refused_like_the_drill(
    env, rollback_cli, migrate_cli, tmp_path, monkeypatch
):
    """I4: both tools refuse a server install, and the lite URL still builds."""
    with monkeypatch.context() as patch:
        patch.setattr(settings, "DATABASE_URL", "postgresql+asyncpg://user@host:5432/ragdb")
        assert migrate_cli.is_sqlite() is False
        with pytest.raises(rollback_cli.RollbackRefused, match="server deployment"):
            rollback_cli.rollback(db_path=env.db_path, chroma_path=tmp_path / "c-server",
                                  embed_passages=_fake_embed)

    # `sqlite+aiosqlite:///…` — the lite path every install here runs — passes.
    report = rollback_cli.rollback(db_path=env.db_path, chroma_path=tmp_path / "chroma-lite",
                                   embed_passages=_fake_embed)
    assert report["ok"] is True


def test_a_second_rollback_refuses_and_names_the_remedy(env, rollback_cli, tmp_path):
    """I2: the documented re-run path is a fresh build, and the refusal says so."""
    first = rollback_cli.rollback(
        db_path=env.db_path, chroma_path=tmp_path / "chroma-rollback",
        embed_passages=_fake_embed,
    )
    emitted = Path(first["db"]["emitted"])
    assert emitted.exists()

    # The default emitted copy is taken: the second run must be told what to do.
    with pytest.raises(rollback_cli.RollbackRefused) as excinfo:
        rollback_cli.rollback(db_path=env.db_path, chroma_path=tmp_path / "chroma-rollback-2",
                              embed_passages=_fake_embed)
    message = str(excinfo.value)
    assert str(emitted) in message and "--out-db" in message

    # ...and the store directory of the first run refuses too, naming both artifacts.
    with pytest.raises(rollback_cli.RollbackRefused) as excinfo:
        rollback_cli.rollback(db_path=env.db_path, chroma_path=tmp_path / "chroma-rollback",
                              embed_passages=_fake_embed)
    assert "--chroma-path" in str(excinfo.value)

    # The remedy the message names works: a second run with its own two paths.
    again = rollback_cli.rollback(
        db_path=env.db_path, chroma_path=tmp_path / "chroma-rollback-3",
        out_db=tmp_path / "second.db", embed_passages=_fake_embed,
    )
    assert again["ok"] is True
    assert Path(again["db"]["emitted"]) == tmp_path / "second.db"


def test_contract_gate_is_the_old_binarys_own_dim_guard(env, rollback_cli, tmp_path):
    """I1: backend, dim, generation and space are gated, not just the fingerprint.

    Each case is a store that reads back with one contract field wrong — the
    shapes a wrong rebuild or a hand-edited store produces. The gate must refuse
    every one of them, because ``check_collection_dim`` (the pre-P1b binary's
    real guard) refuses every one of them.
    """
    cases = {
        "backend": {"orivory_embed_backend": "openai"},
        "dim": {"orivory_embed_dim": 1024},
        "generation": {"orivory_embed_generation": CLS_TOKEN},
        "fingerprint": {"orivory_embed_fingerprint": canonical_fingerprint(ARCTIC_CLS_FINGERPRINT)},
        "space": {"hnsw:space": "l2"},
    }
    for name, override in cases.items():
        metadata = dict(rollback_cli.collection_metadata(DIM))
        metadata.update(override)
        client = _open_chroma(tmp_path / f"chroma-{name}")
        client.get_or_create_collection(MEMORY_COLLECTION, metadata=metadata)
        gates = rollback_cli.gate_findings(
            client=client, memory_collection=MEMORY_COLLECTION, chunk_collections={},
            expected={}, excluded={}, chunk_expected={}, token=MEAN_TOKEN,
        )
        assert f"collection:{name}" in gates["fingerprint"], (name, gates)


def test_contract_evidence_is_read_from_the_install_and_reported(env, rollback_cli, tmp_path):
    """I5: the mean assumption is checked against the install's own records."""
    report = rollback_cli.rollback(
        db_path=env.db_path, chroma_path=tmp_path / "chroma-1", embed_passages=_fake_embed,
    )
    evidence = report["contract_evidence"]
    assert evidence["mean"] is True, evidence
    assert "index_generations" in evidence["sources"]
    assert evidence["unexpected"] == []

    # No record names a pre-P1b contract: the tool still runs, and SAYS the
    # assumption is unverified instead of implying a check that did not run.
    with _Sqlite(env.db_path) as db:
        db.execute(delete(IndexGeneration).where(
            IndexGeneration.generation == MEMORY_COLLECTION))
        db.commit()
    again = rollback_cli.rollback(
        db_path=env.db_path, chroma_path=tmp_path / "chroma-2", out_db=tmp_path / "second.db",
        embed_passages=_fake_embed,
    )
    assert again["ok"] is True
    assert again["contract_evidence"]["mean"] is None
    assert "UNVERIFIED" in again["contract_evidence"]["note"]


def test_contract_evidence_refuses_an_install_that_names_another_contract(
    env, rollback_cli, tmp_path
):
    """I5: a live row recording a non-mean token for the mean generation refuses."""
    with _Sqlite(env.db_path) as db:
        db.execute(update(IndexGeneration)
                   .where(IndexGeneration.generation == MEMORY_COLLECTION)
                   .values(fingerprint=CLS_TOKEN))
        db.commit()
    with pytest.raises(rollback_cli.RollbackRefused, match="different contract"):
        rollback_cli.rollback(db_path=env.db_path, chroma_path=tmp_path / "chroma-1",
                              embed_passages=_fake_embed)

    # The expand record is the second source: a pointer recorded at a
    # non-mean generation refuses even with the manifest row gone.
    with _Sqlite(env.db_path) as db:
        db.execute(delete(IndexGeneration).where(
            IndexGeneration.generation == MEMORY_COLLECTION))
        db.commit()
    Path(f"{env.db_path}.p1b-expand-record.json").write_text(json.dumps(
        {"recorded_at": "2026-09-14T00:00:00+00:00",
         "previous": {"memory": generation_name("memory"), "chunk": None}}
    ))
    with pytest.raises(rollback_cli.RollbackRefused, match="different contract"):
        rollback_cli.rollback(db_path=env.db_path, chroma_path=tmp_path / "chroma-2",
                              embed_passages=_fake_embed)


def test_a_failing_gate_never_reports_ready(env, rollback_cli, tmp_path, monkeypatch):
    def _finding(**_kwargs):
        return {"tenant_isolation": ["leak"]}

    monkeypatch.setattr(rollback_cli, "gate_findings", _finding)
    report = rollback_cli.rollback(
        db_path=env.db_path, chroma_path=tmp_path / "chroma-rollback",
        embed_passages=_fake_embed,
    )
    assert report["ok"] is False and report["ready"] is False
    assert report["findings"] == {"tenant_isolation": ["leak"]}
    # The artifacts self-declare: a store nobody may boot carries the marker.
    assert (Path(report["chroma"]["path"]) / rollback_cli.NOT_READY_MARKER).exists()


def test_tenant_gate_fails_closed_on_a_drifted_point(env, rollback_cli, tmp_path):
    report = rollback_cli.rollback(
        db_path=env.db_path, chroma_path=tmp_path / "chroma-rollback",
        embed_passages=_fake_embed,
    )
    client = _open_chroma(Path(report["chroma"]["path"]))
    memory = client.get_collection(MEMORY_COLLECTION)
    assert report["gates"] == {} or not any(report["gates"].values())

    # A point whose tenant drifted is a security finding, not a detail: the
    # gate must catch it before anything is declared ready.
    memory.update(ids=[str(env.bob_current.id)],
                  metadatas=[{"user_id": str(env.alice.id)}])
    gates = rollback_cli.gate_findings(
        client=client,
        memory_collection=MEMORY_COLLECTION,
        chunk_collections={},
        expected={str(env.alice_current.id): str(env.alice.id),
                  str(env.bob_current.id): str(env.bob.id),
                  str(env.bob_extra.id): str(env.bob.id)},
        excluded={str(env.alice_superseded.id): "superseded",
                  str(env.alice_dirty.id): "dirty",
                  str(env.alice_forgotten.id): "suppressed"},
        chunk_expected={},
        token=MEAN_TOKEN,
    )
    assert str(env.bob_current.id) in gates["tenant_isolation"]
    # ...and bob's tenant no longer serves it: the served set must equal the
    # eligible set PER TENANT, not just as one bag of ids.
    assert str(env.bob_current.id) in gates["id_set"]


def test_gates_cover_correction_and_forget(env, rollback_cli, tmp_path):
    report = rollback_cli.rollback(
        db_path=env.db_path, chroma_path=tmp_path / "chroma-rollback",
        embed_passages=_fake_embed,
    )
    client = _open_chroma(Path(report["chroma"]["path"]))
    gates = rollback_cli.gate_findings(
        client=client,
        memory_collection=MEMORY_COLLECTION,
        chunk_collections={},
        expected={str(env.alice_current.id): str(env.alice.id),
                  str(env.bob_current.id): str(env.bob.id),
                  str(env.bob_extra.id): str(env.bob.id)},
        excluded={str(env.alice_superseded.id): "superseded",
                  str(env.alice_dirty.id): "dirty",
                  str(env.alice_forgotten.id): "suppressed"},
        chunk_expected={},
        token=MEAN_TOKEN,
    )
    assert gates == {}, "the rebuilt store passes every fixture gate"

    # Re-import a corrected row and a forgotten one by hand: both must land in
    # their own finding bucket, and neither may ever be reported ready.
    memory = client.get_collection(MEMORY_COLLECTION)
    memory.upsert(ids=[str(env.alice_superseded.id), str(env.alice_forgotten.id)],
                  embeddings=[_vector("alice old fact"), _vector("alice forgotten")],
                  documents=["alice old fact", "alice forgotten"],
                  metadatas=[{"user_id": str(env.alice.id), "memory_id": str(env.alice_superseded.id)},
                             {"user_id": str(env.alice.id), "memory_id": str(env.alice_forgotten.id)}])
    gates = rollback_cli.gate_findings(
        client=client,
        memory_collection=MEMORY_COLLECTION,
        chunk_collections={},
        expected={str(env.alice_current.id): str(env.alice.id),
                  str(env.bob_current.id): str(env.bob.id),
                  str(env.bob_extra.id): str(env.bob.id)},
        excluded={str(env.alice_superseded.id): "superseded",
                  str(env.alice_dirty.id): "dirty",
                  str(env.alice_forgotten.id): "suppressed"},
        chunk_expected={},
        token=MEAN_TOKEN,
    )
    assert str(env.alice_superseded.id) in gates["correction"]
    assert str(env.alice_forgotten.id) in gates["forget"]


def test_arctic_mean_wrappers_are_the_legacy_contract(monkeypatch):
    """R31: the rollback tool (and Task 8's ablation) embed through ONE path."""
    from app.retrieval import e5_local

    calls: list[dict] = []

    def _spy(texts, sess_fn, tok_fn, pooling="mean"):
        calls.append({"texts": list(texts), "pooling": pooling})
        return [[0.0] * 3 for _ in texts]

    monkeypatch.setattr(e5_local, "_encode_with", _spy)
    monkeypatch.setattr(e5_local, "arctic_files_cached", lambda: True)
    e5_local.arctic_embed_passages_mean(["a message"])
    e5_local.arctic_embed_queries_mean(["a question"])
    e5_local.arctic_embed_passages(["a message"])

    assert calls[0] == {"texts": ["a message"], "pooling": "mean"}
    assert calls[1]["pooling"] == "mean"
    assert calls[1]["texts"] == [e5_local.ARCTIC_QUERY_PREFIX + "a question"]
    assert calls[2]["pooling"] == "cls", "the live contract stays CLS"


# ── the restore drill ───────────────────────────────────────────────────────


def _snapshot_ids(path: Path) -> set[str]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return {row[0] for row in conn.execute("SELECT id FROM memories")}
    finally:
        conn.close()


def test_restore_drill_restores_into_a_new_directory_and_reports_ready(env, migrate_cli, tmp_path):
    dest = tmp_path / "backups"
    dest.mkdir()
    migrate_cli.backup(dest_dir=dest)
    snapshot = dest / f"{DB_NAME}.pre-p1b.bak"
    before = {path.name: _file_state(path) for path in dest.iterdir() if path.is_file()}
    live_before = _file_state(env.db_path)

    target = tmp_path / "restored"
    report = migrate_cli.restore_drill(backup_dir=dest, target=target)

    assert report["ok"] is True, report["checks"]
    restored = target / snapshot.name
    assert restored.exists(), "the drill restores into the NEW directory it was given"
    assert restored.read_bytes() == snapshot.read_bytes()
    assert (target / "uploads" / "note.txt").read_text() == "kept"
    assert sqlite3.connect(f"file:{restored}?mode=ro", uri=True).execute(
        "PRAGMA user_version").fetchone()[0] == 3
    assert report["checks"]["ledger"]["ok"] is True
    assert report["checks"]["absence"]["ok"] is True

    # Never writes into the original volume, and never touches the live DB.
    after = {path.name: _file_state(path) for path in dest.iterdir() if path.is_file()}
    assert after == before
    assert _file_state(env.db_path) == live_before


def test_restore_drill_refuses_a_stale_or_missing_ledger(env, migrate_cli, tmp_path):
    dest = tmp_path / "backups"
    dest.mkdir()
    migrate_cli.backup(dest_dir=dest)
    manifest_path = dest / f"{DB_NAME}.pre-p1b.bak.manifest.json"
    good = json.loads(manifest_path.read_text())
    assert good["deletion_ledger"]["suppressions"]["count"] == 1

    # Stale: the record no longer describes the ledger the backup carries.
    stale = json.loads(json.dumps(good))
    stale["deletion_ledger"]["suppressions"]["sha256"] = hashlib.sha256(b"other").hexdigest()
    manifest_path.write_text(json.dumps(stale))
    report = migrate_cli.restore_drill(backup_dir=dest, target=tmp_path / "restored-stale")
    assert report["ok"] is False
    assert report["checks"]["ledger"]["ok"] is False
    assert report["checks"]["ledger"]["findings"]

    # Missing: no ledger record at all — a restore that cannot prove forgotten
    # content stays forgotten is not a restore.
    missing = json.loads(json.dumps(good))
    missing.pop("deletion_ledger")
    manifest_path.write_text(json.dumps(missing))
    report = migrate_cli.restore_drill(backup_dir=dest, target=tmp_path / "restored-missing")
    assert report["ok"] is False
    assert report["checks"]["ledger"]["findings"]


def test_restore_drill_refuses_a_corrupt_or_unmanifested_backup(env, migrate_cli, tmp_path):
    dest = tmp_path / "backups"
    dest.mkdir()
    migrate_cli.backup(dest_dir=dest)
    snapshot = dest / f"{DB_NAME}.pre-p1b.bak"

    # Checksums: bytes that drifted from the manifest are not restorable.
    original = snapshot.read_bytes()
    snapshot.write_bytes(original + b"tampered")
    report = migrate_cli.restore_drill(backup_dir=dest, target=tmp_path / "restored-tampered")
    assert report["ok"] is False
    assert report["checks"]["checksum"]["ok"] is False
    snapshot.write_bytes(original)

    # A backup directory with no manifest at all cannot be verified.
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(migrate_cli.MigrationRefused):
        migrate_cli.restore_drill(backup_dir=empty, target=tmp_path / "restored-empty")


def test_restore_drill_absence_assertion_catches_a_resurrected_memory(env, migrate_cli, tmp_path):
    """A completed erasure receipt is a hard absence claim the restore must meet."""
    dest = tmp_path / "backups"
    dest.mkdir()
    migrate_cli.backup(dest_dir=dest)
    snapshot = dest / f"{DB_NAME}.pre-p1b.bak"
    manifest_path = dest / f"{DB_NAME}.pre-p1b.bak.manifest.json"

    # Widen the receipt to name a memory the snapshot still carries, then
    # re-record the checksums: only the absence assertion can catch it.
    conn = sqlite3.connect(snapshot)
    try:
        row = conn.execute("SELECT id FROM memories LIMIT 1").fetchone()
        conn.execute("UPDATE erasure_receipts SET requested_memory_ids = ?",
                     (json.dumps([row[0]]),))
        conn.commit()
        conn.execute("VACUUM")
    finally:
        conn.close()
    manifest = json.loads(manifest_path.read_text())
    manifest["db"]["sha256"] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    manifest["db"]["bytes"] = snapshot.stat().st_size
    ledger = migrate_cli._ledger_digest(str(snapshot))
    manifest["deletion_ledger"] = ledger
    manifest_path.write_text(json.dumps(manifest))

    report = migrate_cli.restore_drill(backup_dir=dest, target=tmp_path / "restored-absence")
    assert report["ok"] is False
    assert report["checks"]["absence"]["findings"]
    assert report["checks"]["checksum"]["ok"] is True


def test_restore_drill_uses_a_fresh_target_only(env, migrate_cli, tmp_path):
    dest = tmp_path / "backups"
    dest.mkdir()
    migrate_cli.backup(dest_dir=dest)
    target = tmp_path / "restored"
    target.mkdir()
    (target / "occupied.txt").write_text("mine")

    with pytest.raises(migrate_cli.MigrationRefused, match="not empty"):
        migrate_cli.restore_drill(backup_dir=dest, target=target)


def test_restore_drill_refuses_a_target_inside_the_backup_dir(env, migrate_cli, tmp_path):
    """M7: the drill never writes into the volume it is verifying."""
    dest = tmp_path / "backups"
    dest.mkdir()
    migrate_cli.backup(dest_dir=dest)

    with pytest.raises(migrate_cli.MigrationRefused, match="inside the backup directory"):
        migrate_cli.restore_drill(backup_dir=dest, target=dest / "restored")

    # The documented default is outside it, and still works.
    report = migrate_cli.restore_drill(backup_dir=dest)
    assert report["ok"] is True, report["checks"]
    assert Path(report["target"]) == tmp_path / "backups.restore"


def test_backup_reuses_a_verified_snapshot_and_refreshes_a_legacy_manifest(env, migrate_cli,
                                                                          tmp_path):
    """M9: the reuse branch (and the refresh it triggers) is exercised, not assumed."""
    dest = tmp_path / "backups"
    dest.mkdir()
    first = migrate_cli.backup(dest_dir=dest)
    snapshot = Path(first["db_backup"])
    manifest_path = Path(first["manifest"])
    before = snapshot.read_bytes()

    # Reuse: the same verified bytes are never re-snapshotted or overwritten.
    again = migrate_cli.backup(dest_dir=dest)
    assert again["reused"] is True
    assert Path(again["db_backup"]) == snapshot
    assert snapshot.read_bytes() == before

    # A manifest written before the drill learned to read these records: the
    # bytes are verified, so the records are recomputed from them.
    legacy = json.loads(manifest_path.read_text())
    legacy.pop("deletion_ledger")
    legacy["db"].pop("fingerprint")
    manifest_path.write_text(json.dumps(legacy))
    refreshed = migrate_cli.backup(dest_dir=dest)
    assert refreshed["reused"] is True
    assert snapshot.read_bytes() == before, "a refresh never re-snapshots"
    recorded = json.loads(manifest_path.read_text())
    assert recorded["deletion_ledger"] == migrate_cli._ledger_digest(str(snapshot))
    assert "refreshed_at" in recorded
    assert recorded["db"]["fingerprint"]["memory"], "the active token is recorded"

    # With the records back, the drill reads the same manifest and passes.
    report = migrate_cli.restore_drill(backup_dir=dest, target=tmp_path / "restored")
    assert report["ok"] is True, report["checks"]


def test_restore_drill_refuses_a_snapshot_with_no_checksum_manifest(env, migrate_cli, tmp_path):
    """M9: a snapshot nobody can verify is refused by name, not by accident."""
    dest = tmp_path / "backups"
    dest.mkdir()
    migrate_cli.backup(dest_dir=dest)
    Path(f"{dest / f'{DB_NAME}.pre-p1b.bak'}.manifest.json").unlink()

    with pytest.raises(migrate_cli.MigrationRefused, match="no checksum manifest"):
        migrate_cli.restore_drill(backup_dir=dest, target=tmp_path / "restored")
