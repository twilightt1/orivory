"""SQLite ladder v3 — the P1b generation rows are a DATA step, not DDL (R24).

A v2 install (the schema P1a produced) boots and must end with the TWO real
generation rows (memory + chunk, named by ``generation_name`` and stamped with
the current contract token) written **inactive**: the OLD pointer keeps serving,
and because that pointer IS the P1a transitional row (masked mean, i.e.
``LEGACY_TOKEN``) an install that has not been through ``migrate_qdrant.py
cutover`` fails LOUD — the read path's contract guard raises "different
embedding contract" — instead of answering every recall with an empty result
from a generation nobody has built yet (the test at the end of this module is
the pin). A fresh install has nothing to serve, so its rows are active from the
start. Either way the milestone backup is ``.pre-p1b.bak``.

The loud path needs that mismatch: with NO active row the guard has nothing to
reject — ``outbox.active_generation()`` falls back to the transitional name
with no fingerprint, an EMPTY generation is allowed, and reads answer ``[]``.
That is the Postgres/full-stack state (P1a never seeded ``index_generations``
there) — and this ladder's own v1→v3 path, which never had a P1a row to keep
active.

The carry item this pins (T1-M4): the old ``_seed_transitional_generation``
re-created the ``Orivory_memories`` row on EVERY boot. The ladder writes the real
rows ONCE, on the version transition, and a later boot never touches the
manifest again — a restart must not re-assert a pointer the cutover moved, nor
undo a rollback (F2).
"""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app import database, models  # noqa: F401 — register every model on Base
from app.config import settings
from app.database import Base
from app.retrieval.embedding_fingerprint import (
    LEGACY_MEAN_FINGERPRINT,
    current_fingerprint,
    fingerprint_generation,
    generation_name,
)
from app.retrieval.memory import outbox
from app.retrieval.memory.vector_store import COLLECTION_NAME
from tests.lite.test_sqlite_schema_v2 import V2_TABLES, _seed_v1_rows, _strip_v2_objects

TOKEN = fingerprint_generation(current_fingerprint())
# What the P1a seed stamped: the contract BEFORE P1b (mean pooling). An
# un-migrated install still serves that row, and its vectors were built with it.
LEGACY_TOKEN = fingerprint_generation(LEGACY_MEAN_FINGERPRINT)
KIND_NAMES = {"memory": generation_name("memory"), "chunk": generation_name("chunk")}


async def _engine(tmp_path: Path, name: str):
    eng = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / name}", poolclass=NullPool)
    event.listen(eng.sync_engine, "connect", database._configure_sqlite_connection)
    return eng


async def _schema(conn) -> tuple[int, set[str]]:
    version = int((await conn.execute(text("PRAGMA user_version"))).scalar_one())
    tables = {r[0] for r in (await conn.execute(
        text("SELECT name FROM sqlite_master WHERE type='table'"))).all()}
    return version, tables


async def _generations(conn) -> list[tuple[str, str, str, int]]:
    rows = (await conn.execute(text(
        "SELECT kind, generation, fingerprint, is_active FROM index_generations"))).all()
    return [tuple(row) for row in rows]


async def _manifest_state(eng) -> list[tuple[str, str, str, int]]:
    async with eng.connect() as conn:
        return sorted(await _generations(conn))


async def _insert_transitional(conn) -> None:
    """The P1a seed: one row naming the pre-P1b collection spelling."""
    await conn.execute(text(
        "INSERT INTO index_generations (id, kind, generation, fingerprint, is_active, created_at)"
        " VALUES (:id, 'memory', :gen, :fp, 1, CURRENT_TIMESTAMP)"),
        {"id": uuid.uuid4().hex, "gen": COLLECTION_NAME, "fp": LEGACY_TOKEN})


@pytest_asyncio.fixture
async def v2_db(tmp_path, monkeypatch):
    """A v2 install: current schema, the P1a transitional row, ``user_version`` 2."""
    eng = await _engine(tmp_path, "v2.sqlite")
    sessions = async_sessionmaker(eng, class_=AsyncSession, expire_on_commit=False, autoflush=False)
    monkeypatch.setattr(database, "engine", eng)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    # The read path resolves the pointer through the shared sessionmaker, so a
    # test that READS (not just boots) must point it at this file too.
    monkeypatch.setattr(database, "AsyncSessionLocal", sessions)
    monkeypatch.setattr(outbox, "AsyncSessionLocal", sessions)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _insert_transitional(conn)
        await conn.execute(text("PRAGMA user_version = 2"))
    yield eng, tmp_path
    await eng.dispose()


async def test_v2_upgrade_writes_the_new_rows_inactive_and_keeps_the_old_pointer(v2_db):
    """The expand writes the target rows; the OLD pointer serves until cutover."""
    eng, tmp_path = v2_db
    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version, _ = await _schema(conn)
        rows = await _generations(conn)

    assert version == database.SQLITE_SCHEMA_VERSION == 3
    active = {(kind, generation) for kind, generation, _, is_active in rows if is_active}
    assert active == {("memory", COLLECTION_NAME)}, "the old pointer keeps serving"
    written = {(kind, generation) for kind, generation, _, is_active in rows if not is_active}
    assert written == {("memory", generation_name("memory")), ("chunk", generation_name("chunk"))}
    # The new rows carry the current contract; the predecessor keeps its own
    # (rewriting it would claim the old vectors match the new contract).
    assert {fp for kind, generation, fp, _ in rows if (kind, generation) in written} == {TOKEN}
    assert {fp for kind, generation, fp, _ in rows if generation == COLLECTION_NAME} == {LEGACY_TOKEN}

    backups = list(Path(tmp_path).glob("*.pre-p1b.bak"))
    assert len(backups) == 1, "exactly one pre-P1b backup"
    assert not list(Path(tmp_path).glob("*.pre-v2.bak"))
    snapshot = sqlite3.connect(backups[0])
    try:
        names = [row[0] for row in snapshot.execute(
            "SELECT generation FROM index_generations").fetchall()]
    finally:
        snapshot.close()
    assert names == [COLLECTION_NAME], "the backup predates the v3 data step"


async def test_fresh_install_gets_the_rows_without_a_backup(tmp_path, monkeypatch):
    eng = await _engine(tmp_path, "fresh.sqlite")
    monkeypatch.setattr(database, "engine", eng)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    try:
        await database.bootstrap_sqlite()

        async with eng.connect() as conn:
            version, tables = await _schema(conn)
            rows = await _generations(conn)
        assert version == 3 and V2_TABLES <= tables
        assert {(row[0], row[1]) for row in rows} == {
            ("memory", generation_name("memory")), ("chunk", generation_name("chunk"))}
        assert all(row[3] == 1 for row in rows), "nothing to serve yet: the rows are active"
        assert not list(Path(tmp_path).glob("*.pre-p1b.bak")), "no data step, no backup"
    finally:
        await eng.dispose()


async def test_v1_install_upgrades_straight_through_with_one_p1b_backup(tmp_path, monkeypatch):
    """v1 -> v2 (DDL) -> v3 (data): the milestone backup is the pre-any-change file."""
    eng = await _engine(tmp_path, "v1.sqlite")
    monkeypatch.setattr(database, "engine", eng)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    try:
        async with eng.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await _strip_v2_objects(conn)
            await _seed_v1_rows(conn)
            await conn.execute(text("PRAGMA user_version = 1"))

        await database.bootstrap_sqlite()

        async with eng.connect() as conn:
            version, tables = await _schema(conn)
            memories = (await conn.execute(text("SELECT content, revision FROM memories"))).all()
            rows = await _generations(conn)
        assert version == 3 and V2_TABLES <= tables
        assert [tuple(row) for row in memories] == [("v1 memory text", 1)]
        assert {row[0] for row in rows} == {"memory", "chunk"}
        assert not any(row[3] for row in rows), "an upgrade never moves the pointer"

        backups = list(Path(tmp_path).glob("*.pre-p1b.bak"))
        assert len(backups) == 1, "a v1 install ends with ONE backup, under the P1b name"
        assert not list(Path(tmp_path).glob("*.pre-v2.bak"))
        snapshot = sqlite3.connect(backups[0])
        try:
            backup_tables = {row[0] for row in snapshot.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        finally:
            snapshot.close()
        assert not V2_TABLES & backup_tables, "the renamed file is still the pre-ladder state"
    finally:
        await eng.dispose()


async def test_a_later_boot_never_re_asserts_the_pointer(v2_db):
    """F2: a restart must not undo a rollback (or any other pointer move).

    Task 6's rollback flips the pointer back to the pre-P1b generation. If the
    boot path re-activated ``generation_name(kind)`` — as it did while the data
    step ran on every boot — the rollback would survive exactly until the next
    restart, and the manifest would be re-mutated on every boot.
    """
    eng, _ = v2_db
    await database.bootstrap_sqlite()
    async with eng.begin() as conn:
        await conn.execute(text("UPDATE index_generations SET is_active = 0"))
        await conn.execute(text("UPDATE index_generations SET is_active = 1 WHERE generation = :old"),
                           {"old": COLLECTION_NAME})

    before = await _manifest_state(eng)
    assert [row for row in before if row[3]] == [("memory", COLLECTION_NAME, LEGACY_TOKEN, 1)]

    await database.bootstrap_sqlite()  # rollback, then a restart
    await database.bootstrap_sqlite()  # and another one

    assert await _manifest_state(eng) == before, "a restart never moves the pointer"


async def test_a_deleted_manifest_stays_deleted_on_a_later_boot(v2_db):
    """Carry item T1-M4, now stronger: a v3 boot writes NO manifest row at all."""
    eng, _ = v2_db
    await database.bootstrap_sqlite()
    async with eng.begin() as conn:
        await conn.execute(text("DELETE FROM index_generations"))
    await database.bootstrap_sqlite()  # a later boot, from an empty manifest
    await database.bootstrap_sqlite()

    assert await _manifest_state(eng) == []


async def test_an_unflipped_upgrade_reads_loud_not_empty(v2_db, tmp_path, monkeypatch):
    """F1: with the pointer on the OLD generation, a read must raise — not
    answer "no memories matched" from a generation the migration has not built.

    The read path's contract guard is the loud failure: same dim, different
    fingerprint (the P1b contract change), so the install refuses to serve.
    """
    from app.retrieval import vector_backend
    from app.retrieval.embedder import EmbeddingDimensionMismatch
    from app.retrieval.memory import vector_store

    monkeypatch.setattr(settings, "QDRANT_MODE", "local")
    monkeypatch.setattr(settings, "QDRANT_LOCAL_PATH", str(tmp_path / "qdrant"))
    await database.bootstrap_sqlite()

    try:
        with pytest.raises(EmbeddingDimensionMismatch, match="different embedding contract"):
            await vector_store.search_memories(
                [0.0] * int(current_fingerprint()["dim"]),
                user_id=str(uuid.uuid4()), top_k=5,
            )
    finally:
        await vector_backend.close_clients()


async def test_an_existing_p1b_backup_is_reused_never_overwritten(v2_db):
    eng, tmp_path = v2_db
    existing = tmp_path / "v2.sqlite.pre-p1b.bak"
    existing.write_bytes(b"pre-p1b snapshot from the interrupted run")
    before = existing.read_bytes()

    await database.bootstrap_sqlite()

    assert list(Path(tmp_path).glob("*.pre-p1b.bak")) == [existing]
    assert existing.read_bytes() == before
    async with eng.connect() as conn:
        version, _ = await _schema(conn)
    assert version == 3
