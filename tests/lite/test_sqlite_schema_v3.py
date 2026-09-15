"""SQLite ladder v3 — the P1b generation rows are a DATA step, not DDL (R24).

A v2 install (the schema P1a produced) boots and must end with exactly the TWO
real generation rows (memory + chunk, named by ``generation_name`` and stamped
with the current contract token), the P1a transitional row retired, and the
pre-P1b backup named ``.pre-p1b.bak``. A fresh install gets the same rows with
no backup; a v1 install upgraded straight through keeps exactly one backup.

The carry item this pins: the old ``_seed_transitional_generation`` re-created
the ``Orivory_memories`` row on EVERY boot. The ladder now writes the real rows
once (idempotently) and nothing resurrects the old spelling.
"""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import pytest_asyncio
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app import database, models  # noqa: F401 — register every model on Base
from app.database import Base
from app.retrieval.embedding_fingerprint import (
    current_fingerprint,
    fingerprint_generation,
    generation_name,
)
from app.retrieval.memory.vector_store import COLLECTION_NAME
from tests.lite.test_sqlite_schema_v2 import V2_TABLES, _seed_v1_rows, _strip_v2_objects

TOKEN = fingerprint_generation(current_fingerprint())


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


async def _insert_transitional(conn) -> None:
    """The P1a seed: one row naming the pre-P1b collection spelling."""
    await conn.execute(text(
        "INSERT INTO index_generations (id, kind, generation, fingerprint, is_active, created_at)"
        " VALUES (:id, 'memory', :gen, :fp, 1, CURRENT_TIMESTAMP)"),
        {"id": uuid.uuid4().hex, "gen": COLLECTION_NAME, "fp": TOKEN})


@pytest_asyncio.fixture
async def v2_db(tmp_path, monkeypatch):
    """A v2 install: current schema, the P1a transitional row, ``user_version`` 2."""
    eng = await _engine(tmp_path, "v2.sqlite")
    monkeypatch.setattr(database, "engine", eng)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _insert_transitional(conn)
        await conn.execute(text("PRAGMA user_version = 2"))
    yield eng, tmp_path
    await eng.dispose()


async def test_v2_install_gets_the_two_real_rows_and_a_p1b_backup(v2_db):
    eng, tmp_path = v2_db
    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version, _ = await _schema(conn)
        rows = await _generations(conn)

    assert version == database.SQLITE_SCHEMA_VERSION == 3
    active = {(kind, generation) for kind, generation, _, is_active in rows if is_active}
    assert active == {("memory", generation_name("memory")), ("chunk", generation_name("chunk"))}
    assert all(fp == TOKEN for _, _, fp, _ in rows)
    transitional = [row for row in rows if row[1] == COLLECTION_NAME]
    assert transitional and transitional[0][3] == 0, "the old spelling is retired, never active"

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
        assert all(row[3] == 1 for row in rows)
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


async def test_nothing_resurrects_the_transitional_row_on_a_later_boot(v2_db):
    """Carry item T1-M4: the seed must not re-create ``Orivory_memories``."""
    eng, _ = v2_db
    await database.bootstrap_sqlite()
    async with eng.begin() as conn:
        await conn.execute(text("DELETE FROM index_generations"))
    await database.bootstrap_sqlite()  # a later boot, from an empty manifest
    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        rows = await _generations(conn)

    assert {(row[0], row[1]) for row in rows} == {
        ("memory", generation_name("memory")), ("chunk", generation_name("chunk"))}
    assert sum(row[3] for row in rows) == 2, "exactly one active row per kind"


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


async def test_the_ladder_repairs_a_missing_active_row(v2_db):
    """Idempotent by construction: a wiped manifest is re-created, not fatal."""
    eng, _ = v2_db
    await database.bootstrap_sqlite()
    async with eng.begin() as conn:
        await conn.execute(text("DELETE FROM index_generations WHERE kind = 'chunk'"))
    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        rows = await _generations(conn)
    assert [row for row in rows if row[0] == "chunk" and row[3] == 1]
