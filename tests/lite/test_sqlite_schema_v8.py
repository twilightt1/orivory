"""SQLite ladder v8 — the freshness barrier's count index on ``index_outbox``.

Rulings this module pins (the v6/v7 steps' shape, an index instead of columns):

- The step is ``_upgrade_v7_to_v8`` — its OWN ``.pre-freshness-index.bak``
  milestone backup (the P1b/P2/P4a/P4b/T6 files are never overwritten),
  once-on-transition, and the DDL is ``IF NOT EXISTS`` so a crash between the
  DDL and the stamp resumes instead of failing on a duplicate index.
- The index the ladder installs IS the one ``create_all`` produces (same name,
  same columns, same order, same uniqueness): an upgraded file and a fresh one
  are indistinguishable.
- The index is what the barrier's per-poll count needs — ``kind`` + ``tenant_id``
  filtered, grouped by ``status`` — so ``EXPLAIN QUERY PLAN`` must answer it
  from the index rather than scanning a table that is never pruned.
- Rollback: a pre-v8 binary's ladder tops out at v7 and REFUSES a v8 file
  ("unsupported SQLite schema version 8"), so the rollback re-stamps
  ``user_version = 7``; rolling forward re-runs the step, which must be
  additive and must reuse the existing backup rather than clobber it.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest_asyncio
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app import database, models  # noqa: F401 — register every model on Base
from app.database import Base
from app.models.index_outbox import IndexOutbox

INDEX_NAME = "ix_index_outbox_kind_tenant_status"
BACKUP = "*.pre-freshness-index.bak"
TENANT = "a" * 32


async def _engine(tmp_path: Path, name: str):
    eng = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / name}", poolclass=NullPool)
    event.listen(eng.sync_engine, "connect", database._configure_sqlite_connection)
    return eng


async def _schema(conn) -> int:
    return int((await conn.execute(text("PRAGMA user_version"))).scalar_one())


def _index_columns(db_path: str) -> list[str]:
    """The counted index's columns, read straight from the file (or [] absent)."""
    snapshot = sqlite3.connect(db_path)
    try:
        rows = snapshot.execute(f"PRAGMA index_info({INDEX_NAME})").fetchall()
    finally:
        snapshot.close()
    return [row[2] for row in rows]


def _index_unique(db_path: str) -> bool | None:
    snapshot = sqlite3.connect(db_path)
    try:
        rows = snapshot.execute("PRAGMA index_list(index_outbox)").fetchall()
    finally:
        snapshot.close()
    for row in rows:
        if row[1] == INDEX_NAME:
            return bool(row[2])
    return None


def _index_sql(db_path: str) -> str | None:
    snapshot = sqlite3.connect(db_path)
    try:
        row = snapshot.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name=?", (INDEX_NAME,)
        ).fetchone()
    finally:
        snapshot.close()
    return row[0] if row else None


def _seed_outbox(db_path: str, rows: int = 2000) -> None:
    """A standing outbox: the barrier's count reads it, so there is something to scan."""
    snapshot = sqlite3.connect(db_path)
    try:
        snapshot.executemany(
            "INSERT INTO index_outbox (kind, entity_id, tenant_id, revision, operation,"
            " target_generation, status, attempts, created_at, updated_at)"
            " VALUES ('memory', ?, ?, 1, 'upsert', 'gen', 'done', 0,"
            " '2026-01-01 00:00:00.000000', '2026-01-01 00:00:00.000000')",
            [(f"{i:032x}", TENANT) for i in range(rows)],
        )
        snapshot.commit()
    finally:
        snapshot.close()


@pytest_asyncio.fixture
async def v7_db(tmp_path, monkeypatch):
    """A real v7 install: the current schema minus the v8 index, stamped 7."""
    eng = await _engine(tmp_path, "v7.sqlite")
    monkeypatch.setattr(database, "engine", eng)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text(f"DROP INDEX IF EXISTS {INDEX_NAME}"))
        await conn.execute(text("PRAGMA user_version = 7"))
    assert _index_columns(str(tmp_path / "v7.sqlite")) == [], "fixture must model v7"
    yield eng, tmp_path
    await eng.dispose()


# ── the ladder: v7 -> v8 ────────────────────────────────────────────────────


async def test_v7_install_upgrades_to_v8_with_the_count_index_and_its_backup(v7_db):
    eng, tmp_path = v7_db
    await database.bootstrap_sqlite()
    db_path = str(tmp_path / "v7.sqlite")

    async with eng.connect() as conn:
        version = await _schema(conn)
        integrity = (await conn.execute(text("PRAGMA integrity_check"))).scalar_one()

    assert version == database.SQLITE_SCHEMA_VERSION == 8
    assert _index_columns(db_path) == ["kind", "tenant_id", "status"], (
        "the barrier's count needs this index")
    assert _index_unique(db_path) is False, "a lookup index, not a constraint"
    assert integrity == "ok"

    backups = list(Path(tmp_path).glob(BACKUP))
    assert len(backups) == 1, "exactly one pre-freshness-index milestone backup"
    assert not list(Path(tmp_path).glob("*.pre-retention.bak")), (
        "a v7 install never stood at the pre-retention state: no such snapshot")
    backup_version = sqlite3.connect(backups[0]).execute("PRAGMA user_version").fetchone()[0]
    assert backup_version == 7, "the backup is what a pre-v8 binary opens"
    assert _index_sql(str(backups[0])) is None, "…and it carries no v8 index"


async def test_the_upgraded_index_matches_a_fresh_install(v7_db, tmp_path):
    """Same name, same columns, same order as ``create_all`` installs."""
    _ = v7_db  # fixture side effect: the monkeypatched engine + a real v7 file
    await database.bootstrap_sqlite()
    upgraded = _index_columns(str(tmp_path / "v7.sqlite"))

    fresh = await _engine(tmp_path, "reference.sqlite")
    try:
        async with fresh.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        fresh_path = str(tmp_path / "reference.sqlite")
        assert upgraded == _index_columns(fresh_path) == ["kind", "tenant_id", "status"]
        assert _index_unique(fresh_path) == _index_unique(str(tmp_path / "v7.sqlite"))
    finally:
        await fresh.dispose()


async def test_rebooting_a_v8_install_is_a_no_op(v7_db):
    """Once-on-transition: a restart never re-creates the index or re-backups."""
    eng, tmp_path = v7_db
    await database.bootstrap_sqlite()
    backup = next(iter(tmp_path.glob(BACKUP)))
    backup.unlink()

    await database.bootstrap_sqlite()
    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version = await _schema(conn)
    assert version == 8
    assert _index_columns(str(tmp_path / "v7.sqlite")) == ["kind", "tenant_id", "status"]
    assert not list(tmp_path.glob(BACKUP)), "no transition, no backup"


async def test_a_crashed_v8_step_resumes_and_never_duplicates(v7_db):
    """A kill between the DDL and the stamp must resume, not collide."""
    eng, tmp_path = v7_db
    async with eng.begin() as conn:
        await conn.run_sync(database._upgrade_v7_to_v8)  # the DDL landed, then: dead
        version = await _schema(conn)
    assert version == 7, "fixture must model an interruption, not a finished upgrade"

    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version = await _schema(conn)
    assert version == 8
    assert _index_columns(str(tmp_path / "v7.sqlite")) == ["kind", "tenant_id", "status"]


async def test_fresh_install_runs_the_v8_step_without_a_backup(tmp_path, monkeypatch):
    """A fresh install has nothing to back up — and still gets the index."""
    eng = await _engine(tmp_path, "fresh.sqlite")
    monkeypatch.setattr(database, "engine", eng)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    try:
        await database.bootstrap_sqlite()
        db_path = str(tmp_path / "fresh.sqlite")

        async with eng.connect() as conn:
            version = await _schema(conn)
        assert version == 8
        assert _index_columns(db_path) == ["kind", "tenant_id", "status"]
        assert not list(Path(tmp_path).glob(BACKUP)), "nothing to back up yet"
    finally:
        await eng.dispose()


async def test_rollback_restamps_v7_and_roll_forward_re_runs_the_step(v7_db):
    """The published rollback: re-stamp ``user_version = 7`` (a pre-v8 binary's
    ladder tops out there and refuses a v8 file), then roll forward.

    The step must be additive in both directions: rolling back the INDEX
    re-creates it; rolling back the STAMP ALONE must not collide on the
    existing index. Either way the operator's pre-v8 backup is reused.
    """
    eng, tmp_path = v7_db
    db_path = str(tmp_path / "v7.sqlite")
    await database.bootstrap_sqlite()
    backup = next(iter(tmp_path.glob(BACKUP)))
    before = backup.read_bytes()

    async with eng.begin() as conn:  # the v7 shape, then the re-stamp
        await conn.execute(text(f"DROP INDEX IF EXISTS {INDEX_NAME}"))
        await conn.execute(text("PRAGMA user_version = 7"))

    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version = await _schema(conn)
    assert version == 8
    assert _index_columns(db_path) == ["kind", "tenant_id", "status"]
    assert backup.read_bytes() == before, "an existing backup is never overwritten"

    async with eng.begin() as conn:  # the stamp alone: the index is already there
        await conn.execute(text("PRAGMA user_version = 7"))

    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version = await _schema(conn)
    assert version == 8
    assert _index_columns(db_path) == ["kind", "tenant_id", "status"]
    assert backup.read_bytes() == before


async def test_a_pinned_v7_binary_never_runs_the_v8_step(tmp_path, monkeypatch):
    """A pre-v8 binary stamps its own terminal version and installs no v8 index."""
    eng = await _engine(tmp_path, "pin7.sqlite")
    monkeypatch.setattr(database, "engine", eng)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    monkeypatch.setattr(database, "SQLITE_SCHEMA_VERSION", 7)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text(f"DROP INDEX IF EXISTS {INDEX_NAME}"))
        await conn.execute(text("PRAGMA user_version = 7"))

    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version = await _schema(conn)
    assert version == 7, "a pinned binary stamps its own terminal version"
    assert _index_columns(str(tmp_path / "pin7.sqlite")) == [], (
        "a v7 binary must not run the v8 DDL")


# ── the point of the index: the barrier's count is served, not scanned ──────


async def test_the_count_query_is_served_by_the_index(v7_db):
    """``EXPLAIN QUERY PLAN``: the barrier's per-poll count is an index search.

    The outbox is never pruned, so a SCAN here is a cost that grows with the
    deployment's whole write history and is paid once per poll (finding 564).
    """
    eng, tmp_path = v7_db
    await database.bootstrap_sqlite()
    db_path = str(tmp_path / "v7.sqlite")
    _seed_outbox(db_path)

    plan_sql = (
        "EXPLAIN QUERY PLAN SELECT status, count(*) FROM index_outbox"
        f" WHERE tenant_id = '{TENANT}' AND kind = 'memory' GROUP BY status"
    )
    async with eng.connect() as conn:
        plan = [row[-1] for row in (await conn.execute(text(plan_sql))).all()]

    assert plan, "the planner must say something about the count"
    assert any(INDEX_NAME in line for line in plan), plan
    assert not any(line.strip().upper().startswith("SCAN") for line in plan), plan


def test_the_model_carries_the_index_the_ladder_installs():
    """A drift here is a CREATE_ALL install without the index the ladder adds."""
    indexes = {index.name: index for index in IndexOutbox.__table__.indexes}
    assert INDEX_NAME in indexes, "model metadata must declare the count index"
    assert [column.name for column in indexes[INDEX_NAME].columns] == [
        "kind", "tenant_id", "status"]
