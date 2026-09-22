"""SQLite ladder v6 — ``memory_suppressions.namespace`` + ``content_hash`` (P4b T3).

Rulings this module pins:

- The suppression ledger gains the namespace it pinned and (later, at upload
  time) the content hash of the forgotten source — both NULLABLE and never
  backfilled (R38): a pre-P4b suppression row keeps ``NULL`` for both, which is
  the honest "unknown", not a value the ladder invented.
- The ladder step is ``_upgrade_v5_to_v6`` — its OWN ``.pre-p4b.bak`` milestone
  backup (the P1b/P2/P4a files are never overwritten), once-on-transition, and
  the DDL is inspect-first so a crash between the DDL and the stamp resumes
  instead of failing on a duplicate column.
- The shape the ladder lands IS the shape ``create_all`` produces (``String(32)``
  / ``String(64)``, both nullable): an upgraded file and a fresh one are
  indistinguishable (``PRAGMA table_info`` compared).
- Rollback: a pre-P4b binary's ladder tops out at v5 and REFUSES a v6 file
  ("unsupported SQLite schema version 6"), so the rollback re-stamps
  ``user_version = 5``; rolling forward re-runs the step, which must be
  additive and must reuse the existing backup rather than clobber it.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest_asyncio
from sqlalchemy import event, text
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app import database, models  # noqa: F401 — register every model on Base
from app.database import Base
from app.models.memory import MemorySuppression

TENANT_A = "a" * 32
SUPPRESSION_ID = "f" * 32
SOURCE_REF = "doc-forgotten"
NEW_COLUMNS = ("namespace", "content_hash")
BACKUP = "*.pre-p4b.bak"


async def _engine(tmp_path: Path, name: str):
    eng = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / name}", poolclass=NullPool)
    event.listen(eng.sync_engine, "connect", database._configure_sqlite_connection)
    return eng


async def _schema(conn) -> tuple[int, set[str]]:
    version = int((await conn.execute(text("PRAGMA user_version"))).scalar_one())
    tables = {r[0] for r in (await conn.execute(
        text("SELECT name FROM sqlite_master WHERE type='table'"))).all()}
    return version, tables


async def _table_info(conn, table: str) -> set[tuple]:
    rows = (await conn.execute(text(f"PRAGMA table_info({table})"))).all()
    return {tuple(row[1:]) for row in rows}


async def _suppression_rows(conn) -> list[tuple]:
    return list((await conn.execute(text(
        "SELECT id, user_id, source_ref, reason, namespace, content_hash"
        " FROM memory_suppressions ORDER BY id"))).all())


async def _user(conn, user_id: str) -> None:
    await conn.execute(text(
        "INSERT INTO users (id, email, onboarding_done, is_verified, is_active, is_deleted)"
        " VALUES (:id, :email, 0, 1, 1, 0)"),
        {"id": user_id, "email": f"{user_id}@test.invalid"})


async def _suppression(conn, suppression_id: str, user_id: str) -> None:
    """A pre-P4b suppression row: no namespace, no content hash."""
    await conn.execute(text(
        "INSERT INTO memory_suppressions (id, user_id, source_ref, reason)"
        " VALUES (:id, :uid, :ref, 'forgotten')"),
        {"id": suppression_id, "uid": user_id, "ref": SOURCE_REF})


async def _strip_v6_columns(conn) -> None:
    """Downgrade a create_all schema to the v5 shape of the ledger table."""
    have = await conn.run_sync(
        lambda c: {col["name"] for col in sa_inspect(c).get_columns("memory_suppressions")})
    for column in NEW_COLUMNS:
        if column in have:
            await conn.execute(text(f"ALTER TABLE memory_suppressions DROP COLUMN {column}"))


async def _strip_v7_columns(conn) -> None:
    """Downgrade a create_all ``users`` table to the v6 shape (no retention).

    Without this the fixture's create_all leaves the v7 pair in place, the
    v6→v7 DDL becomes a no-op the test never notices, and the "v5 install"
    would silently be a v7 schema stamped 5 (OCR fidelity fix).
    """
    have = await conn.run_sync(
        lambda c: {col["name"] for col in sa_inspect(c).get_columns("users")})
    for column in ("retention_enabled", "retention_days"):
        if column in have:
            await conn.execute(text(f"ALTER TABLE users DROP COLUMN {column}"))


@pytest_asyncio.fixture
async def v5_db(tmp_path, monkeypatch):
    """A real v5 install: the current schema minus the v6 + v7 columns, rows, stamped 5."""
    eng = await _engine(tmp_path, "v5.sqlite")
    monkeypatch.setattr(database, "engine", eng)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _strip_v6_columns(conn)
        await _strip_v7_columns(conn)
        await _user(conn, TENANT_A)
        await _suppression(conn, SUPPRESSION_ID, TENANT_A)
        await conn.execute(text("PRAGMA user_version = 5"))
    yield eng, tmp_path
    await eng.dispose()


async def test_a_pinned_v5_binary_never_runs_a_newer_step(tmp_path, monkeypatch):
    """OCR fix (P4b review): every ladder step gated on the STARTING version
    only, so a pinned v5 binary (the P4a gate's stand-in for an older install)
    still ran v5→v6 and v6→v7 on boot — a v7 schema stamped 5. A step now also
    requires the target stamp (``SQLITE_SCHEMA_VERSION``)."""
    eng = await _engine(tmp_path, "pin5.sqlite")
    monkeypatch.setattr(database, "engine", eng)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    monkeypatch.setattr(database, "SQLITE_SCHEMA_VERSION", 5)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _strip_v6_columns(conn)
        await _strip_v7_columns(conn)
        await conn.execute(text("PRAGMA user_version = 5"))

    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version, _tables = await _schema(conn)
        ledger_cols = await conn.run_sync(
            lambda c: {col["name"] for col in sa_inspect(c).get_columns("memory_suppressions")})
        user_cols = await conn.run_sync(
            lambda c: {col["name"] for col in sa_inspect(c).get_columns("users")})
    assert version == 5, "a pinned binary stamps its own terminal version"
    assert "namespace" not in ledger_cols and "content_hash" not in ledger_cols, (
        "a v5 binary must not run the v6 DDL")
    assert "retention_enabled" not in user_cols, "nor the v7 DDL"


# ── the ladder: v5 -> v6 ────────────────────────────────────────────────────


async def test_v5_install_upgrades_to_v6_with_the_p4b_backup_and_null_columns(v5_db):
    eng, tmp_path = v5_db
    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version, _tables = await _schema(conn)
        info = await _table_info(conn, "memory_suppressions")
        rows = await _suppression_rows(conn)
        integrity = (await conn.execute(text("PRAGMA integrity_check"))).scalar_one()

    assert version == database.SQLITE_SCHEMA_VERSION == 8
    assert ("namespace", "VARCHAR(32)", 0, None, 0) in info, (
        "nullable, no default: the ladder never invents a value (R38)")
    assert ("content_hash", "VARCHAR(64)", 0, None, 0) in info
    assert rows == [(SUPPRESSION_ID, TENANT_A, SOURCE_REF, "forgotten", None, None)], (
        "the pre-existing row keeps NULL for both: no backfill")
    assert integrity == "ok"

    backups = list(Path(tmp_path).glob(BACKUP))
    assert len(backups) == 1, "exactly one pre-P4b milestone backup"
    assert not list(Path(tmp_path).glob("*.pre-p4.bak")), (
        "a v5 install never stood at the pre-P4a state: no such snapshot exists")
    snapshot = sqlite3.connect(backups[0])
    try:
        backup_cols = {row[1] for row in snapshot.execute(
            "PRAGMA table_info(memory_suppressions)")}
        backup_version = snapshot.execute("PRAGMA user_version").fetchone()[0]
        backup_rows = snapshot.execute(
            "SELECT source_ref FROM memory_suppressions").fetchall()
    finally:
        snapshot.close()
    assert "namespace" not in backup_cols and "content_hash" not in backup_cols, (
        "the backup is what a pre-P4b binary opens")
    assert backup_version == 5
    assert [row[0] for row in backup_rows] == [SOURCE_REF]


async def test_the_upgraded_ledger_matches_a_fresh_install(v5_db, tmp_path):
    """The ADD COLUMN must land the exact schema ``create_all`` produces."""
    eng, _ = v5_db
    await database.bootstrap_sqlite()
    async with eng.connect() as conn:
        upgraded = await _table_info(conn, "memory_suppressions")
    assert any(row[0] == "namespace" for row in upgraded), (
        "the v6 columns are part of the parity this pins")

    fresh = await _engine(tmp_path, "reference.sqlite")
    try:
        async with fresh.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with fresh.connect() as conn:
            assert upgraded == await _table_info(conn, "memory_suppressions"), (
                "ladder schema differs from the create_all schema")
    finally:
        await fresh.dispose()


async def test_rebooting_a_v6_install_is_a_no_op(v5_db):
    """Once-on-transition: a restart never re-adds, re-stamps or re-backups."""
    eng, tmp_path = v5_db
    await database.bootstrap_sqlite()
    async with eng.begin() as conn:
        await conn.execute(text(
            "UPDATE memory_suppressions SET namespace = 'moved' WHERE id = :id"),
            {"id": SUPPRESSION_ID})
    backup = next(iter(tmp_path.glob(BACKUP)))
    backup.unlink()

    await database.bootstrap_sqlite()
    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version, _tables = await _schema(conn)
        moved = (await conn.execute(text(
            "SELECT namespace FROM memory_suppressions WHERE id = :id"), {"id": SUPPRESSION_ID})).scalar_one()
    assert version == 8
    assert moved == "moved", "a later boot never re-asserts a value"
    assert not list(tmp_path.glob(BACKUP)), "no transition, no backup"


async def test_a_crashed_v6_step_resumes_and_never_duplicates(v5_db):
    """A kill between the DDL and the stamp must resume, not collide.

    Two crash points: (1) the whole step ran but the stamp did not — re-entry
    must inspect, not re-ADD; (2) only ``namespace`` landed — re-entry finishes
    the step.
    """
    eng, _ = v5_db
    async with eng.begin() as conn:
        await conn.run_sync(database._upgrade_v5_to_v6)  # the DDL landed, then: dead
        version, _ = await _schema(conn)
    assert version == 5, "fixture must model an interruption, not a finished upgrade"

    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version, _tables = await _schema(conn)
        info = await _table_info(conn, "memory_suppressions")
        rows = await _suppression_rows(conn)
    assert version == 8
    assert {row[0] for row in info} >= set(NEW_COLUMNS)
    assert rows == [(SUPPRESSION_ID, TENANT_A, SOURCE_REF, "forgotten", None, None)]

    async with eng.begin() as conn:  # killed one column in
        await conn.execute(text("ALTER TABLE memory_suppressions DROP COLUMN content_hash"))
        await conn.execute(text("PRAGMA user_version = 5"))

    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version, _tables = await _schema(conn)
        info = await _table_info(conn, "memory_suppressions")
    assert version == 8
    assert {row[0] for row in info} >= set(NEW_COLUMNS), "the step resumed and finished"


async def test_fresh_install_runs_the_v6_step_without_a_backup(tmp_path, monkeypatch):
    """A fresh install has nothing to back up — and still gets both columns."""
    eng = await _engine(tmp_path, "fresh.sqlite")
    monkeypatch.setattr(database, "engine", eng)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    try:
        await database.bootstrap_sqlite()

        async with eng.begin() as conn:
            await _user(conn, TENANT_A)
            await _suppression(conn, SUPPRESSION_ID, TENANT_A)
        async with eng.connect() as conn:
            version, _tables = await _schema(conn)
            info = await _table_info(conn, "memory_suppressions")
            stored = (await conn.execute(text(
                "SELECT namespace, content_hash FROM memory_suppressions WHERE id = :id"),
                {"id": SUPPRESSION_ID})).one()
        assert version == 8
        assert ("namespace", "VARCHAR(32)", 0, None, 0) in info
        assert ("content_hash", "VARCHAR(64)", 0, None, 0) in info
        assert tuple(stored) == (None, None), "an insert that omits them gets NULL"
        assert not list(Path(tmp_path).glob(BACKUP)), "nothing to back up yet"
        assert not list(Path(tmp_path).glob("*.pre-p4.bak"))
        assert not list(Path(tmp_path).glob("*.pre-p2.bak"))
    finally:
        await eng.dispose()


async def test_rollback_restamps_v5_and_roll_forward_re_runs_the_step(v5_db):
    """The published rollback: re-stamp ``user_version = 5`` (a pre-P4b binary's
    ladder tops out there and refuses a v6 file), then roll forward.

    The step must be additive in both directions: rolling back the columns
    re-creates them (NULL, no invented values); rolling back the STAMP ALONE
    must not collide on an existing column. Either way the operator's pre-P4b
    backup is reused, never rewritten.
    """
    eng, tmp_path = v5_db
    await database.bootstrap_sqlite()
    backup = next(iter(tmp_path.glob(BACKUP)))
    before = backup.read_bytes()

    async with eng.begin() as conn:  # the full pre-P4b shape, then the re-stamp
        await conn.execute(text("ALTER TABLE memory_suppressions DROP COLUMN content_hash"))
        await conn.execute(text("ALTER TABLE memory_suppressions DROP COLUMN namespace"))
        await conn.execute(text("PRAGMA user_version = 5"))

    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version, _tables = await _schema(conn)
        rows = await _suppression_rows(conn)
    assert version == 8
    assert rows == [(SUPPRESSION_ID, TENANT_A, SOURCE_REF, "forgotten", None, None)]
    assert backup.read_bytes() == before, "an existing backup is never overwritten"

    async with eng.begin() as conn:  # the stamp alone: the columns are already there
        await conn.execute(text("PRAGMA user_version = 5"))

    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version, _tables = await _schema(conn)
        rows = await _suppression_rows(conn)
    assert version == 8
    assert rows == [(SUPPRESSION_ID, TENANT_A, SOURCE_REF, "forgotten", None, None)]
    assert backup.read_bytes() == before


# ── the model carries the same shape (R38) ──────────────────────────────────


def test_the_ledger_columns_are_nullable_and_never_defaulted():
    """A drift here is a backfilled value the ledger never observed."""
    for name in NEW_COLUMNS:
        column = MemorySuppression.__table__.c[name]
        assert column.nullable is True, f"{name} must stay nullable (no backfill)"
        assert column.server_default is None and column.default is None, (
            f"{name} must have no default: NULL is the honest value")
    assert str(MemorySuppression.__table__.c.content_hash.type) == "VARCHAR(64)"
