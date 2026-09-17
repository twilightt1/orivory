"""SQLite ladder v7 — the opt-in retention settings on ``users`` (P4b T6).

Rulings this module pins:

- Auto expiration is opt-in (spec §8.1), so the ladder may only leave every
  pre-existing user OFF: the ``retention_enabled`` ADD COLUMN carries the
  constant ``'0'`` default (that default IS the backfill), and
  ``retention_days`` is NULLABLE with no default — "no window chosen" is a real
  state, not 0.
- The ladder step is ``_upgrade_v6_to_v7`` — its OWN ``.pre-retention.bak``
  milestone backup (the P1b/P2/P4a/P4b files are never overwritten),
  once-on-transition, and the DDL is inspect-first so a crash between the DDL
  and the stamp resumes instead of failing on a duplicate column.
- The shape the ladder lands IS the shape ``create_all`` produces (``BOOLEAN``
  with ``DEFAULT '0'``, and ``INTEGER`` nullable): an upgraded file and a fresh
  one are indistinguishable (``PRAGMA table_info`` compared).
- Rollback: a pre-T6 binary's ladder tops out at v6 and REFUSES a v7 file
  ("unsupported SQLite schema version 7"), so the rollback re-stamps
  ``user_version = 6``; rolling forward re-runs the step, which must be
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
from app.models.user import User

USER_ID = "a" * 32
NEW_COLUMNS = ("retention_enabled", "retention_days")
BACKUP = "*.pre-retention.bak"


async def _engine(tmp_path: Path, name: str):
    eng = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / name}", poolclass=NullPool)
    event.listen(eng.sync_engine, "connect", database._configure_sqlite_connection)
    return eng


async def _schema(conn) -> tuple[int, set[str]]:
    version = int((await conn.execute(text("PRAGMA user_version"))).scalar_one())
    tables = {r[0] for r in (await conn.execute(
        text("SELECT name FROM sqlite_master WHERE type='table'"))).all()}
    return version, tables


async def _table_info(conn, table: str = "users") -> set[tuple]:
    rows = (await conn.execute(text(f"PRAGMA table_info({table})"))).all()
    return {tuple(row[1:]) for row in rows}


async def _user_row(conn, user_id: str) -> tuple:
    """The retention pair as stored for one user (``<user_id>`` only exists once)."""
    row = (await conn.execute(text(
        "SELECT retention_enabled, retention_days FROM users WHERE id = :id"),
        {"id": user_id})).one()
    return tuple(row)


async def _user(conn, user_id: str) -> None:
    await conn.execute(text(
        "INSERT INTO users (id, email, onboarding_done, is_verified, is_active, is_deleted)"
        " VALUES (:id, :email, 0, 1, 1, 0)"),
        {"id": user_id, "email": f"{user_id}@test.invalid"})


async def _strip_v7_columns(conn) -> None:
    """Downgrade a create_all schema to the v6 shape of the users table."""
    have = await conn.run_sync(
        lambda c: {col["name"] for col in sa_inspect(c).get_columns("users")})
    for column in NEW_COLUMNS:
        if column in have:
            await conn.execute(text(f"ALTER TABLE users DROP COLUMN {column}"))


@pytest_asyncio.fixture
async def v6_db(tmp_path, monkeypatch):
    """A real v6 install: the current schema minus the v7 columns, rows, stamped 6."""
    eng = await _engine(tmp_path, "v6.sqlite")
    monkeypatch.setattr(database, "engine", eng)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _strip_v7_columns(conn)
        await _user(conn, USER_ID)
        await conn.execute(text("PRAGMA user_version = 6"))
    yield eng, tmp_path
    await eng.dispose()


# ── the ladder: v6 -> v7 ────────────────────────────────────────────────────


async def test_v6_install_upgrades_to_v7_with_the_retention_backup_and_off_default(v6_db):
    eng, tmp_path = v6_db
    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version, _tables = await _schema(conn)
        info = await _table_info(conn)
        stored = await _user_row(conn, USER_ID)
        integrity = (await conn.execute(text("PRAGMA integrity_check"))).scalar_one()

    assert version == database.SQLITE_SCHEMA_VERSION == 7
    assert ("retention_enabled", "BOOLEAN", 1, "'0'", 0) in info, (
        "NOT NULL with the OFF default: the ADD COLUMN default IS the backfill (spec §8.1)")
    assert ("retention_days", "INTEGER", 0, None, 0) in info, (
        "nullable, no default: no window chosen is a real state, not 0")
    assert stored == (0, None), "no migration ever opts a user in"
    assert integrity == "ok"

    backups = list(Path(tmp_path).glob(BACKUP))
    assert len(backups) == 1, "exactly one pre-retention milestone backup"
    assert not list(Path(tmp_path).glob("*.pre-p4b.bak")), (
        "a v6 install never stood at the pre-suppression-columns state: no such snapshot")
    snapshot = sqlite3.connect(backups[0])
    try:
        backup_cols = {row[1] for row in snapshot.execute("PRAGMA table_info(users)")}
        backup_version = snapshot.execute("PRAGMA user_version").fetchone()[0]
        backup_users = snapshot.execute("SELECT id FROM users").fetchall()
    finally:
        snapshot.close()
    assert "retention_enabled" not in backup_cols and "retention_days" not in backup_cols, (
        "the backup is what a pre-T6 binary opens")
    assert backup_version == 6
    assert [row[0] for row in backup_users] == [USER_ID]


async def test_the_upgraded_users_table_matches_a_fresh_install(v6_db, tmp_path):
    """The ADD COLUMN must land the exact schema ``create_all`` produces."""
    eng, _ = v6_db
    await database.bootstrap_sqlite()
    async with eng.connect() as conn:
        upgraded = await _table_info(conn)
    assert any(row[0] == "retention_enabled" for row in upgraded), (
        "the v7 columns are part of the parity this pins")

    fresh = await _engine(tmp_path, "reference.sqlite")
    try:
        async with fresh.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with fresh.connect() as conn:
            assert upgraded == await _table_info(conn), (
                "ladder schema differs from the create_all schema")
    finally:
        await fresh.dispose()


async def test_rebooting_a_v7_install_is_a_no_op(v6_db):
    """Once-on-transition: a restart never re-adds, re-stamps or re-backups."""
    eng, tmp_path = v6_db
    await database.bootstrap_sqlite()
    async with eng.begin() as conn:
        await conn.execute(text(
            "UPDATE users SET retention_enabled = 1, retention_days = 30 WHERE id = :id"),
            {"id": USER_ID})
    backup = next(iter(tmp_path.glob(BACKUP)))
    backup.unlink()

    await database.bootstrap_sqlite()
    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version, _tables = await _schema(conn)
        stored = await _user_row(conn, USER_ID)
    assert version == 7
    assert stored == (1, 30), "a later boot never re-asserts a setting the user turned on"
    assert not list(tmp_path.glob(BACKUP)), "no transition, no backup"


async def test_a_crashed_v7_step_resumes_and_never_duplicates(v6_db):
    """A kill between the DDL and the stamp must resume, not collide.

    Two crash points: (1) the whole step ran but the stamp did not — re-entry
    must inspect, not re-ADD; (2) only ``retention_enabled`` landed — re-entry
    finishes the step.
    """
    eng, _ = v6_db
    async with eng.begin() as conn:
        await conn.run_sync(database._upgrade_v6_to_v7)  # the DDL landed, then: dead
        version, _ = await _schema(conn)
    assert version == 6, "fixture must model an interruption, not a finished upgrade"

    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version, _tables = await _schema(conn)
        info = await _table_info(conn)
        stored = await _user_row(conn, USER_ID)
    assert version == 7
    assert {row[0] for row in info} >= set(NEW_COLUMNS)
    assert stored == (0, None)

    async with eng.begin() as conn:  # killed one column in
        await conn.execute(text("ALTER TABLE users DROP COLUMN retention_days"))
        await conn.execute(text("PRAGMA user_version = 6"))

    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version, _tables = await _schema(conn)
        info = await _table_info(conn)
    assert version == 7
    assert {row[0] for row in info} >= set(NEW_COLUMNS), "the step resumed and finished"


async def test_fresh_install_runs_the_v7_step_without_a_backup(tmp_path, monkeypatch):
    """A fresh install has nothing to back up — and still gets both columns."""
    eng = await _engine(tmp_path, "fresh.sqlite")
    monkeypatch.setattr(database, "engine", eng)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    try:
        await database.bootstrap_sqlite()

        async with eng.begin() as conn:
            await _user(conn, USER_ID)
        async with eng.connect() as conn:
            version, _tables = await _schema(conn)
            info = await _table_info(conn)
            stored = await _user_row(conn, USER_ID)
        assert version == 7
        assert ("retention_enabled", "BOOLEAN", 1, "'0'", 0) in info
        assert ("retention_days", "INTEGER", 0, None, 0) in info
        assert stored == (0, None), "an insert that omits them gets the OFF default"
        assert not list(Path(tmp_path).glob(BACKUP)), "nothing to back up yet"
        assert not list(Path(tmp_path).glob("*.pre-p4b.bak"))
        assert not list(Path(tmp_path).glob("*.pre-p4.bak"))
    finally:
        await eng.dispose()


async def test_rollback_restamps_v6_and_roll_forward_re_runs_the_step(v6_db):
    """The published rollback: re-stamp ``user_version = 6`` (a pre-T6 binary's
    ladder tops out there and refuses a v7 file), then roll forward.

    The step must be additive in both directions: rolling back the columns
    re-creates them (OFF, no invented values); rolling back the STAMP ALONE
    must not collide on an existing column. Either way the operator's
    pre-retention backup is reused, never rewritten.
    """
    eng, tmp_path = v6_db
    await database.bootstrap_sqlite()
    backup = next(iter(tmp_path.glob(BACKUP)))
    before = backup.read_bytes()

    async with eng.begin() as conn:  # the full pre-T6 shape, then the re-stamp
        await conn.execute(text("ALTER TABLE users DROP COLUMN retention_days"))
        await conn.execute(text("ALTER TABLE users DROP COLUMN retention_enabled"))
        await conn.execute(text("PRAGMA user_version = 6"))

    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version, _tables = await _schema(conn)
        stored = await _user_row(conn, USER_ID)
    assert version == 7
    assert stored == (0, None), "a rolled-back user is OFF, not invented ON"
    assert backup.read_bytes() == before, "an existing backup is never overwritten"

    async with eng.begin() as conn:  # the stamp alone: the columns are already there
        await conn.execute(text("PRAGMA user_version = 6"))

    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version, _tables = await _schema(conn)
        stored = await _user_row(conn, USER_ID)
    assert version == 7
    assert stored == (0, None)
    assert backup.read_bytes() == before


# ── the model carries the same shape (spec §8.1) ────────────────────────────


def test_the_retention_columns_carry_the_off_default_and_a_nullable_window():
    """A drift here is either a silent opt-in or a window nobody chose."""
    enabled = User.__table__.c.retention_enabled
    assert enabled.server_default is not None, (
        "the ladder mirrors this default: without it an upgraded file and a fresh one diverge")
    assert str(enabled.server_default.arg) == "0", "OFF is the only backfill a migration may write"
    assert enabled.default is not None and enabled.default.arg is False

    days = User.__table__.c.retention_days
    assert days.nullable is True, "no window chosen must stay representable"
    assert days.server_default is None and days.default is None, (
        "a default window is a decision the user never made")
    assert str(enabled.type) == "BOOLEAN" and str(days.type) == "INTEGER"
