"""SQLite ladder v5 — ``memories.namespace``, personal-only (P4a Task 1).

Rulings this module pins:

- Namespace is a REAL column, and it is NOT derived from anything the client
  sends (plan §Global Constraints). In this phase the only value in existence is
  ``personal``, so the v4 -> v5 step BACKFILLS every pre-existing row with the
  column default. A deployment with one namespace must answer exactly as it did
  before the column existed — the predicate it enables (Task 2+) matches every
  row.
- The ladder step is ``_upgrade_v4_to_v5`` — its OWN ``.pre-p4.bak`` milestone
  backup (the P1b/P2 files are never overwritten), once-on-transition (a restart
  never re-asserts or repairs), fresh installs run it too with nothing to back
  up, and the DDL is inspect-first so a crash between the DDL and the stamp
  resumes instead of failing on a duplicate column.
- The shape the ladder lands IS the shape ``create_all`` produces (plan
  §Interfaces: ``String(32)``, NOT NULL, ``server_default 'personal'``, index on
  ``(namespace, user_id)``): the ADD COLUMN is written to match the model column
  byte for byte, so a v4 install upgraded in place is indistinguishable from a
  fresh v5 install.
- Rollback: a pre-P4 binary's ladder tops out at v4 and REFUSES a v5 file
  ("unsupported SQLite schema version 5"), so the rollback re-stamps
  ``user_version = 4``; rolling forward re-runs the step, which must be additive
  (the column is inspected, never re-added) and must reuse the existing backup
  rather than clobber the operator's snapshot.
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
from app.models.memory import Memory
from app.retrieval.memory import namespaces

TENANT_A = "a" * 32
TENANT_B = "b" * 32
MEM_A = "1" * 32
MEM_B = "2" * 32
MEM_C = "3" * 32
INDEX_NAME = "ix_memories_namespace_user"
BACKUP = "*.pre-p4.bak"


async def _engine(tmp_path: Path, name: str):
    eng = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / name}", poolclass=NullPool)
    event.listen(eng.sync_engine, "connect", database._configure_sqlite_connection)
    return eng


async def _schema(conn) -> tuple[int, set[str]]:
    version = int((await conn.execute(text("PRAGMA user_version"))).scalar_one())
    tables = {r[0] for r in (await conn.execute(
        text("SELECT name FROM sqlite_master WHERE type='table'"))).all()}
    return version, tables


async def _columns(conn) -> set[str]:
    return {col["name"] for col in await conn.run_sync(
        lambda c: sa_inspect(c).get_columns("memories"))}


async def _table_info(conn, table: str = "memories") -> set[tuple]:
    rows = (await conn.execute(text(f"PRAGMA table_info({table})"))).all()
    return {tuple(row[1:]) for row in rows}


async def _index_sql(conn) -> str | None:
    """The ``sqlite_master`` DDL of the namespace index, or None when absent."""
    row = (await conn.execute(text(
        "SELECT sql FROM sqlite_master WHERE type='index' AND name=:name"),
        {"name": INDEX_NAME})).first()
    return row[0] if row else None


async def _namespaces(conn) -> list[str]:
    return [r[0] for r in (await conn.execute(
        text("SELECT namespace FROM memories ORDER BY id"))).all()]


async def _contents(conn) -> list[str]:
    return [r[0] for r in (await conn.execute(
        text("SELECT content FROM memories ORDER BY id"))).all()]


async def _user(conn, user_id: str) -> None:
    await conn.execute(text(
        "INSERT INTO users (id, email, onboarding_done, is_verified, is_active, is_deleted)"
        " VALUES (:id, :email, 0, 1, 1, 0)"),
        {"id": user_id, "email": f"{user_id}@test.invalid"})


async def _memory(conn, memory_id: str, user_id: str, content: str) -> None:
    await conn.execute(text(
        "INSERT INTO memories (id, user_id, content, tags, pinned, is_shared, recall_count)"
        " VALUES (:id, :uid, :content, '[]', 0, 0, 0)"),
        {"id": memory_id, "uid": user_id, "content": content})


async def _strip_namespace(conn) -> None:
    """Downgrade a create_all schema to the v4 shape (pre-namespace column).

    A v4 install predates the column AND its index, and SQLite refuses to drop an
    indexed column — hence the index first. No-op when the model has no namespace
    column at all, so this fixture still builds on the RED run.
    """
    have = await conn.run_sync(lambda c: {col["name"] for col in sa_inspect(c).get_columns("memories")})
    if "namespace" not in have:
        return
    await conn.execute(text(f"DROP INDEX IF EXISTS {INDEX_NAME}"))
    await conn.execute(text("ALTER TABLE memories DROP COLUMN namespace"))


@pytest_asyncio.fixture
async def v4_db(tmp_path, monkeypatch):
    """A real v4 install: the current schema minus the v5 column, rows, stamped 4.

    v5 adds no table and no other column, so "create_all then drop the namespace
    column" IS the v4 shape this ladder upgrades — a real database file, not a
    mock.
    """
    eng = await _engine(tmp_path, "v4.sqlite")
    monkeypatch.setattr(database, "engine", eng)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _strip_namespace(conn)
        await _user(conn, TENANT_A)
        await _user(conn, TENANT_B)
        await _memory(conn, MEM_A, TENANT_A, "alpha")
        await _memory(conn, MEM_B, TENANT_A, "beta")
        await _memory(conn, MEM_C, TENANT_B, "gamma")
        await conn.execute(text("PRAGMA user_version = 4"))
    yield eng, tmp_path
    await eng.dispose()


# ── the ladder: v4 -> v5 ────────────────────────────────────────────────────


async def test_v4_install_upgrades_to_v5_with_the_p4_backup_and_personal_backfill(v4_db):
    eng, tmp_path = v4_db
    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version, _tables = await _schema(conn)
        columns = await _columns(conn)
        info = await _table_info(conn)
        index = await _index_sql(conn)
        namespaces = await _namespaces(conn)
        contents = await _contents(conn)
        integrity = (await conn.execute(text("PRAGMA integrity_check"))).scalar_one()

    assert version == database.SQLITE_SCHEMA_VERSION == 6
    assert "namespace" in columns, "the v5 step adds the column"
    assert ("namespace", "VARCHAR(32)", 1, "'personal'", 0) in info, (
        "NOT NULL with the 'personal' default: that default IS the backfill")
    assert index is not None, "the (namespace, user_id) index is part of the step"
    assert "namespace" in index and "user_id" in index and INDEX_NAME in index
    assert namespaces == ["personal"] * 3, "every pre-existing row is personal"
    assert contents == ["alpha", "beta", "gamma"], "the backfill is additive, never a rewrite"
    assert integrity == "ok"

    backups = list(Path(tmp_path).glob(BACKUP))
    assert len(backups) == 1, "exactly one pre-P4 milestone backup"
    assert not list(Path(tmp_path).glob("*.pre-p2.bak")), (
        "a v4 install never stood at the pre-P2 state: no such snapshot exists")
    assert not list(Path(tmp_path).glob("*.pre-v2.bak"))
    snapshot = sqlite3.connect(backups[0])
    try:
        backup_info = {tuple(row[1:]) for row in snapshot.execute("PRAGMA table_info(memories)")}
        backup_version = snapshot.execute("PRAGMA user_version").fetchone()[0]
        backup_rows = snapshot.execute("SELECT content FROM memories ORDER BY id").fetchall()
        backup_integrity = snapshot.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        snapshot.close()
    assert not {row for row in backup_info if row[0] == "namespace"}, (
        "the backup is what a pre-P4 binary opens: no namespace column")
    assert backup_version == 4
    assert [row[0] for row in backup_rows] == ["alpha", "beta", "gamma"]
    assert backup_integrity == "ok"


async def test_the_upgraded_memories_table_matches_a_fresh_install(v4_db, tmp_path):
    """The ADD COLUMN must land the exact schema ``create_all`` produces.

    A ladder that adds ``TEXT`` where the model declares ``String(32)`` (or a
    different default spelling) would leave two schemas in the field that the
    model claims are one — the drift ``tests/migrations`` catches on Postgres.
    """
    eng, _ = v4_db
    await database.bootstrap_sqlite()
    async with eng.connect() as conn:
        upgraded = await _table_info(conn)
    assert any(row[0] == "namespace" for row in upgraded), (
        "the v5 column is part of the parity this pins")

    fresh = await _engine(tmp_path, "reference.sqlite")
    try:
        async with fresh.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with fresh.connect() as conn:
            assert upgraded == await _table_info(conn), (
                "ladder schema differs from the create_all schema")
    finally:
        await fresh.dispose()


async def test_rebooting_a_v5_install_is_a_no_op(v4_db):
    """Once-on-transition: a restart never re-adds, re-stamps or re-backups.

    The observable pins: a namespace an operator moved by hand stays put, a
    deleted milestone backup is NOT taken again (nothing re-runs), and the index
    is left exactly as the operator left it — the v5 step is a transition, not a
    repair loop.
    """
    eng, tmp_path = v4_db
    await database.bootstrap_sqlite()
    async with eng.begin() as conn:
        await conn.execute(text("UPDATE memories SET namespace = 'moved' WHERE id = :id"),
                           {"id": MEM_A})
    backup = next(iter(tmp_path.glob(BACKUP)))
    backup.unlink()

    await database.bootstrap_sqlite()
    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version, _tables = await _schema(conn)
        moved = (await conn.execute(text(
            "SELECT namespace FROM memories WHERE id = :id"), {"id": MEM_A})).scalar_one()
    assert version == 6
    assert moved == "moved", "a later boot never re-asserts a value"
    assert not list(tmp_path.glob(BACKUP)), "no transition, no backup"


async def test_a_later_boot_never_re_asserts_the_index(v4_db):
    """The v4 step's rule, one version on: a restart does not rebuild DDL."""
    eng, tmp_path = v4_db
    await database.bootstrap_sqlite()
    async with eng.begin() as conn:
        await conn.execute(text(f"DROP INDEX {INDEX_NAME}"))  # drift, deliberately

    await database.bootstrap_sqlite()
    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version, _tables = await _schema(conn)
        assert await _index_sql(conn) is None, "a v5 boot does not re-create the index"
    assert version == 6
    assert list(tmp_path.glob(BACKUP)), "and it takes no second backup"


async def test_a_crashed_v5_step_resumes_and_never_duplicates(v4_db):
    """A kill between the DDL and the stamp must resume, not collide.

    Two crash points, both real: (1) the whole step ran but the version stamp did
    not (``user_version`` still 4) — re-entry must inspect, not re-ADD; (2) the
    column landed but the index did not — re-entry finishes the step. SQLite has
    no ``ADD COLUMN IF NOT EXISTS``, so only the inspect-first guard makes this
    work.
    """
    eng, _ = v4_db
    async with eng.begin() as conn:
        await conn.run_sync(database._upgrade_v4_to_v5)  # the DDL landed, then: dead
        version, _ = await _schema(conn)
    assert version == 4, "fixture must model an interruption, not a finished upgrade"

    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version, _tables = await _schema(conn)
        assert await _namespaces(conn) == ["personal"] * 3
        assert await _index_sql(conn) is not None
        assert len(await _columns(conn)) > 0

    async with eng.begin() as conn:  # the same, one DDL further out: killed before the index
        await conn.execute(text(f"DROP INDEX {INDEX_NAME}"))
        await conn.execute(text("PRAGMA user_version = 4"))

    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version, _tables = await _schema(conn)
        assert await _index_sql(conn) is not None, "the step resumed and finished"
        assert await _namespaces(conn) == ["personal"] * 3, "and never duplicated a column"
    assert version == 6


async def test_fresh_install_runs_the_v5_step_without_a_backup(tmp_path, monkeypatch):
    """A fresh install has nothing to back up — and still gets the column+index."""
    eng = await _engine(tmp_path, "fresh.sqlite")
    monkeypatch.setattr(database, "engine", eng)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    try:
        await database.bootstrap_sqlite()

        async with eng.begin() as conn:
            await _user(conn, TENANT_A)
            await _memory(conn, MEM_A, TENANT_A, "fresh row")
        async with eng.connect() as conn:
            version, _tables = await _schema(conn)
            info = await _table_info(conn)
            assert await _index_sql(conn) is not None
            stored = (await conn.execute(text(
                "SELECT namespace FROM memories WHERE id = :id"), {"id": MEM_A})).scalar_one()
        assert version == 6
        assert ("namespace", "VARCHAR(32)", 1, "'personal'", 0) in info
        assert stored == "personal", "an insert that omits the column gets the default"
        assert not list(Path(tmp_path).glob("*.pre-p4.bak")), "nothing to back up yet"
        assert not list(Path(tmp_path).glob("*.pre-p2.bak"))
        assert not list(Path(tmp_path).glob("*.pre-p1b.bak"))
    finally:
        await eng.dispose()


async def test_rollback_restamps_v4_and_roll_forward_re_runs_the_step(v4_db):
    """The published rollback: re-stamp ``user_version = 4`` (a pre-P4 binary's
    ladder tops out there and refuses a v5 file), then roll forward.

    The step must be additive in both directions: rolling back a full snapshot
    (column + index dropped) re-creates them and backfills 'personal'; rolling
    back the STAMP ALONE must not collide on an existing column. Either way the
    operator's pre-P4 backup is reused, never rewritten.
    """
    eng, tmp_path = v4_db
    await database.bootstrap_sqlite()
    backup = next(iter(tmp_path.glob(BACKUP)))
    before = backup.read_bytes()

    async with eng.begin() as conn:  # the full pre-P4 shape, then the re-stamp
        await conn.execute(text(f"DROP INDEX {INDEX_NAME}"))
        await conn.execute(text("ALTER TABLE memories DROP COLUMN namespace"))
        await conn.execute(text("PRAGMA user_version = 4"))

    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version, _tables = await _schema(conn)
        assert await _namespaces(conn) == ["personal"] * 3
        assert await _index_sql(conn) is not None
    assert version == 6
    assert backup.read_bytes() == before, "an existing backup is never overwritten"

    async with eng.begin() as conn:  # the stamp alone: the column is already there
        await conn.execute(text("PRAGMA user_version = 4"))

    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version, _tables = await _schema(conn)
        assert await _namespaces(conn) == ["personal"] * 3
    assert version == 6
    assert backup.read_bytes() == before


# ── the value: one spelling, shared with the rest of P4a ─────────────────────


def test_personal_is_the_one_spelling():
    """The constant Task 2+ predicates use is the value the schema stores.

    A drift here is a silent deny-everything: a predicate spelled differently
    from the backfilled value matches no row.
    """
    assert namespaces.PERSONAL == "personal"
    column = Memory.__table__.c.namespace
    assert column.nullable is False
    assert column.server_default is not None and column.server_default.arg == namespaces.PERSONAL
