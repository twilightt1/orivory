"""SQLite schema v2: revisions + outbox/generation tables via versioned ladder.

Real SQLite files, no mocks: a synthetic v1 install (full v1 schema,
``user_version`` 0 or 1) is migrated by ``bootstrap_sqlite`` and the result is
re-read from the file — schema, backfilled data, pre-DDL backup, generation
rows. The ladder's terminal stamp is v3 (the P1b data step, which writes the two
real generation rows INACTIVE on an upgrade — the old pointer keeps serving until
``migrate_qdrant.py cutover`` flips it — and renames the milestone backup; see
``test_sqlite_schema_v3.py`` for that step).
"""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import String, event, select, text
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app import database, models  # noqa: F401 — register every model on Base
from app.config import settings
from app.database import Base
from app.models.index_outbox import IndexGeneration
from app.retrieval.embedding_fingerprint import current_fingerprint, fingerprint_generation, generation_name
from app.retrieval.memory.vector_store import COLLECTION_NAME

V2_TABLES = {"index_outbox", "index_generations", "memory_suppressions"}
V2_COLUMNS = {"memories": "revision", "document_chunks": "revision"}


def _assert_fingerprint_fits_column(token: str) -> None:
    """SQLite ignores VARCHAR length; Postgres does not — the seed must fit."""
    column_type = IndexGeneration.__table__.c.fingerprint.type
    assert isinstance(column_type, String) and column_type.length is not None, (
        "fingerprint stays a length-bounded String (not Text)"
    )
    assert len(token) <= column_type.length, (
        f"fingerprint is {len(token)} chars, column allows {column_type.length}"
    )


async def _engine(tmp_path: Path, name: str):
    eng = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / name}", poolclass=NullPool)
    event.listen(eng.sync_engine, "connect", database._configure_sqlite_connection)
    return eng


def _columns_of(sync_conn, table: str) -> set[str]:
    return {c["name"] for c in sa_inspect(sync_conn).get_columns(table)}


def _table_info(sync_conn, table: str) -> set[tuple]:
    """Per-column schema of a table (PRAGMA table_info, order-insensitive)."""
    rows = sync_conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
    return {(r[1], r[2], r[3], r[4], r[5]) for r in rows}


async def _strip_v2_objects(conn) -> None:
    """Downgrade a create_all schema to the v1 shape (pre-P1a install)."""
    existing = set(await conn.run_sync(lambda c: sa_inspect(c).get_table_names()))
    for table, column in V2_COLUMNS.items():
        if column in await conn.run_sync(_columns_of, table):
            await conn.execute(text(f"ALTER TABLE {table} DROP COLUMN {column}"))
    for table in sorted(V2_TABLES & existing):
        await conn.execute(text(f"DROP TABLE {table}"))


async def _seed_v1_rows(conn) -> None:
    """Pre-upgrade v1 data: one user, conversation, document, chunk, memory."""
    ids = {name: uuid.uuid4().hex for name in
           ("user", "conversation", "document", "chunk", "memory")}
    await conn.execute(text(
        "INSERT INTO users (id, email, onboarding_done, is_verified, is_active, is_deleted)"
        " VALUES (:id, :email, 0, 1, 1, 0)"
    ), {"id": ids["user"], "email": f"{ids['user']}@test.invalid"})
    await conn.execute(text(
        "INSERT INTO conversations (id, user_id, document_count) VALUES (:id, :uid, 1)"
    ), {"id": ids["conversation"], "uid": ids["user"]})
    await conn.execute(text(
        "INSERT INTO documents (id, conversation_id, filename, file_path, chunk_count)"
        " VALUES (:id, :cid, 'v1.txt', '/tmp/v1.txt', 1)"
    ), {"id": ids["document"], "cid": ids["conversation"]})
    await conn.execute(text(
        "INSERT INTO document_chunks (id, document_id, content, chunk_index)"
        " VALUES (:id, :did, 'v1 chunk text', 0)"
    ), {"id": ids["chunk"], "did": ids["document"]})
    await conn.execute(text(
        "INSERT INTO memories (id, user_id, content, tags, pinned, is_shared, recall_count)"
        " VALUES (:id, :uid, 'v1 memory text', '[]', 0, 0, 0)"
    ), {"id": ids["memory"], "uid": ids["user"]})


@pytest_asyncio.fixture
async def v1_db(tmp_path: Path, monkeypatch):
    """A version-1 install: full v1 schema + rows, stamped ``user_version = 1``."""
    eng = await _engine(tmp_path, "v1.sqlite")
    monkeypatch.setattr(database, "engine", eng)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _strip_v2_objects(conn)
        await _seed_v1_rows(conn)
        await conn.execute(text("PRAGMA user_version = 1"))
    yield eng, tmp_path
    await eng.dispose()


async def _schema(conn) -> tuple[int, set[str]]:
    version = int((await conn.execute(text("PRAGMA user_version"))).scalar_one())
    tables = {r[0] for r in (await conn.execute(
        text("SELECT name FROM sqlite_master WHERE type='table'"))).all()}
    return version, tables


async def test_v1_install_is_upgraded_to_v2_and_backed_up(v1_db):
    eng, tmp_path = v1_db
    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version, tables = await _schema(conn)
        memories = (await conn.execute(text("SELECT content, revision FROM memories"))).all()
        chunks = (await conn.execute(text("SELECT content, revision FROM document_chunks"))).all()
        generations = (await conn.execute(text(
            "SELECT kind, generation, fingerprint, is_active FROM index_generations"))).all()
        fk_violations = (await conn.execute(text("PRAGMA foreign_key_check"))).all()
        integrity = (await conn.execute(text("PRAGMA integrity_check"))).scalar_one()

    assert version == 3
    assert V2_TABLES <= tables
    assert [tuple(row) for row in memories] == [("v1 memory text", 1)]  # existing rows: revision 1
    assert [tuple(row) for row in chunks] == [("v1 chunk text", 1)]
    assert fk_violations == []
    assert integrity == "ok"
    # An UPGRADE writes the two real rows INACTIVE: the install keeps serving its
    # OLD pointer — loud (the guard rejects the pre-P1b masked-mean contract)
    # until `migrate_qdrant.py cutover` flips it.
    assert {(r[0], r[1]) for r in generations} == {
        ("memory", generation_name("memory")), ("chunk", generation_name("chunk"))}
    assert not any(r[3] for r in generations), "the ladder never moves the pointer"
    # Same 64-char generation token the vector payload stamps as orivory_embed_generation.
    assert {r[2] for r in generations} == {fingerprint_generation(current_fingerprint())}
    _assert_fingerprint_fits_column(generations[0][2])

    # The milestone backup exists and predates the ladder's DDL.
    backups = list(Path(tmp_path).glob("*.pre-p1b.bak"))
    assert len(backups) == 1, "exactly one pre-ladder backup is required"
    backup = sqlite3.connect(backups[0])
    try:
        backup_tables = {r[0] for r in backup.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        backup_memory_cols = {r[1] for r in backup.execute("PRAGMA table_info(memories)")}
        backup_rows = backup.execute("SELECT content FROM memories").fetchall()
    finally:
        backup.close()
    assert not V2_TABLES & backup_tables, "backup must be pre-DDL (no v2 tables)"
    assert "revision" not in backup_memory_cols, "backup must be pre-DDL (no revision column)"
    assert backup_rows == [("v1 memory text",)]


async def test_unversioned_v1_shape_is_adopted_then_upgraded(tmp_path, monkeypatch):
    """The user's real case: full v1 install (27 tables) at ``user_version = 0``."""
    eng = await _engine(tmp_path, "legacy.sqlite")
    monkeypatch.setattr(database, "engine", eng)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    expected_v1_tables = {t.name for t in Base.metadata.sorted_tables} - V2_TABLES
    try:
        async with eng.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await _strip_v2_objects(conn)
            await _seed_v1_rows(conn)
            version, tables = await _schema(conn)
        assert version == 0
        assert tables == expected_v1_tables, "fixture must be a full v1 install"

        await database.bootstrap_sqlite()

        async with eng.connect() as conn:
            version, tables = await _schema(conn)
            memories = (await conn.execute(text("SELECT content, revision FROM memories"))).all()
        assert version == 3
        assert V2_TABLES <= tables
        assert [tuple(row) for row in memories] == [("v1 memory text", 1)]
        assert list(Path(tmp_path).glob("*.pre-p1b.bak"))
    finally:
        await eng.dispose()


async def test_upgraded_schema_matches_a_fresh_install(v1_db, tmp_path):
    """The ladder must land the same per-column schema create_all produces."""
    eng, _ = v1_db
    await database.bootstrap_sqlite()
    upgraded = {}
    async with eng.connect() as conn:
        for table in (*V2_COLUMNS, *V2_TABLES):
            upgraded[table] = await conn.run_sync(_table_info, table)

    fresh_eng = await _engine(tmp_path, "reference.sqlite")
    try:
        async with fresh_eng.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with fresh_eng.connect() as conn:
            for table in (*V2_COLUMNS, *V2_TABLES):
                assert upgraded[table] == await conn.run_sync(_table_info, table), (
                    f"{table}: ladder schema differs from create_all schema"
                )
    finally:
        await fresh_eng.dispose()


async def test_divergent_schema_still_fails_closed(tmp_path, monkeypatch):
    eng = await _engine(tmp_path, "divergent.sqlite")
    monkeypatch.setattr(database, "engine", eng)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    try:
        async with eng.begin() as conn:
            await conn.execute(text("CREATE TABLE legacy_marker (id INTEGER PRIMARY KEY)"))
        with pytest.raises(RuntimeError, match=r"does not match|requires a versioned"):
            await database.bootstrap_sqlite()
        async with eng.connect() as conn:
            version, tables = await _schema(conn)
        assert version == 0  # untouched, not silently stamped
        assert "index_outbox" not in tables
    finally:
        await eng.dispose()


async def test_bootstrap_is_idempotent_and_writes_the_two_real_generation_rows(tmp_path, monkeypatch):
    eng = await _engine(tmp_path, "fresh.sqlite")
    monkeypatch.setattr(database, "engine", eng)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    try:
        await database.bootstrap_sqlite()
        await database.bootstrap_sqlite()  # rerun on a v3 install: no-op

        async with eng.connect() as conn:
            version, tables = await _schema(conn)
            generations = (await conn.execute(text(
                "SELECT kind, generation, fingerprint, is_active FROM index_generations"))).all()
        assert version == database.SQLITE_SCHEMA_VERSION == 3
        assert V2_TABLES <= tables
        assert {(r[0], r[1]) for r in generations} == {
            ("memory", generation_name("memory")), ("chunk", generation_name("chunk"))}
        assert all(r[2] == fingerprint_generation(current_fingerprint()) for r in generations)
        _assert_fingerprint_fits_column(generations[0][2])
        assert all(r[3] == 1 for r in generations), "one active row per kind"
        assert COLLECTION_NAME not in {r[1] for r in generations}, (
            "the P1a transitional spelling is never seeded again")
        assert not list(Path(tmp_path).glob("*.pre-p1b.bak")), "no data step, no backup"
    finally:
        await eng.dispose()


async def test_upgraded_tables_enforce_outbox_and_generation_uniques(v1_db):
    from app.models.index_outbox import IndexGeneration, IndexOutbox

    eng, _ = v1_db
    await database.bootstrap_sqlite()

    first = IndexOutbox(
        kind="memory", entity_id=uuid.uuid4().hex, tenant_id=uuid.uuid4().hex,
        revision=1, operation="upsert", target_generation="gen-1",
    )
    async with AsyncSession(eng, expire_on_commit=False) as db:
        db.add(first)
        await db.commit()
        db.add(IndexOutbox(
            kind=first.kind, entity_id=first.entity_id, tenant_id=first.tenant_id,
            revision=first.revision, operation=first.operation,
            target_generation=first.target_generation,
        ))
        with pytest.raises(IntegrityError):
            await db.commit()
        await db.rollback()

        db.add(IndexGeneration(id=uuid.uuid4().hex, kind="memory", generation="gen-dup",
                               fingerprint="fp"))
        await db.commit()
        db.add(IndexGeneration(id=uuid.uuid4().hex, kind="memory", generation="gen-dup",
                               fingerprint="fp"))
        with pytest.raises(IntegrityError):
            await db.commit()


async def test_interrupted_upgrade_resumes_and_keeps_the_existing_backup(v1_db):
    """A crash after the v2 DDL but before the version stamp must resume.

    Reproduces the review repro: v2 DDL applied (it autocommits), ``user_version``
    still 1, a pre-existing non-empty ``.pre-v2.bak``. Re-entry must complete the
    upgrade and leave that backup's bytes untouched (the P1b step takes the
    milestone name, so the file is renamed, never rewritten).
    """
    eng, tmp_path = v1_db
    legacy = tmp_path / "v1.sqlite.pre-v2.bak"
    legacy.write_bytes(b"pre-v2 backup from the interrupted run")
    async with eng.begin() as conn:
        await conn.run_sync(database._upgrade_v1_to_v2)  # the DDL of the partial run
        version, _ = await _schema(conn)
    assert version == 1, "fixture must model an interruption, not a finished upgrade"

    before = legacy.read_bytes()
    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version, tables = await _schema(conn)
    milestone = tmp_path / "v1.sqlite.pre-p1b.bak"
    assert version == 3
    assert V2_TABLES <= tables
    assert list(Path(tmp_path).glob("*.pre-p1b.bak")) == [milestone], (
        "the interrupted run's backup must be reused, not replaced"
    )
    assert not legacy.exists(), "the milestone name is what the P1b step leaves behind"
    assert milestone.read_bytes() == before, "an existing backup is never overwritten"


async def test_empty_existing_backup_still_refuses(v1_db):
    """The refusal survives only for an unusable (empty) backup file."""
    _, tmp_path = v1_db
    (tmp_path / "v1.sqlite.pre-v2.bak").write_bytes(b"")
    with pytest.raises(RuntimeError, match="empty or unreadable"):
        await database.bootstrap_sqlite()


# ── T5: the boot hook drains the outbox on a real v2 install ─────────────────


async def test_app_boot_drains_the_outbox_and_survives_a_chroma_outage(tmp_path, monkeypatch):
    """The lifespan replays pending intents after the bootstrap; Chroma down ≠ no boot."""
    from app.main import app as fastapi_app
    from app.main import lifespan
    from app.models.index_outbox import IndexOutbox
    from app.models.memory import Memory
    from app.models.user import User
    from app.retrieval.memory import outbox

    eng = await _engine(tmp_path, "boot.sqlite")
    sessions = async_sessionmaker(eng, class_=AsyncSession, expire_on_commit=False, autoflush=False)
    monkeypatch.setattr(database, "engine", eng)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    monkeypatch.setattr(database, "AsyncSessionLocal", sessions)
    monkeypatch.setattr(outbox, "AsyncSessionLocal", sessions)
    # The lifespan gates the bootstrap and the drain on settings.DATABASE_URL
    # itself, so it must point at this test's file whatever the ambient URL is
    # (the shared tests/conftest.py defaults it to Postgres for the full-stack
    # suites). Without the pin the hook no-ops and the row is never attempted.
    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'boot.sqlite'}")

    async def offline_storage():
        return None

    import app.storage

    monkeypatch.setattr(app.storage, "ensure_bucket", offline_storage)  # storage is not the contract here

    try:
        await database.bootstrap_sqlite()  # the real ladder: fresh v2 install
        owner = uuid.uuid4()
        async with sessions() as db:
            db.add(User(id=owner, email=f"{owner.hex}@test.invalid", hashed_password="x",
                        display_name="Owner", is_verified=True, is_active=True))
            memory = Memory(user_id=owner, content="heal me", tags=[])
            db.add(memory)
            outbox.bump_revision(memory)
            await outbox.enqueue_upsert(db, memory)
            await db.commit()
            memory_id = memory.id

        async def chroma_down(_memory):
            raise RuntimeError("chroma connection refused")

        monkeypatch.setattr(outbox, "upsert_memory", chroma_down)

        booted = False
        async with lifespan(fastapi_app):
            booted = True  # the app reached readiness despite the vector outage

        assert booted
        async with sessions() as db:
            row = (await db.execute(select(IndexOutbox))).scalars().one()
        assert row.entity_id == memory_id.hex
        assert (row.status, row.attempts) == ("pending", 1)  # stays pending with backoff
        assert "chroma connection refused" in row.last_error
    finally:
        await eng.dispose()
