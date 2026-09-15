import logging
import os
import shutil
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import lru_cache

from sqlalchemy import event, insert, select, text, update
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

log = logging.getLogger(__name__)

IS_SQLITE = settings.DATABASE_URL.startswith("sqlite")
# v1 = the pre-versioning schema, v2 = revisions + outbox/generation tables,
# v3 = the P1b generation rows (a DATA step: no DDL, no new tables/columns).
SQLITE_SCHEMA_VERSION = 3

# Objects added by the v1 -> v2 ladder; excluded from the v1 shape check.
V2_TABLES = ("index_outbox", "index_generations", "memory_suppressions")
V2_COLUMNS = {"memories": "revision", "document_chunks": "revision"}


def _make_engine():
    """Create the async engine for the configured DATABASE_URL.

    Full-stack deployments use Postgres (pool sizing applies). Lite mode
    uses SQLite (`sqlite+aiosqlite`) with no pool sizing args — SQLite has
    no pool to size and SQLAlchemy would reject pool_size there.
    """
    if IS_SQLITE:
        return create_async_engine(
            settings.DATABASE_URL,
            echo=settings.ENVIRONMENT == "development",
            connect_args={"check_same_thread": False},
        )
    return create_async_engine(
        settings.DATABASE_URL,
        pool_size=settings.DATABASE_POOL_SIZE,
        max_overflow=settings.DATABASE_MAX_OVERFLOW,
        pool_pre_ping=True,
        echo=settings.ENVIRONMENT == "development",
    )


engine = _make_engine()


def _configure_sqlite_connection(dbapi_connection, _record=None):
    """Apply the canonical durability policy to one SQLite connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.execute("PRAGMA synchronous=FULL")
    cursor.close()


if IS_SQLITE:
    # SQLite does not enforce foreign keys unless asked, and the memory
    # hub relies on ON DELETE CASCADE everywhere.
    event.listen(engine.sync_engine, "connect", _configure_sqlite_connection)


AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    pass


def _upgradable_v1_schema(sync_conn) -> bool:
    """True when an unversioned SQLite schema matches the v1 shape.

    Pre-versioning installs (<= v1.1.0) built the full v1 schema with
    ``create_all`` and left ``PRAGMA user_version`` at 0. v1 is the model
    metadata minus the v2 additions (revision columns + index_outbox /
    index_generations / memory_suppressions), so such a DB is adoptable and
    then upgraded by the ladder. Any missing v1 table or column is divergence
    and must fail closed.
    """
    insp = sa_inspect(sync_conn)
    existing = set(insp.get_table_names())
    for table in Base.metadata.sorted_tables:
        if table.name in V2_TABLES:
            continue
        if table.name not in existing:
            return False
        v2_column = V2_COLUMNS.get(table.name)
        have = {col["name"] for col in insp.get_columns(table.name)}
        if {col.name for col in table.columns if col.name != v2_column} - have:
            return False
    return True


def _backup_before_ddl(db_path: str, suffix: str = "pre-v2") -> str:
    """Consistent pre-DDL backup (VACUUM INTO includes committed WAL frames).

    Resumable: a crash between the backup and the version stamp leaves a
    complete backup behind, so a later attempt reuses it instead of refusing
    (the ladder is idempotent and safe to re-run). Only an empty or unreadable
    backup file is an error — never silently reuse a truncated one.
    """
    dest = f"{db_path}.{suffix}.bak"
    if os.path.exists(dest):
        if os.path.getsize(dest) == 0 or not os.access(dest, os.R_OK):
            raise RuntimeError(f"existing migration backup {dest} is empty or unreadable")
        log.warning("reusing pre-migration backup %s from an interrupted upgrade", dest)
        return dest
    size = os.path.getsize(db_path)
    if shutil.disk_usage(os.path.dirname(db_path) or ".").free < size * 2:
        raise RuntimeError("insufficient free disk for pre-migration backup")
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("VACUUM INTO ?", (dest,))
    finally:
        conn.close()
    return dest


def _upgrade_v1_to_v2(sync_conn) -> None:
    """v1 -> v2 DDL: revision counters + durable index-intent tables.

    Idempotent: a current-model install that was never stamped (user_version 0)
    passes the shape check with the v2 objects already in place, so each step
    inspects before touching the schema.
    """
    from app import models  # noqa: F401 — register every v2 table on Base

    insp = sa_inspect(sync_conn)
    for table_name, column in V2_COLUMNS.items():
        have = {col["name"] for col in insp.get_columns(table_name)}
        if column not in have:
            # DDL mirrors the model's server_default="1" (SQLAlchemy renders
            # DEFAULT '1'); SQLite's INTEGER affinity stores both as 1.
            sync_conn.exec_driver_sql(
                f"ALTER TABLE {table_name} ADD COLUMN {column} INTEGER NOT NULL DEFAULT '1'"
            )
    Base.metadata.create_all(sync_conn, tables=[Base.metadata.tables[name] for name in V2_TABLES])


def _check_sqlite_foreign_keys(sync_conn) -> None:
    violations = sync_conn.exec_driver_sql("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise RuntimeError(
            f"SQLite foreign_key_check found violations after schema migration: {violations[:5]}"
        )


def activate_generations(conn) -> dict[str, tuple[str, str]]:
    """Insert-or-activate the real generation row for BOTH kinds (spec §4.2, R24).

    The ONE definition of "the active generation is memory + chunk at the
    current embedding contract": the SQLite ladder (v2 -> v3) and
    ``migrate_qdrant.py cutover`` both call it, so the pointer cannot drift from
    the contract and the two paths can never disagree. Dialect-neutral Core SQL
    (ruling R10: the CLI must also serve Postgres, where no ladder runs).

    Every other row for the kind is retired FIRST: with two active rows the
    runtime's ``is_active`` lookup would be a coin toss — the P1a transitional
    ``Orivory_memories`` row was exactly that case. Returns
    ``{kind: (generation, fingerprint_token)}``.
    """
    from app.models.index_outbox import IndexGeneration
    from app.retrieval.embedding_fingerprint import (
        current_fingerprint,
        fingerprint_generation,
        generation_name,
    )

    table = IndexGeneration.__table__
    token = fingerprint_generation(current_fingerprint())
    activated: dict[str, tuple[str, str]] = {}
    for kind in ("memory", "chunk"):
        generation = generation_name(kind)
        conn.execute(
            update(table)
            .where(table.c.kind == kind, table.c.generation != generation)
            .values(is_active=False)
        )
        exists = conn.execute(
            select(table.c.id).where(table.c.kind == kind, table.c.generation == generation)
        ).first()
        if exists is None:
            conn.execute(
                insert(table).values(
                    id=uuid.uuid4().hex, kind=kind, generation=generation,
                    fingerprint=token, is_active=True, created_at=datetime.now(UTC),
                )
            )
        else:
            conn.execute(
                update(table)
                .where(table.c.kind == kind, table.c.generation == generation)
                .values(fingerprint=token, is_active=True)
            )
        activated[kind] = (generation, token)
    return activated


def _p1b_backup(db_path: str) -> str:
    """The P1b milestone backup: the pre-ladder bytes live under ``.pre-p1b.bak``.

    A v1 install upgraded in one boot already produced ``.pre-v2.bak`` holding
    the pre-any-change state; that file is RENAMED into the milestone name (the
    same bytes, never a second copy, never overwritten). Anything else gets a
    fresh consistent snapshot.
    """
    dest, legacy = f"{db_path}.pre-p1b.bak", f"{db_path}.pre-v2.bak"
    if not os.path.exists(dest) and os.path.exists(legacy):
        os.replace(legacy, dest)
        log.info("P1b milestone backup %s now carries the pre-ladder snapshot", dest)
    return _backup_before_ddl(db_path, suffix="pre-p1b")


def _upgrade_v2_to_v3(sync_conn) -> None:
    """v2 -> v3: the REAL generation rows — a data step, no DDL (ruling R24).

    P1a seeded ONE transitional row and re-created it on every boot
    (``Orivory_memories`` + the then-current fingerprint). The ladder now writes
    the two rows the runtime actually serves — named by ``generation_name(kind)``
    and stamped with ``fingerprint_generation(current_fingerprint())`` — and
    retires every other row per kind. Idempotent, so a later boot is a no-op and
    nothing resurrects the old spelling.
    """
    activate_generations(sync_conn)


def _conn_sqlite_path(conn) -> str:
    """File path of the SQLite database behind an engine/connection."""
    path = conn.engine.url.database
    if not path:
        raise RuntimeError("SQLite DATABASE_URL has no file path")
    return path


def upgrade_sqlite_schema(conn) -> None:
    """The versioned SQLite ladder, on a SYNC connection.

    ONE definition of the ladder: the async boot path
    (:func:`bootstrap_sqlite`) and the offline migration CLI
    (``scripts/migrate_qdrant.py``) both call it, so the CLI can never upgrade a
    database differently from the app it will serve.

    Full-stack (Postgres) deployments use Alembic migrations instead. SQLite
    deployments are created fresh from the model metadata; an existing install
    goes through ``user_version``: an unversioned v1-shape schema is adopted and
    upgraded v1 -> v2 (backup before DDL, foreign-key + integrity checks after),
    then v2 -> v3 (the P1b generation-rows data step). Divergence fails closed —
    ``create_all`` is never used as an existing-schema migration mechanism.
    """
    from app import models  # noqa: F401 — register every model on Base

    path = _conn_sqlite_path(conn)
    version = int(conn.execute(text("PRAGMA user_version")).scalar_one())
    tables = set(sa_inspect(conn).get_table_names())
    if version not in (0, 1, 2, SQLITE_SCHEMA_VERSION):
        raise RuntimeError(
            f"unsupported SQLite schema version {version}; expected {SQLITE_SCHEMA_VERSION}"
        )
    if version == 0:
        if not tables:
            Base.metadata.create_all(conn)
        elif not _upgradable_v1_schema(conn):
            raise RuntimeError(
                "existing SQLite schema does not match SQLITE_SCHEMA_VERSION "
                f"{SQLITE_SCHEMA_VERSION}; requires a versioned SQLite migration"
            )
    elif {"users", "memories"} - tables:
        raise RuntimeError(
            "SQLite schema version is marked active but required tables are missing"
        )
    if version in (0, 1) and tables:
        _backup_before_ddl(path)
        _upgrade_v1_to_v2(conn)
        _check_sqlite_foreign_keys(conn)
    if version in (0, 1, 2) and tables:
        # P1b milestone backup (R24): reused if present, renamed if the
        # v1 -> v2 step just produced the pre-ladder file, else a fresh
        # consistent snapshot. Never overwritten.
        _p1b_backup(path)
    # Idempotent data step, also for a fresh install (ruling R24): the two real
    # rows are the runtime's only pointer, and nothing re-seeds the P1a
    # transitional spelling on a later boot.
    _upgrade_v2_to_v3(conn)
    conn.execute(text(f"PRAGMA user_version = {SQLITE_SCHEMA_VERSION}"))
    integrity = conn.exec_driver_sql("PRAGMA integrity_check").fetchone()
    if integrity is None or integrity[0] != "ok":
        raise RuntimeError("SQLite integrity_check failed after schema migration")


async def bootstrap_sqlite() -> None:
    """Create or upgrade the canonical SQLite schema for lite mode.

    Thin async wrapper over :func:`upgrade_sqlite_schema` (the one ladder
    definition shared with the migration CLI).
    """
    if not IS_SQLITE:
        raise RuntimeError("bootstrap_sqlite() is only for SQLite deployments")
    async with engine.begin() as conn:
        await conn.run_sync(upgrade_sqlite_schema)


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ── Synchronous session (in-process background work) ─────────────────────────
# Sync helpers (ingestion pipeline, graph build, reindex) cannot use the async
# engine above, so they share one process-level sync engine + sessionmaker
# instead of creating a brand-new connection pool per invocation.


@lru_cache(maxsize=1)
def get_sync_engine() -> Engine:
    """Return the process-wide synchronous engine, creating it on first use."""
    from sqlalchemy import create_engine

    url = settings.DATABASE_URL
    if url.startswith("sqlite"):
        sync_url = url.replace("+aiosqlite", "")
        eng = create_engine(sync_url, connect_args={"check_same_thread": False})

        @event.listens_for(eng, "connect")
        def _enable_sync_sqlite_pragmas(dbapi_connection, _record):
            _configure_sqlite_connection(dbapi_connection, _record)

        return eng

    sync_url = url.replace("+asyncpg", "+psycopg2")
    return create_engine(
        sync_url,
        pool_pre_ping=True,
        pool_size=settings.DATABASE_POOL_SIZE,
        max_overflow=settings.DATABASE_MAX_OVERFLOW,
        pool_recycle=1800,
    )


@lru_cache(maxsize=1)
def _get_sync_sessionmaker() -> sessionmaker:
    return sessionmaker(bind=get_sync_engine(), expire_on_commit=False, autoflush=False)


@contextmanager
def sync_session() -> Iterator[Session]:
    """Yield a synchronous session bound to the shared engine."""
    session = _get_sync_sessionmaker()()
    try:
        yield session
    finally:
        session.close()
