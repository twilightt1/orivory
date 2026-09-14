import logging
import os
import shutil
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import event, text
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

log = logging.getLogger(__name__)

IS_SQLITE = settings.DATABASE_URL.startswith("sqlite")
SQLITE_SCHEMA_VERSION = 2

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


def _sqlite_path() -> str:
    """File path of the canonical SQLite database behind the module engine."""
    path = engine.url.database
    if not path:
        raise RuntimeError("SQLite DATABASE_URL has no file path")
    return path


def _backup_before_ddl(db_path: str) -> str:
    """Consistent pre-DDL backup (VACUUM INTO includes committed WAL frames).

    Resumable: a crash between the backup and the version stamp leaves a
    complete backup behind, so a later attempt reuses it instead of refusing
    (the ladder is idempotent and safe to re-run). Only an empty or unreadable
    backup file is an error — never silently reuse a truncated one.
    """
    dest = f"{db_path}.pre-v2.bak"
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


def _seed_transitional_generation(sync_conn) -> None:
    """INSERT-IF-ABSENT: the vector generation this install currently serves.

    P1b keeps Qdrant (``vector_store.COLLECTION_NAME`` + the active embedding
    fingerprint) as that generation. The ``fingerprint`` column stores the
    64-char ``fingerprint_generation`` token (the same family the vector payload
    stamps as ``orivory_embed_generation``), not the long canonical contract
    string — it must fit ``String(128)`` on Postgres. The P1b cutover replaces
    this row with the real one. Idempotent, so it also repairs an install
    missing the row.
    """
    from app.retrieval.embedding_fingerprint import current_fingerprint, fingerprint_generation
    from app.retrieval.memory.vector_store import COLLECTION_NAME

    sync_conn.exec_driver_sql(
        "INSERT INTO index_generations (id, kind, generation, fingerprint, is_active, created_at)"
        " SELECT ?, 'memory', ?, ?, 1, CURRENT_TIMESTAMP"
        " WHERE NOT EXISTS (SELECT 1 FROM index_generations"
        "                   WHERE kind = 'memory' AND generation = ?)",
        (uuid.uuid4().hex, COLLECTION_NAME, fingerprint_generation(current_fingerprint()),
         COLLECTION_NAME),
    )


async def bootstrap_sqlite() -> None:
    """Create or upgrade the canonical SQLite schema for lite mode.

    Full-stack (Postgres) deployments use Alembic migrations instead —
    `docker compose up migrate` / `alembic upgrade head`. SQLite deployments
    are created fresh from the model metadata. An existing install goes
    through the versioned ``user_version`` ladder: an unversioned v1-shape
    schema is adopted and upgraded v1 -> v2 (backup before DDL, foreign-key +
    integrity checks after); divergence fails closed — ``create_all`` is never
    used as an existing-schema migration mechanism.
    """
    if not IS_SQLITE:
        raise RuntimeError("bootstrap_sqlite() is only for SQLite deployments")
    from app import models  # noqa: F401 — register every model on Base

    async with engine.begin() as conn:
        version = int((await conn.execute(text("PRAGMA user_version"))).scalar_one())
        tables = await conn.run_sync(lambda sync_conn: set(sa_inspect(sync_conn).get_table_names()))
        if version not in (0, 1, SQLITE_SCHEMA_VERSION):
            raise RuntimeError(
                f"unsupported SQLite schema version {version}; expected "
                f"{SQLITE_SCHEMA_VERSION}"
            )
        if version == 0:
            if not tables:
                await conn.run_sync(Base.metadata.create_all)
            elif not await conn.run_sync(_upgradable_v1_schema):
                raise RuntimeError(
                    "existing SQLite schema does not match SQLITE_SCHEMA_VERSION "
                    f"{SQLITE_SCHEMA_VERSION}; requires a versioned SQLite migration"
                )
        elif {"users", "memories"} - tables:
            raise RuntimeError(
                "SQLite schema version is marked active but required tables are missing"
            )
        if version in (0, 1) and tables:
            _backup_before_ddl(_sqlite_path())
            await conn.run_sync(_upgrade_v1_to_v2)
            await conn.run_sync(_check_sqlite_foreign_keys)
        await conn.execute(text(f"PRAGMA user_version = {SQLITE_SCHEMA_VERSION}"))
        integrity = await conn.run_sync(
            lambda sync_conn: sync_conn.exec_driver_sql("PRAGMA integrity_check").fetchone()
        )
        if integrity is None or integrity[0] != "ok":
            raise RuntimeError("SQLite integrity_check failed after schema migration")
        await conn.run_sync(_seed_transitional_generation)


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
