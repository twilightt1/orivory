from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import event, text
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

IS_SQLITE = settings.DATABASE_URL.startswith("sqlite")
SQLITE_SCHEMA_VERSION = 1


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


def _adoptable_sqlite_schema(sync_conn) -> bool:
    """True when an unversioned SQLite schema matches the model metadata.

    Pre-versioning installs (<= v1.1.0) built the full schema with
    ``create_all`` and left ``PRAGMA user_version`` at 0. Such a DB is
    schema-identical to version 1 by construction, so startup may adopt it
    by stamping the version. Divergence (missing table/column) is not a
    match and must fail closed.
    """
    insp = sa_inspect(sync_conn)
    existing = set(insp.get_table_names())
    for table in Base.metadata.sorted_tables:
        if table.name not in existing:
            return False
        have = {col["name"] for col in insp.get_columns(table.name)}
        if {col.name for col in table.columns} - have:
            return False
    return True


async def bootstrap_sqlite() -> None:
    """Create a fresh, versioned SQLite schema for lite mode.

    Full-stack (Postgres) deployments use Alembic migrations instead —
    `docker compose up migrate` / `alembic upgrade head`. SQLite deployments
    are created fresh from the model metadata. An existing unversioned
    schema is adopted as version 1 only when it matches the metadata exactly
    (schema-identical, no migration needed); any divergence fails closed —
    ``create_all`` is never used as an existing-schema migration mechanism.
    """
    if not IS_SQLITE:
        raise RuntimeError("bootstrap_sqlite() is only for SQLite deployments")
    from app import models  # noqa: F401 — register every model on Base

    async with engine.begin() as conn:
        version = int((await conn.execute(text("PRAGMA user_version"))).scalar_one())
        tables = await conn.run_sync(lambda sync_conn: set(sa_inspect(sync_conn).get_table_names()))
        if version not in (0, SQLITE_SCHEMA_VERSION):
            raise RuntimeError(
                f"unsupported SQLite schema version {version}; expected "
                f"{SQLITE_SCHEMA_VERSION}"
            )
        if version == 0:
            if not tables:
                await conn.run_sync(Base.metadata.create_all)
                await conn.execute(text(f"PRAGMA user_version = {SQLITE_SCHEMA_VERSION}"))
            elif await conn.run_sync(_adoptable_sqlite_schema):
                await conn.execute(text(f"PRAGMA user_version = {SQLITE_SCHEMA_VERSION}"))
            else:
                raise RuntimeError(
                    "existing SQLite schema does not match SQLITE_SCHEMA_VERSION "
                    f"{SQLITE_SCHEMA_VERSION}; requires a versioned SQLite migration"
                )
        elif {"users", "memories"} - tables:
            raise RuntimeError(
                "SQLite schema version is marked active but required tables are missing"
            )


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
