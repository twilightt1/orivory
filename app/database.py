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
# v3 = the P1b generation rows (a DATA step: no DDL, no new tables/columns),
# v4 = the P2 FTS5 memory index + its triggers (DDL: a virtual table, which can
# never come from model metadata — the ladder creates it with exec_driver_sql).
SQLITE_SCHEMA_VERSION = 4

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


def activate_generations(conn, *, activate: bool = True) -> dict[str, tuple[str, str]]:
    """Insert the real generation row for BOTH kinds (spec §4.2, R24).

    The ONE definition of "the active generation is memory + chunk at the
    current embedding contract": the SQLite ladder (v2 -> v3) and
    ``migrate_qdrant.py cutover`` both call it, so the pointer cannot drift from
    the contract and the two paths can never disagree. Dialect-neutral Core SQL
    (ruling R10: the CLI must also serve Postgres, where no ladder runs).

    ``activate=False`` is the EXPAND half of an upgrade: the rows are written
    (or refreshed) and stay INACTIVE, so the install keeps serving its OLD
    pointer. That is loud — the read path's contract guard raises — only where
    the old pointer names a contract the new code no longer matches (the P1a
    transitional row: masked mean). With NO active row (Postgres: P1a never
    seeded this table) ``active_generation`` falls back to the transitional
    generation name with no fingerprint, an EMPTY generation is deliberately
    allowed, and reads answer ``[]`` until ``cutover`` flips the pointer.
    ``activate=True`` is the FLIP (``cutover``, and a fresh install with nothing
    to serve yet): every other row for the kind is retired FIRST — with two
    active rows the runtime's ``is_active`` lookup would be a coin toss, the
    P1a transitional ``Orivory_memories`` row was exactly that case — then the
    target row is inserted-or-activated. Returns
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
        if activate:
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
                    fingerprint=token, is_active=activate, created_at=datetime.now(UTC),
                )
            )
        else:
            values: dict = {"fingerprint": token}
            if activate:
                values["is_active"] = True
            conn.execute(
                update(table)
                .where(table.c.kind == kind, table.c.generation == generation)
                .values(**values)
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


def _upgrade_v2_to_v3(sync_conn, *, activate: bool) -> None:
    """v2 -> v3: the REAL generation rows — a data step, no DDL (ruling R24).

    Runs ONCE, on the version transition (a later boot never touches the
    manifest again: a restart must not re-assert a pointer that ``cutover``
    moved — or that a rollback moved back).

    ``activate`` is True only for a FRESH install, which has nothing to serve
    yet. An UPGRADE writes the rows INACTIVE: the OLD pointer keeps serving and
    an install that has not been through ``migrate_qdrant.py cutover`` fails
    LOUD (the read path's contract guard) instead of answering every recall
    with an empty result from a generation nobody built — true of the lite v2
    install, whose old pointer IS the masked-mean transitional row. With no
    active row at all there is nothing for the guard to reject and reads fall
    back to the transitional name (``activate_generations`` documents that).

    P1a seeded ONE transitional row and re-created it on every boot
    (``Orivory_memories`` + the then-current fingerprint); the two rows written
    here are the ones ``generation_name(kind)`` names and the runtime will
    serve.
    """
    activate_generations(sync_conn, activate=activate)


def _upgrade_v3_to_v4(sync_conn) -> None:
    """v3 -> v4: the FTS5 memory index and its triggers (ruling R18(p2)).

    Runs ONCE, on the version transition, and on a fresh install too (which has
    nothing to back up). FTS DDL cannot come from model metadata — there is no
    ORM model for a virtual table and no Alembic step for it: SQLite-only DDL
    belongs in the ladder. The step creates the index and BACKFILLS it from
    ``memories`` in the same transaction as the stamp, so the index is complete
    the moment a v4 install can serve; a crash mid-step leaves v3 stamped and
    the step re-runs (its DDL is IF NOT EXISTS, its backfill is coverage-driven).

    A later boot never re-runs it: a restart must not silently rebuild an index
    an operator repaired (or mask the drift ``rebuild`` reports).
    """
    from app.retrieval.memory import lexical_index

    lexical_index.create_index(sync_conn)
    report = lexical_index.rebuild(sync_conn)
    log.info("SQLite schema v4: FTS5 memory index ready", extra=report)


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

    Full-stack (Postgres) deployments use Alembic migrations instead — and get
    no FTS: the lexical leg is SQLite-only and reports itself unavailable there
    (ruling R3(p2), no Alembic step). SQLite deployments are created fresh from
    the model metadata; an existing install goes through ``user_version``: an
    unversioned v1-shape schema is adopted and upgraded v1 -> v2 (backup before
    DDL, foreign-key + integrity checks after), then v2 -> v3 (the P1b
    generation-rows data step, which runs ONCE — the CLI's ``cutover`` owns
    every later pointer move), then v3 -> v4 (the P2 FTS5 memory index + its
    triggers, also ONCE — a restart never rebuilds an index an operator
    repaired). Divergence fails closed — ``create_all`` is never used as an
    existing-schema migration mechanism.
    """
    from app import models  # noqa: F401 — register every model on Base

    path = _conn_sqlite_path(conn)
    version = int(conn.execute(text("PRAGMA user_version")).scalar_one())
    tables = set(sa_inspect(conn).get_table_names())
    fresh_install = version == 0 and not tables
    if version not in (0, 1, 2, 3, SQLITE_SCHEMA_VERSION):
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
    if version in (0, 1, 2):
        # The v2 -> v3 DATA step, ONCE, on the transition: the two real rows.
        # A fresh install (no tables before this call) may have them active —
        # nothing to serve yet. An UPGRADE must not: the old pointer keeps
        # serving — loud (a contract mismatch the guard raises) when that
        # pointer is the masked-mean P1a row, empty results when there is no
        # active row at all; `cutover` flips it either way.
        # A later boot (already v3) never re-asserts the pointer (that would
        # undo a rollback) and never re-mutates the manifest.
        _upgrade_v2_to_v3(conn, activate=fresh_install)
    if version in (0, 1, 2, 3):
        # P2 milestone backup (R18): its OWN name, taken where the ladder stands
        # now (v3, pre-FTS) — nothing to inherit from the P1b file, so no
        # rename. A fresh install has nothing to back up.
        if tables:
            _backup_before_ddl(path, suffix="pre-p2")
        # The v3 -> v4 DDL step, ONCE, on the transition; a fresh install runs
        # it too, so every v4 install serves a lexical leg.
        _upgrade_v3_to_v4(conn)
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
