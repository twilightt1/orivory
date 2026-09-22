import logging
import os
import shutil
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import lru_cache
from urllib.parse import quote

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
# never come from model metadata — the ladder creates it with exec_driver_sql),
# v5 = the P4a namespace column + its (namespace, user_id) index (DDL; the
# backfill IS the column default, so every pre-existing row is 'personal'),
# v6 = the P4b suppression-ledger columns (DDL: namespace + content_hash on
# memory_suppressions, both NULLABLE and never backfilled — NULL is the honest
# "unknown" for a row that predates the hash being computed at upload time),
# v7 = the P4b opt-in retention settings on users (DDL: retention_enabled, a
# NOT NULL Boolean whose constant default IS the backfill — every pre-existing
# user is OFF, spec §8.1 never lets a migration turn auto-expiration on — and
# retention_days, NULLABLE: "no window chosen" is a real state, not 0).
# v8 = the P3 freshness barrier's count index on index_outbox (DDL: an index on
# (kind, tenant_id, status), the shape create_all installs — the barrier counts
# this tenant's memory intents once per poll and the outbox is never pruned,
# so without it every poll scanned the table).
SQLITE_SCHEMA_VERSION = 8

# Objects added by the v1 -> v2 ladder; excluded from the v1 shape check.
V2_TABLES = ("index_outbox", "index_generations", "memory_suppressions")
V2_COLUMNS = {"memories": "revision", "document_chunks": "revision"}
# Columns added by LATER ladder steps, excluded from the v1 shape check for the
# same reason: a pre-versioning install predates them too, and adoption must
# accept the genuine v1 shape whether or not an earlier boot already got this
# far (a partially-upgraded file is upgraded again, not refused).
V5_COLUMNS = {"memories": "namespace"}
# Columns added by the v4 -> v5 and v5 -> v6 -> v7 steps are excluded the same
# way; V7 (P4b/T6) adds two columns to ``users`` (a v1 file predates them).
V7_COLUMNS = {"users": ("retention_enabled", "retention_days")}


def _make_engine():
    """Create the async engine for the configured DATABASE_URL.

    SQLite (`sqlite+aiosqlite`) is the only supported dialect: the product is
    one container with a canonical SQLite store, and there is no pool to size.
    Anything else fails fast at import instead of booting a deployment the
    rest of the stack (the versioned ladder, FTS5, the single-process
    assumptions) cannot serve.
    """
    if not IS_SQLITE:
        raise ValueError(
            f"DATABASE_URL must be a SQLite URL (sqlite+aiosqlite:///...), got {settings.DATABASE_URL!r}"
        )
    return create_async_engine(
        settings.DATABASE_URL,
        echo=settings.ENVIRONMENT == "development",
        connect_args={"check_same_thread": False},
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
    metadata minus the ladder's own additions (the v2 tables + revision columns,
    the v5 namespace column, and the v7 users retention settings), so such a DB
    is adoptable and then upgraded by the ladder. Any missing v1 table or column
    is divergence and must fail closed.
    """
    insp = sa_inspect(sync_conn)
    existing = set(insp.get_table_names())
    for table in Base.metadata.sorted_tables:
        if table.name in V2_TABLES:
            continue
        if table.name not in existing:
            return False
        later_columns = (V2_COLUMNS.get(table.name), V5_COLUMNS.get(table.name),
                         *V7_COLUMNS.get(table.name, ()))
        have = {col["name"] for col in insp.get_columns(table.name)}
        if {col.name for col in table.columns if col.name not in later_columns} - have:
            return False
    return True


def _is_valid_sqlite_backup(path: str) -> bool:
    """True when *path* opens as a SQLite database whose quick_check says ok."""
    try:
        # The path is percent-escaped: interpolated raw, a '?' or '#' in it is
        # read as the URI's query/fragment marker, so 'mode=ro' vanishes and
        # the probe opens a DIFFERENT (just-created, empty) file and answers
        # "valid" for a corrupt backup.
        conn = sqlite3.connect(f"file:{quote(path)}?mode=ro", uri=True)
    except sqlite3.Error:
        return False
    try:
        return conn.execute("PRAGMA quick_check").fetchone() == ("ok",)
    except sqlite3.DatabaseError:
        return False
    finally:
        conn.close()


def _backup_before_ddl(db_path: str, suffix: str = "pre-v2") -> str:
    """Consistent pre-DDL backup (VACUUM INTO includes committed WAL frames).

    Resumable: a crash between the backup and the version stamp leaves a
    complete backup behind, so a later attempt reuses it instead of refusing
    (the ladder is idempotent and safe to re-run). Only an empty or unreadable
    backup file is an error — never silently reuse a truncated one — and the
    file must still BE a SQLite database (quick_check ok): reusing arbitrary
    non-empty bytes would run the destructive DDL with no recoverable copy.
    """
    dest = f"{db_path}.{suffix}.bak"
    if os.path.exists(dest):
        if os.path.getsize(dest) == 0 or not os.access(dest, os.R_OK):
            raise RuntimeError(f"existing migration backup {dest} is empty or unreadable")
        if not _is_valid_sqlite_backup(dest):
            raise RuntimeError(
                f"existing migration backup {dest} is not a valid SQLite database "
                "(PRAGMA quick_check failed): remove it and re-run the upgrade so a "
                "fresh snapshot can be taken"
            )
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


def _upgrade_v4_to_v5(sync_conn) -> None:
    """v4 -> v5: ``memories.namespace`` + its ``(namespace, user_id)`` index.

    Runs ONCE, on the version transition, and on a fresh install too (which has
    nothing to back up). The column is NOT NULL with a constant default, so the
    ADD COLUMN is itself the backfill: every pre-existing row IS ``'personal'`` —
    the only namespace this phase can produce (it is never derived from client
    input, so there is nothing per-row to compute).

    The DDL mirrors the model byte for byte: ``String(32) NOT NULL
    server_default="personal"`` renders as exactly the raw SQL below, so an
    upgraded file and a fresh one are indistinguishable (``tests/lite`` compares
    their ``PRAGMA table_info``). ``VARCHAR(32)``, not ``TEXT``: the type the
    model declares is the type the ladder installs.

    Idempotent: a crash between the DDL and the version stamp re-enters with the
    column already present, and SQLite has no ``ADD COLUMN IF NOT EXISTS`` — so
    the column is inspected first (the index has ``IF NOT EXISTS``). A later boot
    never re-runs it: a restart must not re-assert a namespace an operator moved.
    """
    insp = sa_inspect(sync_conn)
    have = {col["name"] for col in insp.get_columns("memories")}
    if "namespace" not in have:
        sync_conn.exec_driver_sql(
            "ALTER TABLE memories ADD COLUMN namespace VARCHAR(32) NOT NULL DEFAULT 'personal'"
        )
    sync_conn.exec_driver_sql(
        "CREATE INDEX IF NOT EXISTS ix_memories_namespace_user ON memories(namespace, user_id)"
    )


def _upgrade_v5_to_v6(sync_conn) -> None:
    """v5 -> v6: ``memory_suppressions.namespace`` + ``content_hash``.

    Runs ONCE, on the version transition, and on a fresh install too (which has
    nothing to back up). Both columns are NULLABLE with no default: a
    pre-existing suppression row has neither value, and NULL is the honest
    "unknown" — the content hash is computed at UPLOAD time by the P4b/T4
    guards, never backfilled here (R38).

    The DDL mirrors the model: ``String(32)`` / ``String(64)`` render as
    ``VARCHAR(32)`` / ``VARCHAR(64)``, the exact types ``create_all`` installs,
    so an upgraded file and a fresh one are indistinguishable
    (``tests/lite`` compares their ``PRAGMA table_info``).

    Idempotent: a crash between the DDL and the version stamp re-enters with
    the columns already present, and SQLite has no ``ADD COLUMN IF NOT EXISTS``
    — hence the per-column inspect. A later boot never re-runs the step.
    """
    insp = sa_inspect(sync_conn)
    have = {col["name"] for col in insp.get_columns("memory_suppressions")}
    if "namespace" not in have:
        sync_conn.exec_driver_sql(
            "ALTER TABLE memory_suppressions ADD COLUMN namespace VARCHAR(32)"
        )
    if "content_hash" not in have:
        sync_conn.exec_driver_sql(
            "ALTER TABLE memory_suppressions ADD COLUMN content_hash VARCHAR(64)"
        )


def _upgrade_v6_to_v7(sync_conn) -> None:
    """v6 -> v7: the opt-in retention settings on ``users`` (P4b/T6, spec §8.1).

    Runs ONCE, on the version transition, and on a fresh install too (which has
    nothing to back up). ``retention_enabled`` is NOT NULL with a constant
    default, so the ADD COLUMN IS the backfill: every pre-existing user is OFF —
    auto expiration is opt-in, and a migration must never turn it on for a user
    who did not ask. ``retention_days`` is NULLABLE with no default: "no window
    chosen" is a real state, and 0 would be a window that expires everything.

    The DDL mirrors the model byte for byte: ``Boolean()`` renders as
    ``BOOLEAN`` with ``DEFAULT '0'`` and ``Integer()`` as ``INTEGER``, the exact
    types ``create_all`` installs, so an upgraded file and a fresh one are
    indistinguishable (``tests/lite`` compares their ``PRAGMA table_info``).

    Idempotent: a crash between the DDL and the version stamp re-enters with the
    columns already present, and SQLite has no ``ADD COLUMN IF NOT EXISTS`` —
    hence the per-column inspect. A later boot never re-runs the step: a restart
    must not re-assert a setting the user turned on.
    """
    insp = sa_inspect(sync_conn)
    have = {col["name"] for col in insp.get_columns("users")}
    if "retention_enabled" not in have:
        sync_conn.exec_driver_sql(
            "ALTER TABLE users ADD COLUMN retention_enabled BOOLEAN NOT NULL DEFAULT '0'"
        )
    if "retention_days" not in have:
        sync_conn.exec_driver_sql(
            "ALTER TABLE users ADD COLUMN retention_days INTEGER"
        )


def _upgrade_v7_to_v8(sync_conn) -> None:
    """v7 -> v8: the freshness barrier's count index on ``index_outbox``.

    Runs ONCE, on the version transition, and on a fresh install too (which has
    nothing to back up). Pure additive index DDL, spelled exactly as
    ``create_all`` installs it, so an upgraded file and a fresh one are
    indistinguishable.

    Why it exists: the barrier counts this tenant's ``kind='memory'`` intents
    grouped by ``status`` on every poll (once per recall on the happy path, up
    to ~40 times while a write is pending) and the outbox is never pruned
    (no retention policy in P3), so without this index each count was a full
    scan of a monotonically growing table — measured 0.86 / 22.5 / 114.9 ms at
    5k / 100k / 500k rows.

    Idempotent: ``IF NOT EXISTS``, so a crash between the DDL and the version
    stamp resumes instead of failing on a duplicate index. A later boot never
    re-runs it (it is gated on the starting version), and re-creating an index
    an operator dropped is not a repair this ladder performs.
    """
    sync_conn.exec_driver_sql(
        "CREATE INDEX IF NOT EXISTS ix_index_outbox_kind_tenant_status"
        " ON index_outbox(kind, tenant_id, status)"
    )


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
    repaired), then v4 -> v5 (the P4a namespace column + its index, ONCE — the
    column default backfills every existing row with ``'personal'``), then
    v5 -> v6 (the P4b ledger columns on ``memory_suppressions``, ONCE — nullable
    and never backfilled: NULL is the honest value for a row that predates them),
    then v6 -> v7 (the P4b opt-in retention settings on ``users``, ONCE — the
    NOT NULL ``retention_enabled`` default backfills every existing user OFF,
    which is the only value a migration may choose for it), then v7 -> v8 (the
    P3 freshness barrier's count index on ``index_outbox``, ONCE — additive
    index DDL, the exact shape ``create_all`` installs).
    Divergence fails closed — ``create_all`` is never used as an
    existing-schema migration mechanism.
    """
    from app import models  # noqa: F401 — register every model on Base

    path = _conn_sqlite_path(conn)
    version = int(conn.execute(text("PRAGMA user_version")).scalar_one())
    tables = set(sa_inspect(conn).get_table_names())
    fresh_install = version == 0 and not tables
    # Every version below the terminal, plus the terminal itself: a file NEWER
    # than this binary's vocabulary is a refusal, never a repair. Spelled from
    # the constant — a pinned constant (the gate's stand-in for an older
    # binary) must narrow the vocabulary exactly like a real older binary did,
    # which a literal list would silently stop doing at the next bump.
    if version not in (*range(SQLITE_SCHEMA_VERSION), SQLITE_SCHEMA_VERSION):
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
    if version in (0, 1) and tables and SQLITE_SCHEMA_VERSION >= 2:
        _backup_before_ddl(path)
        _upgrade_v1_to_v2(conn)
        _check_sqlite_foreign_keys(conn)
    if version in (0, 1, 2) and tables and SQLITE_SCHEMA_VERSION >= 3:
        # P1b milestone backup (R24): reused if present, renamed if the
        # v1 -> v2 step just produced the pre-ladder file, else a fresh
        # consistent snapshot. Never overwritten.
        _p1b_backup(path)
    if version in (0, 1, 2) and SQLITE_SCHEMA_VERSION >= 3:
        # The v2 -> v3 DATA step, ONCE, on the transition: the two real rows.
        # A fresh install (no tables before this call) may have them active —
        # nothing to serve yet. An UPGRADE must not: the old pointer keeps
        # serving — loud (a contract mismatch the guard raises) when that
        # pointer is the masked-mean P1a row, empty results when there is no
        # active row at all; `cutover` flips it either way.
        # A later boot (already v3) never re-asserts the pointer (that would
        # undo a rollback) and never re-mutates the manifest.
        _upgrade_v2_to_v3(conn, activate=fresh_install)
    if version in (0, 1, 2, 3) and SQLITE_SCHEMA_VERSION >= 4:
        # P2 milestone backup (R18): its OWN name, taken where the ladder stands
        # now (v3, pre-FTS) — nothing to inherit from the P1b file, so no
        # rename. A fresh install has nothing to back up.
        if tables:
            _backup_before_ddl(path, suffix="pre-p2")
        # The v3 -> v4 DDL step, ONCE, on the transition; a fresh install runs
        # it too, so every v4 install serves a lexical leg.
        _upgrade_v3_to_v4(conn)
    if version in (0, 1, 2, 3, 4) and SQLITE_SCHEMA_VERSION >= 5:
        # P4a milestone backup: its OWN name, taken where the ladder stands now
        # (v4, pre-namespace). The P1b/P2 files are snapshots of EARLIER states —
        # nothing to inherit, so no rename — and are never overwritten. A fresh
        # install has nothing to back up.
        if tables:
            _backup_before_ddl(path, suffix="pre-p4")
        # The v4 -> v5 DDL step, ONCE, on the transition; a fresh install runs it
        # too, so every v5 install carries the column the ACL predicates read.
        _upgrade_v4_to_v5(conn)
    if version in (0, 1, 2, 3, 4, 5) and SQLITE_SCHEMA_VERSION >= 6:
        # P4b milestone backup: its OWN name, taken where the ladder stands now
        # (v5, pre-suppression-columns). The earlier files are snapshots of
        # EARLIER states — nothing to inherit, so no rename — and are never
        # overwritten. A fresh install has nothing to back up.
        if tables:
            _backup_before_ddl(path, suffix="pre-p4b")
        # The v5 -> v6 DDL step, ONCE, on the transition; a fresh install runs
        # it too, so every v6 install carries the ledger columns (nullable —
        # NULL is the honest value for a row that predates them).
        _upgrade_v5_to_v6(conn)
    if version in (0, 1, 2, 3, 4, 5, 6) and SQLITE_SCHEMA_VERSION >= 7:
        # T6 milestone backup: its OWN name, taken where the ladder stands now
        # (v6, pre-retention-settings). The earlier files are snapshots of
        # EARLIER states — nothing to inherit, so no rename — and are never
        # overwritten. A fresh install has nothing to back up.
        if tables:
            _backup_before_ddl(path, suffix="pre-retention")
        # The v6 -> v7 DDL step, ONCE, on the transition; a fresh install runs
        # it too, so every v7 install carries the retention settings the
        # service reads — OFF for every pre-existing user (the column default
        # IS the backfill: spec §8.1 never lets a migration opt a user in).
        _upgrade_v6_to_v7(conn)
    if version in (0, 1, 2, 3, 4, 5, 6, 7) and SQLITE_SCHEMA_VERSION >= 8:
        # P3-freshness milestone backup: its OWN name, taken where the ladder
        # stands now (v7, pre-count-index). The earlier files are snapshots of
        # EARLIER states — nothing to inherit, so no rename — and are never
        # overwritten. A fresh install has nothing to back up.
        if tables:
            _backup_before_ddl(path, suffix="pre-freshness-index")
        # The v7 -> v8 DDL step, ONCE, on the transition; a fresh install runs
        # it too, so every v8 install serves the barrier's count from an index.
        _upgrade_v7_to_v8(conn)
    conn.execute(text(f"PRAGMA user_version = {SQLITE_SCHEMA_VERSION}"))
    integrity = conn.exec_driver_sql("PRAGMA integrity_check").fetchone()
    if integrity is None or integrity[0] != "ok":
        raise RuntimeError("SQLite integrity_check failed after schema migration")


async def bootstrap_sqlite() -> None:
    """Create or upgrade the canonical SQLite schema.

    Thin async wrapper over :func:`upgrade_sqlite_schema` (the one ladder
    definition shared with the migration CLI). It deliberately does NOTHING
    else: a process whose terminal version is older than this file's ladder
    must still be able to boot against an older database and stamp its own
    version (tests/lite/test_sqlite_schema_v6.py). The install's single
    identity is ensured by the app boot (app/main.py), not by the ladder.
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

    sync_url = settings.DATABASE_URL.replace("+aiosqlite", "")
    eng = create_engine(sync_url, connect_args={"check_same_thread": False})

    @event.listens_for(eng, "connect")
    def _enable_sync_sqlite_pragmas(dbapi_connection, _record):
        _configure_sqlite_connection(dbapi_connection, _record)

    return eng


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
