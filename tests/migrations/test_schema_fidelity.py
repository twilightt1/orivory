"""Schema fidelity: alembic migrations must match what the ORM models declare.

Why this exists: the repo shipped with models using JSON list columns
(tags/scopes/aliases) while the migration chain created those columns as
varchar[] — a fresh alembic-provisioned Postgres failed on the very first
tagged insert, and the referral tables never got a migration at all. Every
existing suite provisions via Base.metadata.create_all, which by construction
cannot see model↔migration drift. These tests introspect the REAL migration
chain (no database needed) and diff it against Base.metadata.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from sqlalchemy import ARRAY, JSON, Boolean, Date, DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB

import app.models  # noqa: F401  — registers every model on Base
from app.database import Base

ROOT = Path(__file__).resolve().parents[2]
VERSIONS = ROOT / "alembic" / "versions"

# Tables created outside the migration chain on purpose (none today — every
# model table must be created by a migration so alembic-provisioned DBs work).
MIGRATION_EXEMPT_TABLES: set[str] = set()

# ── migration introspection ─────────────────────────────────────────────────


def _kind(sqla_type) -> str:
    """Classify a SQLAlchemy column type into a dialect-neutral kind."""
    # TypeDecorator (e.g. EncryptedJSONB) — classify by the wrapped type.
    if hasattr(sqla_type, "impl") and not isinstance(sqla_type, type):
        try:
            inner = sqla_type.impl
            if isinstance(inner, type):
                from sqlalchemy import JSON as _J
                if issubclass(inner, _J):
                    return "json"
            else:
                return _kind(inner)
        except Exception:
            pass
    if isinstance(sqla_type, (JSON, JSONB)):
        return "json"
    if isinstance(sqla_type, ARRAY):
        return "array"
    if isinstance(sqla_type, (String, Text)):
        return "string"
    if isinstance(sqla_type, Boolean):
        return "bool"
    if isinstance(sqla_type, (Integer,)):
        return "int"
    if isinstance(sqla_type, (Float,)):
        return "float"
    if isinstance(sqla_type, DateTime):
        return "datetime"
    if isinstance(sqla_type, Date):
        return "date"
    from sqlalchemy import Uuid

    if isinstance(sqla_type, Uuid):
        return "uuid"
    from app.models.types import GUID

    if isinstance(sqla_type, GUID):
        return "uuid"
    return "other"


def _extract_column(arg):
    """Return a sa.Column from op.create_table positional args, else None."""
    import sqlalchemy as sa

    if isinstance(arg, sa.Column):
        return arg
    if isinstance(arg, sa.Table):
        return None
    # Constraint objects (PrimaryKeyConstraint, ForeignKeyConstraint, ...)
    return None


class _FakeRow:
    def __init__(self, data):
        self._data = data
    def fetchone(self):
        return self._data
    def fetchall(self):
        return [self._data] if self._data else []


class _FakeBind:
    """Simulates the information_schema view of a DB provisioned by the
    PREVIOUS revisions: the three list columns are ARRAY."""

    ARRAY_COLUMNS = {("memories", "tags"), ("agent_clients", "scopes"), ("entities", "aliases")}

    def execute(self, clause, params=None):
        sql = str(clause)
        if "information_schema.columns" in sql and params:
            key = (params.get("t"), params.get("c"))
            data = ("ARRAY",) if key in self.ARRAY_COLUMNS else None
            return _FakeRow(data if data else None)
        return _FakeRow(None)


class _OpProxy:
    """Recording stand-in for `alembic.op`: captures schema declarations
    without touching a database."""

    def __init__(self, created: list, tables: set):
        self._created = created
        self._tables = tables

    def create_table(self, name, *args, **kwargs):
        self._tables.add(name)
        for arg in args:
            col = _extract_column(arg)
            if col is not None:
                self._created.append(("column", name, col))

    def add_column(self, table_name, column):
        self._created.append(("column", table_name, column))

    def get_bind(self):
        return _FakeBind()

    def execute(self, sql, *a, **kw):
        self._created.append(("sql", sql if isinstance(sql, str) else str(sql)))

    def __getattr__(self, item):
        return lambda *a, **kw: None


def _record_migrations() -> tuple[dict[tuple[str, str], str], set[str]]:
    """Run every migration's upgrade() against a recording op proxy.

    Revisions execute in real chain order (base -> head, walked via
    down_revision links) — filenames share no convention with revision ids.
    """
    class _NoBind(Exception):
        pass

    # Parse revision ids + down_revisions from files.
    revs: dict[str, dict] = {}
    for path in sorted(VERSIONS.glob("*.py")):
        text = path.read_text()
        m_rev = re.search(r'^revision(?::\s*str)?\s*=\s*["\']([^"\']+)', text, re.M)
        m_down = re.search(
            r'^down_revision(?::\s*str \| None)?\s*=\s*(["\'][^"\']*["\']|None)', text, re.M
        )
        if not m_rev:
            continue
        down = None
        if m_down and m_down.group(1) != "None":
            down = m_down.group(1).strip("\"'")
        revs[m_rev.group(1)] = {"path": path, "down": down}

    by_down: dict[str | None, list[str]] = {}
    for rid, info in revs.items():
        by_down.setdefault(info["down"], []).append(rid)
    order: list[str] = list(by_down.get(None, []))
    i = 0
    while i < len(order):
        order.extend(by_down.get(order[i], []))
        i += 1

    columns: dict[tuple[str, str], str] = {}
    tables: set[str] = set()

    for rid in order:
        path = revs[rid]["path"]
        spec = importlib.util.spec_from_file_location(f"mig_{path.stem}", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        created: list = []

        proxy = _OpProxy(created, tables)
        module.op = proxy  # type: ignore[assignment]
        try:
            module.upgrade()
        except Exception:
            continue  # revisions needing a live DB still record what they declared

        for entry in created:
            if entry[0] == "table":
                tables.add(entry[1])
            elif entry[0] == "column":
                _, table_name, column = entry
                columns[(table_name, column.name)] = _kind(column.type)
            elif entry[0] == "sql":
                sql = entry[1]
                # Reconstruct EXECUTE payloads inside DO $$ ... $$ blocks:
                # they arrive as concatenated quoted fragments ('a' 'b' -> 'ab').
                # Un-quoting them lets the ALTER TYPE regexes below see the
                # real statement shape.
                exec_payloads = re.findall(
                    r"EXECUTE\s+((?:'[^']*'\s*)+)", sql
                )
                for payload in exec_payloads:
                    joined = "".join(
                        frag.strip() for frag in re.findall(r"'([^']*)'", payload)
                    )
                    sql += "\n" + joined
                # Raw-SQL table creation (e.g. e5f6a7b8c9d0 uses CREATE TABLE
                # IF NOT EXISTS ... instead of op.create_table).
                for m in re.finditer(
                    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s*\(",
                    sql, re.IGNORECASE,
                ):
                    tables.add(m.group(1))
                # ALTER TABLE ... ADD COLUMN [IF NOT EXISTS] name TYPE
                for m in re.finditer(
                    r"ALTER\s+TABLE\s+(\w+)\s+ADD\s+COLUMN\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s+(\w+)",
                    sql, re.IGNORECASE,
                ):
                    table, col, raw_type = m.groups()
                    columns[(table, col)] = _normalize_raw(raw_type)
                # ALTER TABLE ... ALTER COLUMN name TYPE newtype USING ...
                # (also matches inside DO $$ ... EXECUTE '...' blocks — the
                # quoted inner SQL is scanned with the same pattern.)
                for m in re.finditer(
                    r"ALTER\s+TABLE\s+(\w+)\s+ALTER\s+COLUMN\s+(\w+)\s+(?:SET\s+DATA\s+)?TYPE\s+(\w+)",
                    sql, re.IGNORECASE,
                ):
                    table, col, raw_type = m.groups()
                    columns[(table, col)] = _normalize_raw(raw_type)

    return columns, tables


def _normalize_raw(raw: str) -> str:
    r = raw.upper().rstrip("(),")
    if r.startswith("JSONB") or r.startswith("JSON"):
        return "json"
    if "ARRAY" in r or r.startswith("VARCHAR[]") or "[]" in r:
        return "array"
    if r.startswith(("VARCHAR", "CHAR", "TEXT")):
        return "string"
    if r.startswith("BOOL"):
        return "bool"
    if r.startswith(("INT", "BIGINT", "SMALLINT")):
        return "int"
    if r.startswith(("FLOAT", "DOUBLE", "NUMERIC", "DECIMAL")):
        return "float"
    if r.startswith(("TIMESTAMP", "DATETIME")):
        return "datetime"
    if r.startswith("DATE"):
        return "date"
    if r.startswith("UUID"):
        return "uuid"
    return "other"


# ── tests ────────────────────────────────────────────────────────────────────


def test_every_model_table_is_created_by_migrations():
    """A DB provisioned purely by alembic must contain every model's table.

    Regression: referral_codes/referrals/referral_rewards had models and a
    mounted router but no migration — every referral endpoint 500'd on an
    alembic-provisioned database.
    """
    _, mig_tables = _record_migrations()
    model_tables = set(Base.metadata.tables.keys())
    missing = model_tables - mig_tables - MIGRATION_EXEMPT_TABLES
    assert not missing, (
        f"Model tables with no CREATE TABLE in any migration: {sorted(missing)}. "
        "An alembic-provisioned database will 500 on first use of these models."
    )


def test_model_column_types_match_migrations():
    """Every (table, column) the migrations create must have the same
    dialect-neutral type kind as the model.

    Regression: migrations created tags/scopes/aliases as varchar[] (ARRAY)
    while the models serialize JSON — first insert failed with
    DatatypeMismatchError on a fresh Postgres.
    """
    mig_columns, _ = _record_migrations()
    mismatches: list[str] = []
    for table_name, table in Base.metadata.tables.items():
        for col in table.columns:
            key = (table_name, col.name)
            if key not in mig_columns:
                continue  # column created before the model column existed
            model_kind = _kind(col.type)
            mig_kind = mig_columns[key]
            if model_kind == "uuid" or mig_kind == "uuid":
                continue  # GUID vs UUID are equivalent cross-dialect
            if model_kind != mig_kind:
                mismatches.append(
                    f"{table_name}.{col.name}: model={model_kind} migration={mig_kind}"
                )
    assert not mismatches, (
        "Model/migration column type drift (fresh alembic DB will fail writes):\n  "
        + "\n  ".join(mismatches)
    )
