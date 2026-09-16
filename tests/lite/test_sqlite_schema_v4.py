"""SQLite ladder v4 — the FTS5 memory index and its transactional triggers.

Rulings this module pins:

- R3(p2): FTS5 is SQLite-only. There is NO model metadata for the virtual
  table (``create_all`` never sees it) and no Alembic step for Postgres; the
  lexical leg on Postgres reports ``LexicalUnavailable`` — typed, loud, never
  a silent "no matches".
- R16(p2): identity never rides an FTS ``rowid``. ``memory_id``/``user_id``
  are UNINDEXED columns, the triggers maintain the index in the SAME
  transaction as the Memory write, and an UPDATE is delete-by-``memory_id``
  followed by an insert.
- R17(p2): the MATCH expression is parameterized and the user's text is escaped
  into literals (quotes/``*``/NEAR/``-``/operators are text, not grammar), the
  query is bounded, the tenant + ``current_memory_predicate()`` clauses are
  applied BEFORE the LIMIT, and BM25 ascending (lower is better) IS the rank.
- R18(p2): the ladder step is ``_upgrade_v3_to_v4`` — its own ``.pre-p2.bak``
  backup, once-on-transition, fresh installs run it too, ``integrity_check``
  after (the shared tail of ``upgrade_sqlite_schema``).

A v4 install is unservable by a pre-P2 binary, whose ladder tops out at v3 and
refuses to boot the file: the rollback path re-stamps ``user_version = 3`` (and
drops the v4-only objects) — see ``docs/ROLLBACK_P1B.md``.
"""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app import database, models  # noqa: F401 — register every model on Base
from app.database import Base
from app.retrieval.memory import lexical_index

TENANT_A = "a" * 32
TENANT_B = "b" * 32
TENANT_C = "c" * 32
TRIGGERS = {"memories_fts_ai", "memories_fts_au", "memories_fts_ad"}

# The three seeded memories of tenant A: a superseded and a dirty row that match
# the probe term BETTER than the current one, so a post-LIMIT filter (or a
# tenant-blind global top-N) is visible as a wrong answer, not a lucky pass.
SUPERSEDED = "1" * 32
DIRTY = "2" * 32
CURRENT = "3" * 32


async def _engine(tmp_path: Path, name: str):
    eng = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / name}", poolclass=NullPool)
    event.listen(eng.sync_engine, "connect", database._configure_sqlite_connection)
    return eng


async def _schema(conn) -> tuple[int, set[str]]:
    version = int((await conn.execute(text("PRAGMA user_version"))).scalar_one())
    tables = {r[0] for r in (await conn.execute(
        text("SELECT name FROM sqlite_master WHERE type='table'"))).all()}
    return version, tables


async def _triggers(conn) -> set[str]:
    return {r[0] for r in (await conn.execute(
        text("SELECT name FROM sqlite_master WHERE type='trigger'"))).all()}


async def _count(conn, sql: str, **params) -> int:
    return int((await conn.execute(text(sql), params or None)).scalar_one())


async def _indexed_count(conn) -> int:
    return await _count(conn, "SELECT count(*) FROM memory_fts")


async def _canonical_count(conn) -> int:
    return await _count(conn, "SELECT count(*) FROM memories")


async def _indexed_ids(conn) -> set[str]:
    return {r[0] for r in (await conn.execute(text("SELECT memory_id FROM memory_fts"))).all()}


async def _canonical_ids(conn) -> set[str]:
    return {r[0] for r in (await conn.execute(text("SELECT id FROM memories"))).all()}


async def _canonical_ids_of(eng) -> set[str]:
    async with eng.connect() as conn:
        return await _canonical_ids(conn)


async def _user(conn, user_id: str) -> None:
    await conn.execute(text(
        "INSERT INTO users (id, email, onboarding_done, is_verified, is_active, is_deleted)"
        " VALUES (:id, :email, 0, 1, 1, 0)"),
        {"id": user_id, "email": f"{user_id}@test.invalid"})


async def _memory(conn, memory_id: str, user_id: str, content: str, *,
                  title: str | None = None, metadata: str = "{}") -> str:
    await conn.execute(text(
        "INSERT INTO memories (id, user_id, title, content, tags, pinned, is_shared,"
        " recall_count, metadata)"
        " VALUES (:id, :uid, :title, :content, '[]', 0, 0, 0, :metadata)"),
        {"id": memory_id, "uid": user_id, "title": title, "content": content,
         "metadata": metadata})
    return memory_id


async def _search(eng, query: str, *, user_id: str, limit: int = 10):
    async with eng.connect() as conn:
        return await conn.run_sync(
            lambda c: lexical_index.search(c, query, user_id=user_id, limit=limit))


async def _rebuild(eng) -> dict:
    async with eng.begin() as conn:
        return await conn.run_sync(lexical_index.rebuild)


async def _coverage(eng) -> dict:
    async with eng.connect() as conn:
        return await conn.run_sync(lexical_index.coverage)


@pytest_asyncio.fixture
async def v3_db(tmp_path, monkeypatch):
    """A real v3 install: the v1+v2 schema with rows, ``user_version`` stamped 3.

    v4 adds no model tables or columns — the FTS5 table is not metadata — so a
    ``create_all`` schema stamped 3 IS the v3 shape, and this is a real database
    file, not a mock.
    """
    eng = await _engine(tmp_path, "v3.sqlite")
    monkeypatch.setattr(database, "engine", eng)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _user(conn, TENANT_A)
        await _user(conn, TENANT_B)
        await _memory(conn, SUPERSEDED, TENANT_A, "alpha alpha alpha alpha alpha",
                      title="superseded", metadata='{"cm_superseded_by": "x"}')
        await _memory(conn, DIRTY, TENANT_A, "alpha alpha alpha alpha",
                      title="dirty", metadata='{"cm_derived_dirty": true}')
        await _memory(conn, CURRENT, TENANT_A, "alpha", title="current")
        await _memory(conn, "4" * 32, TENANT_B, "alpha alpha alpha alpha alpha alpha",
                      title="foreign tenant")
        await conn.execute(text("PRAGMA user_version = 3"))
    yield eng, tmp_path
    await eng.dispose()


# ── the ladder: v3 -> v4 ────────────────────────────────────────────────────


async def test_v3_install_upgrades_to_v4_with_the_p2_backup_and_a_populated_index(v3_db):
    eng, tmp_path = v3_db
    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version, tables = await _schema(conn)
        triggers = await _triggers(conn)
        indexed, canonical = await _indexed_count(conn), await _canonical_count(conn)
        ids = await _indexed_ids(conn)
        integrity = (await conn.execute(text("PRAGMA integrity_check"))).scalar_one()

    assert version == database.SQLITE_SCHEMA_VERSION == 5
    assert lexical_index.TABLE in tables
    assert triggers >= TRIGGERS, "the index is maintained by triggers, not by callers"
    assert indexed == canonical == 4 and ids == await _canonical_ids_of(eng)
    assert integrity == "ok"

    hits = await _search(eng, "alpha", user_id=TENANT_A, limit=10)
    assert [hit["memory_id"] for hit in hits] == [str(uuid.UUID(CURRENT))], (
        "the backfill indexed the current row and the visibility clauses held")
    assert hits[0]["rank"] == 0 and hits[0]["score"] < 0

    backups = list(Path(tmp_path).glob("*.pre-p2.bak"))
    assert len(backups) == 1, "exactly one pre-P2 backup"
    assert not list(Path(tmp_path).glob("*.pre-v2.bak"))
    snapshot = sqlite3.connect(backups[0])
    try:
        backup_tables = {r[0] for r in snapshot.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        backup_triggers = {r[0] for r in snapshot.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger'")}
        backup_version = snapshot.execute("PRAGMA user_version").fetchone()[0]
        backup_memories = snapshot.execute("SELECT count(*) FROM memories").fetchone()[0]
        backup_integrity = snapshot.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        snapshot.close()
    assert lexical_index.TABLE not in backup_tables, "the backup predates the v4 DDL"
    assert not TRIGGERS & backup_triggers
    assert backup_version == 3, "the backup is what a pre-P2 binary opens"
    assert backup_memories == 4 and backup_integrity == "ok"


async def test_rebooting_a_v4_install_is_a_no_op(v3_db):
    eng, _ = v3_db
    await database.bootstrap_sqlite()
    async with eng.connect() as conn:
        before = (await _indexed_count(conn), await _canonical_count(conn))

    await database.bootstrap_sqlite()
    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version, _tables = await _schema(conn)
        assert (await _indexed_count(conn), await _canonical_count(conn)) == before
    assert version == 5


async def test_a_later_boot_never_re_asserts_the_index(v3_db):
    """Once-on-transition (R18): a restart must not re-run the v4 step.

    Rebuilding on every boot would silently undo an operator's repair target or
    mask drift the coverage report is supposed to surface — the same reason the
    v2 -> v3 step never re-asserts the generation pointer.
    """
    eng, _ = v3_db
    await database.bootstrap_sqlite()
    async with eng.begin() as conn:
        await conn.execute(text("DELETE FROM memory_fts"))  # drift, deliberately

    await database.bootstrap_sqlite()
    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version, _tables = await _schema(conn)
        assert await _indexed_count(conn) == 0, "a v4 boot does not rebuild"
    assert version == 5


async def test_a_crashed_v4_step_resumes_and_never_duplicates(v3_db):
    """The DDL and the stamp are one transaction, but a kill can land between
    runs: v3 + the index present (empty, or already backfilled) must resume."""
    eng, _ = v3_db
    async with eng.begin() as conn:
        await conn.run_sync(lexical_index.create_index)  # the DDL landed, then: dead

    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        version, _tables = await _schema(conn)
        assert await _indexed_count(conn) == await _canonical_count(conn) == 4, (
            "the step resumed and backfilled the empty index")
    assert version == 5

    async with eng.begin() as conn:  # the same, one step further: killed after the backfill
        await conn.execute(text("PRAGMA user_version = 3"))
    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        assert await _indexed_count(conn) == await _canonical_count(conn) == 4, (
            "a coverage-driven backfill never duplicates rows")
        assert await _indexed_ids(conn) == await _canonical_ids(conn)


async def test_fresh_install_runs_the_v4_step_without_a_backup(tmp_path, monkeypatch):
    eng = await _engine(tmp_path, "fresh.sqlite")
    monkeypatch.setattr(database, "engine", eng)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    try:
        await database.bootstrap_sqlite()

        async with eng.connect() as conn:
            version, tables = await _schema(conn)
            triggers = await _triggers(conn)
        assert version == 5 and lexical_index.TABLE in tables
        assert triggers >= TRIGGERS
        assert not list(Path(tmp_path).glob("*.pre-p2.bak")), "nothing to back up yet"
        assert not list(Path(tmp_path).glob("*.pre-p1b.bak"))
    finally:
        await eng.dispose()


# ── triggers: parity with the canonical rows, in the same transaction ───────


async def test_triggers_keep_the_index_in_the_writing_transaction(v3_db):
    """R16: the FTS row exists inside the writer's transaction, and dies with it."""
    eng, _ = v3_db
    await database.bootstrap_sqlite()

    async with eng.connect() as conn:
        await _user(conn, TENANT_C)
        await _memory(conn, "5" * 32, TENANT_C, "transactional text")
        assert "5" * 32 in await _indexed_ids(conn), "visible to the writer, pre-commit"
        await conn.rollback()

    async with eng.connect() as conn:
        assert "5" * 32 not in await _indexed_ids(conn), "and rolled back with the write"
        assert await _indexed_count(conn) == await _canonical_count(conn) == 4


async def test_insert_update_delete_keep_the_index_at_parity(v3_db):
    eng, _ = v3_db
    await database.bootstrap_sqlite()
    await _search(eng, "alpha", user_id=TENANT_A, limit=10)

    async with eng.begin() as conn:
        await _user(conn, TENANT_C)
        await _memory(conn, "6" * 32, TENANT_C, "lorem ipsum dolor")
    async with eng.connect() as conn:
        assert await _indexed_count(conn) == await _canonical_count(conn) == 5

    assert [hit["memory_id"] for hit in await _search(eng, "lorem", user_id=TENANT_C)] == \
        [str(uuid.UUID("6" * 32))]

    async with eng.begin() as conn:
        await conn.execute(text("UPDATE memories SET content = 'dolor sit amet' WHERE id = :id"),
                           {"id": "6" * 32})
    assert await _search(eng, "lorem", user_id=TENANT_C) == [], "the old text is gone"
    assert [hit["memory_id"] for hit in await _search(eng, "amet", user_id=TENANT_C)] == \
        [str(uuid.UUID("6" * 32))]
    async with eng.connect() as conn:
        assert await _indexed_count(conn) == await _canonical_count(conn) == 5, (
            "the update replaced the row, it did not duplicate it")

    async with eng.begin() as conn:
        await conn.execute(text("DELETE FROM memories WHERE id = :id"), {"id": "6" * 32})
    async with eng.connect() as conn:
        assert await _indexed_count(conn) == await _canonical_count(conn) == 4
        assert await _indexed_ids(conn) == await _canonical_ids(conn)
    assert await _search(eng, "amet", user_id=TENANT_C) == []


async def test_identical_text_different_uuids_stay_two_rows(v3_db):
    """No content/parent dedupe: two memories are two memories (spec §7.4)."""
    eng, _ = v3_db
    await database.bootstrap_sqlite()
    first, second = "7" * 32, "8" * 32
    async with eng.begin() as conn:
        await _memory(conn, first, TENANT_A, "duplicate text here")
        await _memory(conn, second, TENANT_A, "duplicate text here")

    hits = await _search(eng, "duplicate", user_id=TENANT_A, limit=10)
    assert {hit["memory_id"] for hit in hits} == {str(uuid.UUID(first)), str(uuid.UUID(second))}
    assert len(hits) == 2 and {hit["rank"] for hit in hits} == {0, 1}


async def test_rebuild_is_the_repair_path_and_reports_the_drift(v3_db):
    eng, _ = v3_db
    await database.bootstrap_sqlite()
    assert (await _rebuild(eng))["rebuilt"] is False, "a healthy index is left alone"

    async with eng.begin() as conn:
        await conn.execute(text(
            "DELETE FROM memory_fts WHERE memory_id = (SELECT min(memory_id) FROM memory_fts)"))
    report = await _rebuild(eng)
    assert report["rebuilt"] is True
    assert (report["canonical"], report["indexed"], report["missing"], report["orphan"]) == \
        (4, 3, 1, 0)
    async with eng.connect() as conn:
        assert await _indexed_ids(conn) == await _canonical_ids(conn)
    assert [hit["memory_id"] for hit in await _search(eng, "alpha", user_id=TENANT_A)] == \
        [str(uuid.UUID(CURRENT))]

    async with eng.begin() as conn:  # an orphan (no canonical row) is drift too
        await conn.execute(text(
            "INSERT INTO memory_fts (title, content, memory_id, user_id)"
            " VALUES ('orphan', 'orphan text', 'ffffffffffffffffffffffffffffffff', :u)"),
            {"u": TENANT_A})
    report = await _rebuild(eng)
    assert (report["indexed"], report["missing"], report["orphan"], report["rebuilt"]) == \
        (5, 0, 1, True)
    assert (await _rebuild(eng))["rebuilt"] is False


async def test_rebuild_repairs_duplicate_only_drift(v3_db):
    """I1: canonical 4 / indexed 5 with CLEAN set differences is still drift.

    ``missing`` and ``orphan`` are both 0 here, so the set checks alone pass the
    duplicate through; only the count comparison sees it. Left in place it makes
    ``search`` return the same memory twice (the join has no DISTINCT), so the
    backfill must fire on the counts too.
    """
    eng, _ = v3_db
    await database.bootstrap_sqlite()

    async with eng.begin() as conn:  # a write that bypassed the triggers
        await conn.execute(text(
            "INSERT INTO memory_fts (title, content, memory_id, user_id)"
            " SELECT title, content, memory_id, user_id FROM memory_fts"
            " WHERE memory_id = :id"), {"id": CURRENT})

    assert await _coverage(eng) == \
        {"canonical": 4, "indexed": 5, "missing": 0, "orphan": 0}

    report = await _rebuild(eng)
    assert (report["canonical"], report["indexed"], report["missing"], report["orphan"]) == \
        (4, 5, 0, 0), "the report names the pre-repair counts"
    assert report["rebuilt"] is True, "a duplicate is drift the counts see"
    assert report["indexed_after"] == 4

    assert await _coverage(eng) == \
        {"canonical": 4, "indexed": 4, "missing": 0, "orphan": 0}, "repaired"
    assert [hit["memory_id"] for hit in await _search(eng, "alpha", user_id=TENANT_A)] == \
        [str(uuid.UUID(CURRENT))], "the memory is returned once, not twice"
    assert (await _rebuild(eng))["rebuilt"] is False, "healthy again"


# ── the query: literal escaping, the query budget, and the clauses before LIMIT ──


@pytest.mark.parametrize(("query", "expected"), [
    ('alpha "beta"', '"alpha" "beta"'),
    ("hello*", '"hello"'),
    ("a NEAR b", '"a" "NEAR" "b"'),
    ("title:secret", '"title" "secret"'),
    ("-alpha", '"alpha"'),
    ("alpha OR beta", '"alpha" "OR" "beta"'),
    ("a^2 (b)", '"a" "2" "b"'),
    ("", ""),
    ("!!! *** \"", ""),
])
def test_match_expression_escapes_user_text_into_literals(query, expected):
    assert lexical_index.match_expression(query) == expected


async def test_search_never_parses_user_text_as_fts_grammar(v3_db):
    eng, _ = v3_db
    await database.bootstrap_sqlite()
    nasty = "9" * 32
    async with eng.begin() as conn:
        await _memory(conn, nasty, TENANT_A, "alpha OR NEAR beta gamma-delta")

    def ids(hits):
        return [hit["memory_id"] for hit in hits]

    # A SHAPE pin, not the discriminator: raw, `alpha " OR "` is valid FTS5 too —
    # the quoted phrase is just the token `or`, which this row happens to contain.
    # The probes whose raw forms are a syntax/column error are what make this
    # discriminate: `NEAR(` and `beta -gamma` below, `-alpha` / `a^2 (b)` above.
    assert ids(await _search(eng, 'alpha \" OR \"', user_id=TENANT_A)) == [str(uuid.UUID(nasty))], (
        "as literals it is \"alpha\" AND \"OR\"")
    assert ids(await _search(eng, "NEAR(", user_id=TENANT_A)) == [str(uuid.UUID(nasty))], (
        "NEAR is a word, not the proximity operator")
    assert ids(await _search(eng, "beta -gamma", user_id=TENANT_A)) == [str(uuid.UUID(nasty))], (
        "`-` is a separator, not a NOT")
    assert ids(await _search(eng, "gamma-delta", user_id=TENANT_A)) == [str(uuid.UUID(nasty))], (
        "a dashed identifier is a literal phrase, not an operator")
    assert ids(await _search(eng, "alpha'^beta", user_id=TENANT_A)) == [str(uuid.UUID(nasty))]
    assert ids(await _search(eng, "alpha*", user_id=TENANT_A)) == \
        ids(await _search(eng, "alpha", user_id=TENANT_A)), (
        "`*` is not a prefix operator: the same rows, in the same rank order as the bare word")
    assert await _search(eng, "*", user_id=TENANT_A) == [], "no tokens, no MATCH"
    assert await _search(eng, "!!! ***", user_id=TENANT_A) == []
    assert await _search(eng, "a NEAR b", user_id=TENANT_A) == []
    assert await _search(eng, "alpha", user_id=TENANT_A, limit=0) == [], "a zero limit is a zero"


async def test_the_query_budget_bounds_the_match_expression(v3_db):
    eng, _ = v3_db
    await database.bootstrap_sqlite()
    long_query = " ".join(f"tok{i}" for i in range(lexical_index.MAX_QUERY_TOKENS + 50))
    expression = lexical_index.match_expression(long_query)
    assert expression.count('"') == lexical_index.MAX_QUERY_TOKENS * 2
    assert len(lexical_index.match_expression("x" * 10_000)) <= \
        lexical_index.MAX_QUERY_CHARS + 2
    assert await _search(eng, "x" * 10_000, user_id=TENANT_A) == [], "bounded, not a crash"


async def test_tenant_and_visibility_are_applied_before_the_limit(v3_db):
    """A better-scoring foreign/superseded/dirty row must never take the slot."""
    eng, _ = v3_db
    await database.bootstrap_sqlite()

    hits = await _search(eng, "alpha", user_id=TENANT_A, limit=1)
    assert [hit["memory_id"] for hit in hits] == [str(uuid.UUID(CURRENT))], (
        "the superseded and dirty rows outrank the current one on BM25 and are "
        "excluded BEFORE the LIMIT; the foreign tenant's row never enters at all")

    assert [hit["memory_id"] for hit in await _search(eng, "alpha", user_id=TENANT_A, limit=10)] == \
        [str(uuid.UUID(CURRENT))]

    foreign = await _search(eng, "alpha", user_id=TENANT_B, limit=10)
    assert [hit["memory_id"] for hit in foreign] == [str(uuid.UUID("4" * 32))]


async def test_bm25_ascending_is_the_lexical_rank(v3_db):
    eng, _ = v3_db
    await database.bootstrap_sqlite()
    weak, strong = "a" * 31 + "0", "a" * 31 + "1"
    async with eng.begin() as conn:
        await _memory(conn, weak, TENANT_A, "beta")
        await _memory(conn, strong, TENANT_A, "beta beta beta beta")

    hits = await _search(eng, "beta", user_id=TENANT_A, limit=10)
    assert [hit["memory_id"] for hit in hits] == [str(uuid.UUID(strong)), str(uuid.UUID(weak))]
    assert [hit["rank"] for hit in hits] == [0, 1]
    assert hits[0]["score"] < hits[1]["score"], "lower BM25 is better"


# ── Postgres: explicitly unavailable, never silently empty (R3) ─────────────


class _NonSqliteConnection:
    """A connection the guard must answer WITHOUT touching (no server in tests)."""

    dialect = SimpleNamespace(name="postgresql")

    def execute(self, *_args, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError("the lexical leg must not run SQL off SQLite")

    def exec_driver_sql(self, *_args, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError("the lexical leg must not run SQL off SQLite")


def test_postgres_reports_unavailable_and_never_queries():
    conn = _NonSqliteConnection()
    assert issubclass(lexical_index.LexicalUnavailable, RuntimeError)
    assert lexical_index.is_available(conn) is False
    with pytest.raises(lexical_index.LexicalUnavailable, match="SQLite"):
        lexical_index.search(conn, "alpha", user_id=TENANT_A, limit=10)
    with pytest.raises(lexical_index.LexicalUnavailable, match="SQLite"):
        lexical_index.rebuild(conn)


async def test_is_available_is_false_without_the_v4_ladder(tmp_path):
    """A pre-v4 SQLite file (no ladder run) has no index — reported, not crashed."""
    eng = await _engine(tmp_path, "unladdered.sqlite")
    try:
        async with eng.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with eng.connect() as conn:
            assert await conn.run_sync(lexical_index.is_available) is False
    finally:
        await eng.dispose()
