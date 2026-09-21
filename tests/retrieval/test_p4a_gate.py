"""P4a acceptance gate — the §9 P4 row (namespace/ACL part), over the REAL stores.

Everything here runs against a real SQLite file and a real embedded Qdrant
folder, through the P1b gate's own harness (``pytest_plugins`` below, reused by
name): a private per-test database, a private embedded-Qdrant folder, the real
cutover install (both manifests ACTIVE) and deterministic unit vectors for the
embedding CONTRACT. Nothing about SQL or the vector store is mocked.

The only substitutions are the seams that are out of process in production, and
they are the P1b/P3 gates' own seams rather than new ones: the embedder
(deterministic unit vectors — no claim here is about embedding QUALITY) and the
recall path's LLM query rewrite (an LLM call, never the claim).

The §9 P4 row, bullet by bullet — one test each:

* cross-user + same-user cross-namespace for every READER (REST list, the
  shared recall, MCP search/list) → test 1;
* ... every WRITER (REST patch/delete/create-with-parent, MCP
  delete/correct/forget) → test 2;
* ... every EXPORT surface (the migration CLI's memory export; its
  ``namespace=None`` audit escape hatch is pinned as the documented residual)
  → test 4;
* token revoke + permission class: an agent token reads its owner's own
  namespace and nothing else, a write-only token cannot read at all, and a
  revoked token dies at ``resolve_principal`` → test 5;
* the rollback compatibility gate: a pre-P4 ladder refuses the v5 file with the
  documented message, and the documented way back (re-stamp ``user_version = 4``
  then roll forward) round-trips over real data → test 6;
* carried from the T1 review: an UNKNOWN ``user_version`` fails closed and
  leaves the file untouched → test 8;
* the runbook's snapshot warning, MEASURED: a one-boot upgrade's
  ``.pre-p4.bak`` already contains the v4 FTS objects at the stamp the install
  stood at (3) — a milestone snapshot's name is not its version stamp → test 7;
* carried from the T3 review: R34's failure mode — a key-less point (a pre-P4
  payload) is deleted by its own delete intent, which deletes by POINT ID and
  never by a namespace-filtered selector → test 9;
* §4.3's cache-key intent, answered HONESTLY: no cache read/write path is live
  in P4a, so no namespace component is owed yet — and the pin goes red the day a
  directly-called producer or reader goes live without one (the scan's own
  ceiling is stated at the test: a meta-programmed reference slips through)
  → test 10;
* the CI pin: the workflow runs this file, wires the P4a suites, keeps the
  Qdrant parity suite wired somewhere, names no path that does not exist, and
  cannot be stood down by `--ignore`/`if:`/`continue-on-error` → test 11.

The gate is FALLIBLE by mutation (recorded in the task report): dropping
``visibility.namespace_predicate``, the MCP ownership check or the export
namespace each turns its bullet red on a copy of the tree.

``TEAM`` is the namespace P4a deliberately cannot create (client input never
sets a namespace): rows outside ``personal`` are seeded directly, the same
technique ``tests/retrieval/test_qdrant_parity.py`` uses.
"""
from __future__ import annotations

import ast
import hashlib
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
import yaml
from httpx import ASGITransport, AsyncClient
from qdrant_client import models as qm
from sqlalchemy import event, text
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

import app.models  # noqa: F401 — register every ORM table on Base
from app import database
from app.database import Base, sync_session
from app.main import app as asgi_app
from app.mcp_hub import tools as hub_tools
from app.mcp_hub.identity import AgentPrincipal, resolve_principal
from app.models.agent_client import AgentClient
from app.models.memory import Memory
from app.retrieval import vector_backend
from app.retrieval.embedding_fingerprint import generation_name
from app.retrieval.memory import namespaces, outbox, vector_store
from app.retrieval.memory import retriever as retriever_module
from app.retrieval.memory.namespaces import PERSONAL
from app.retrieval.memory.retriever import MemoryRetriever
from app.services.agent_token_service import generate_token, hash_token
from app.utils.dependencies import get_current_verified_user
from tests.retrieval.test_p1b_gate import (
    _intents,
    _memory,
    _payloads,
    _vector_for,
)

# The P1b gate's real-store fixtures (``env`` / ``world`` / ``live``), reused
# rather than copied: this gate has to prove the SAME store the migration
# installs, not a second harness that happens to look like it.
pytest_plugins = ["tests.retrieval.test_p1b_gate"]

REPO = Path(__file__).resolve().parents[2]
APP = REPO / "app"
CI_YML = REPO / ".github" / "workflows" / "ci.yml"
CI_STEP_NAME = "Run P4a namespace/ACL gate suites (temp SQLite, no services)"
GATE_MODULE = "tests/retrieval/test_p4a_gate.py"
TEAM = "team"  # the namespace P4a cannot create — seeded directly

# The P4a suites the new CI step must wire (explicit paths: a suite never runs
# unless it is listed) plus the lifecycle suites the P4a recon found unwired.
P4_CI_SUITES = (
    "tests/retrieval/test_p4a_gate.py",
    "tests/retrieval/test_namespace_acl.py",
    "tests/lite/test_sqlite_schema_v5.py",
    "tests/retrieval/test_correction.py",
    "tests/services/test_compression_service.py",
    "tests/api/test_erasure_router.py",
    "tests/api/test_memories_router.py",
    "tests/services/test_import_service.py",
)
# Already wired by the P1a/P3 steps — the pin only requires they stay wired
# somewhere, not re-run in this step (the brief: "nếu chưa wired ở step khác").
P4_ERASURE_SUITES = (
    "tests/services/test_erasure_service.py",
    "tests/services/test_durable_erasure.py",
    "tests/services/test_erasure_reconcile.py",
)
P4A_DOCS = (
    REPO / "docs" / "ARCHITECTURE.md",
    REPO / "docs" / "OPERATIONS_RUNBOOK.md",
    REPO / "docs" / "API.md",
    REPO / "docs" / "ROLLBACK_P1B.md",
)


# ── helpers: the second namespace, and the surfaces over the real store ─────


@pytest_asyncio.fixture
async def namespaced(live):
    """The live P1b install + the second namespace P4a cannot create.

    ``team`` stands in for the namespace P4b brings. Rows outside ``personal``
    are seeded directly (no writer for them exists in P4a) and their vectors go
    through the REAL store writer, so the dense leg really has something to
    leak. The same-account team row carries the SAME text as the caller's own
    row: identical vectors mean nothing but the namespace clause can tell them
    apart.
    """
    async with live.sessions() as db:
        alice_team = _memory(live.alice.id, "alice current", namespace=TEAM)
        bob_team = _memory(live.bob.id, "bob extra", namespace=TEAM)
        db.add_all([alice_team, bob_team])
        await db.commit()
    vector_store.upsert_memories_sync([alice_team, bob_team])
    return SimpleNamespace(**vars(live), alice_team=alice_team, bob_team=bob_team)


@pytest.fixture
def llm_seam(monkeypatch):
    """The recall path's LLM query rewrite — out of process, never the claim."""
    async def _identity(query, context=None):
        return {"rewritten_query": query, "entities": [], "reasoning": None,
                "_fallback_used": False}

    monkeypatch.setattr(retriever_module, "rewrite_query", _identity)


def _auth_client(user) -> AsyncClient:
    """An ASGI client over the real app, authenticated as ``user``."""
    async def _current_user():
        return user

    asgi_app.dependency_overrides[get_current_verified_user] = _current_user
    return AsyncClient(transport=ASGITransport(app=asgi_app), base_url="http://gate")


def _mcp_seams(monkeypatch, principal) -> None:
    """Point the MCP tools at the private stores with ``principal`` in place."""
    monkeypatch.setattr(hub_tools, "_current_principal", lambda: principal)
    monkeypatch.setattr(hub_tools, "_session", database.AsyncSessionLocal)


@pytest.fixture(autouse=True)
def _clear_app_overrides():
    """The ASGI app is module-global: never leave an auth override behind."""
    yield
    asgi_app.dependency_overrides.clear()


# ── 1. readers ───────────────────────────────────────────────────────────────


async def test_readers_serve_only_the_callers_namespace_cross_user_and_cross_namespace(
    namespaced, monkeypatch, llm_seam
):
    """Every reader answers as the caller: same-user cross-namespace AND cross-user.

    Three readers, one store: the REST list, the shared recall (the API's own
    seam), and the MCP search/list pair. The team twin carries the caller's own
    text, and the cross-user row ranks first for its own text — so a reader that
    forgot the boundary has the row right there to leak.
    """
    alice = namespaced.alice

    async with _auth_client(alice) as client:
        listing = (await client.get("/api/v1/memories")).json()

    served = {item["id"] for item in listing["items"]}
    assert str(namespaced.alice_team.id) not in served, "same-user cross-namespace row served"
    assert str(namespaced.bob_current.id) not in served, "cross-user row served"
    assert str(namespaced.alice_current.id) in served
    assert served == {str(namespaced.alice_current.id), str(namespaced.alice_superseded.id),
                     str(namespaced.alice_forgotten.id)}
    assert listing["total"] == 3, "the pagination total carries the same predicate"

    async with namespaced.sessions() as db:
        ranked, _why = await MemoryRetriever(db, alice.id).recall_ids("alice current", top_k=10)
    ranked_ids = {str(memory_id) for memory_id, _score in ranked}
    assert str(namespaced.alice_team.id) not in ranked_ids, "recall ranked the team twin"
    assert str(namespaced.bob_current.id) not in ranked_ids, "recall ranked a foreign tenant"
    assert str(namespaced.alice_current.id) in ranked_ids

    _mcp_seams(monkeypatch, AgentPrincipal(
        user_id=alice.id, agent_client_id=uuid.uuid4(), name="gate",
        scopes=frozenset({"memory:read"}),
    ))
    search = await hub_tools.search_memory("bob current")
    search_ids = {row["id"] for row in search["results"]}
    assert str(namespaced.bob_current.id) not in search_ids, "MCP search served the ranking tenant"
    assert str(namespaced.alice_team.id) not in search_ids
    recent = await hub_tools.list_recent()
    recent_ids = {row["id"] for row in recent["results"]}
    assert str(namespaced.alice_team.id) not in recent_ids
    assert str(namespaced.alice_current.id) in recent_ids
    # The primary-key surfaces check the loaded row (they cannot carry a
    # predicate): the same "not found" a missing id gets, no existence oracle.
    assert await hub_tools.timeline(str(namespaced.alice_team.id)) == {"error": "memory not found"}
    assert await hub_tools.timeline(str(namespaced.bob_current.id)) == {"error": "memory not found"}
    assert (await hub_tools.timeline(str(namespaced.alice_current.id)))["anchor"]["id"] == str(
        namespaced.alice_current.id)


# ── 2. writers ───────────────────────────────────────────────────────────────


async def test_writers_refuse_another_namespace_and_another_users_row(namespaced, monkeypatch):
    """The write surfaces answer "not found" and change nothing outside the namespace."""
    alice = namespaced.alice
    team_id, bob_id = namespaced.alice_team.id, namespaced.bob_current.id

    async with _auth_client(alice) as client:
        patched = await client.patch(f"/api/v1/memories/{team_id}", json={"title": "hit"})
        deleted = await client.delete(f"/api/v1/memories/{team_id}")
        foreign = await client.delete(f"/api/v1/memories/{bob_id}")
        child = await client.post("/api/v1/memories",
                                  json={"content": "child", "parent_id": str(team_id)})

    assert (patched.status_code, deleted.status_code, foreign.status_code) == (404, 404, 404)
    assert child.status_code == 404 and child.json()["detail"] == "Parent memory not found"

    _mcp_seams(monkeypatch, AgentPrincipal(
        user_id=alice.id, agent_client_id=uuid.uuid4(), name="gate",
        scopes=frozenset({"memory:read", "memory:write"}),
    ))
    mcp_delete = await hub_tools.delete_memory(str(team_id))
    mcp_correct = await hub_tools.correct_memory(memory_id=str(team_id), content="a correction")
    mcp_forget = await hub_tools.forget_memory([str(team_id), str(namespaced.bob_extra.id)])

    assert mcp_delete == {"error": "memory not found"}
    assert mcp_correct == {"error": "memory not found"}, (
        "correcting is a write: a row outside the namespace is the same not-found")
    assert mcp_forget["invalidated"] == 0
    assert mcp_forget["skipped"] == 2, "an id outside the namespace is the same not-found as a missing one"

    async with namespaced.sessions() as db:
        for memory_id in (team_id, bob_id, namespaced.bob_extra.id):
            row = await db.get(Memory, memory_id)
            assert row is not None, f"{memory_id} was mutated outside the caller's namespace"
        assert (await db.get(Memory, team_id)).title != "hit"


# ── 4. export ────────────────────────────────────────────────────────────────


async def test_the_export_tooling_exports_one_namespace_only(namespaced):
    """The migration CLI's memory export defaults to ONE namespace, never all.

    An operator export has no caller, so its boundary is the namespace and its
    default is the safe one. The whole-database read is the deliberate
    ``namespace=None`` audit escape hatch (P4b must revisit it when a second
    namespace has its own points: it will then also see team points as
    "orphans").
    """
    cli = namespaced.cli
    with sync_session() as db:
        default_rows = cli.memory_rows(db)
        team_rows = cli.memory_rows(db, namespace=TEAM)
        audit_rows = cli.memory_rows(db, namespace=None)

    team_id = str(namespaced.alice_team.id)
    assert team_id not in default_rows, "the default export swept in another namespace"
    assert str(namespaced.alice_current.id) in default_rows  # the operator's own data stays
    assert str(namespaced.bob_current.id) in default_rows, (
        "an offline export is operator-scoped, not caller-scoped: cross-user rows of the "
        "exported namespace are expected")
    assert set(team_rows) == {team_id, str(namespaced.bob_team.id)}
    assert team_id in audit_rows and str(namespaced.alice_current.id) in audit_rows, (
        "the documented audit escape hatch reads the whole database")


# ── 5. agent tokens ──────────────────────────────────────────────────────────


async def test_an_agent_token_reads_its_owners_namespace_and_dies_on_revoke(namespaced, monkeypatch):
    """A real agent token: resolve_principal → the tools → the real store.

    The token is a real ``agent_clients`` row (sha256 only), resolved through
    the hub's own ``resolve_principal``. Revocation and the scope class are
    enforced there, and the read boundary is enforced again below it — the
    answer must never carry another namespace, before or after the token dies.
    """
    alice = namespaced.alice

    def _token(scopes: list[str]) -> tuple[str, AgentClient]:
        plaintext = generate_token()
        row = AgentClient(user_id=alice.id, name=f"gate-{'-'.join(scopes)}",
                          token_hash=hash_token(plaintext), scopes=scopes)
        return plaintext, row

    read_token, read_client = _token(["memory:read"])
    write_token, write_client = _token(["memory:write"])
    async with namespaced.sessions() as db:
        db.add_all([read_client, write_client])
        await db.commit()

    async with namespaced.sessions() as db:
        principal = await resolve_principal(db, read_token)
        assert principal is not None and principal.user_id == alice.id
        assert principal.can_read() and not principal.can_write()
    _mcp_seams(monkeypatch, principal)

    search = await hub_tools.search_memory("alice current")
    search_ids = {row["id"] for row in search["results"]}
    assert str(namespaced.alice_current.id) in search_ids
    assert str(namespaced.alice_team.id) not in search_ids, "the token read another namespace"
    assert str(namespaced.bob_current.id) not in search_ids, "the token read another tenant"
    assert await hub_tools.get_memory(str(namespaced.alice_team.id)) == {"error": "memory not found"}
    assert await hub_tools.get_memory(str(namespaced.bob_current.id)) == {"error": "memory not found"}
    assert (await hub_tools.get_memory(str(namespaced.alice_current.id)))["id"] == str(
        namespaced.alice_current.id)

    # The permission class: write-only is not a reader, and it is not a key to
    # another namespace either.
    async with namespaced.sessions() as db:
        writer = await resolve_principal(db, write_token)
    assert writer is not None and writer.can_write() and not writer.can_read()
    _mcp_seams(monkeypatch, writer)
    assert await hub_tools.search_memory("alice current") == hub_tools.READ_SCOPE_ERROR
    assert await hub_tools.list_recent() == hub_tools.READ_SCOPE_ERROR
    assert await hub_tools.delete_memory(str(namespaced.alice_team.id)) == {"error": "memory not found"}
    async with namespaced.sessions() as db:
        assert await db.get(Memory, namespaced.alice_team.id) is not None

    # Revoked: the token resolves to no principal at all, so every read dies at
    # the identity gate — the boundary does not depend on the store.
    async with namespaced.sessions() as db:
        row = await db.get(AgentClient, read_client.id)
        row.status, row.revoked_at = "revoked", datetime.now(UTC)
        await db.commit()
    async with namespaced.sessions() as db:
        assert await resolve_principal(db, read_token) is None
    _mcp_seams(monkeypatch, None)
    assert await hub_tools.search_memory("alice current") == hub_tools.IDENTITY_ERROR
    assert await hub_tools.list_recent() == hub_tools.IDENTITY_ERROR


# ── 6. rollback compatibility ────────────────────────────────────────────────

_INDEX_NAME = "ix_memories_namespace_user"


async def _engine_for(path: Path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}", poolclass=NullPool)
    event.listen(engine.sync_engine, "connect", database._configure_sqlite_connection)
    return engine


async def _strip_namespace(conn) -> None:
    have = await conn.run_sync(
        lambda c: {col["name"] for col in sa_inspect(c).get_columns("memories")})
    if "namespace" not in have:
        return
    await conn.execute(text(f"DROP INDEX IF EXISTS {_INDEX_NAME}"))
    await conn.execute(text("ALTER TABLE memories DROP COLUMN namespace"))


async def _strip_fts(conn) -> None:
    for trigger in ("memories_fts_ai", "memories_fts_au", "memories_fts_ad"):
        await conn.execute(text(f"DROP TRIGGER IF EXISTS {trigger}"))
    await conn.execute(text("DROP TABLE IF EXISTS memory_fts"))


async def _strip_v2(conn) -> None:
    """Drop what the v1 -> v2 step adds: the revision columns and its tables."""
    for table in ("index_outbox", "index_generations", "memory_suppressions"):
        await conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
    await conn.execute(text("ALTER TABLE memories DROP COLUMN revision"))
    await conn.execute(text("ALTER TABLE document_chunks DROP COLUMN revision"))


@pytest_asyncio.fixture
async def _ladder_install(tmp_path, monkeypatch):
    """A private SQLite file bound in as the app's own engine, for a given stamp.

    The ``create_all`` schema minus the ladder-owned objects IS the pre-ladder
    install this fixture models: ``with_v2=False`` drops what the v1 -> v2 step
    adds (an adopted, unversioned install), the namespace column is always
    dropped (the v5 object), and ``version`` is the ``user_version`` it boots
    with.
    """
    async def _build(name: str, *, version: int, with_v2: bool, rows: bool):
        path = tmp_path / name
        engine = await _engine_for(path)
        monkeypatch.setattr(database, "engine", engine)
        monkeypatch.setattr(database, "IS_SQLITE", True)
        monkeypatch.setattr(database, "SQLITE_SCHEMA_VERSION", 5)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await _strip_fts(conn)  # no ORM model creates it: never there, belt and braces
            await _strip_namespace(conn)
            if not with_v2:
                await _strip_v2(conn)
            if rows:
                await conn.execute(text(
                    "INSERT INTO users (id, email, onboarding_done, is_verified, is_active,"
                    " is_deleted) VALUES (:id, :email, 0, 1, 1, 0)"),
                    {"id": "a" * 32, "email": f"{version}@gate.invalid"})
                await conn.execute(text(
                    "INSERT INTO memories (id, user_id, content, tags, pinned, is_shared,"
                    " recall_count) VALUES (:id, :uid, :content, '[]', 0, 0, 0)"),
                    {"id": "1" * 32, "uid": "a" * 32, "content": f"row at v{version}"})
            await conn.execute(text(f"PRAGMA user_version = {version}"))
        return SimpleNamespace(path=path, engine=engine)

    engines: list = []

    async def _factory(*, version: int, with_v2: bool = True, rows: bool = True):
        name = f"gate-v{version}{'-v2' if with_v2 else '-v1shape'}.sqlite"
        install = await _build(name, version=version, with_v2=with_v2, rows=rows)
        engines.append(install.engine)
        return install

    try:
        yield _factory
    finally:
        for engine in engines:
            await engine.dispose()


@pytest_asyncio.fixture
async def v4_install(_ladder_install):
    """A real v4 install with rows: the current schema minus the v5 column.

    v5 adds no table and no other column, so "create_all then drop the
    namespace column" IS the v4 shape this ladder upgrades (the pattern
    ``tests/lite/test_sqlite_schema_v5.py`` pins in isolation).
    """
    return await _ladder_install(version=4)


def _read(path: Path, sql: str) -> list[tuple]:
    """What a plain ``sqlite3`` reader sees: the committed state on the file."""
    conn = sqlite3.connect(path)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _write(path: Path, sql: str) -> None:
    """A committed single statement on the file (autocommit, like ``sqlite3``)."""
    conn = sqlite3.connect(path)
    try:
        conn.isolation_level = None
        conn.execute(sql)
    finally:
        conn.close()


def _version(path: Path) -> int:
    return int(_read(path, "PRAGMA user_version")[0][0])


def _namespaces(path: Path) -> list[str]:
    return [row[0] for row in _read(path, "SELECT namespace FROM memories ORDER BY id")]


def _has_fts(path: Path) -> bool:
    return any(row[0] == "memory_fts"
               for row in _read(path, "SELECT name FROM sqlite_master WHERE type='table'"))


def _sidecars(directory: Path) -> set[str]:
    return {entry.name for entry in directory.iterdir() if entry.is_file() and ".bak" in entry.name}


async def test_the_pre_p4_ladder_refuses_a_v5_file_and_the_rollback_round_trips(v4_install):
    """§11's rollback checkpoint, on real data.

    A pre-P4 binary's ladder tops out at v4 and REFUSES a v5 file: for a v5 file
    the refusal is the module's version guard and nothing else of that ladder can
    run, so pinning the constant it compared against (`SQLITE_SCHEMA_VERSION` at
    4) reproduces the old binary's refusal faithfully. The documented way back is
    then re-stamping ``user_version = 4`` and rolling forward with a P4a binary —
    which must be additive (the column is inspected, not re-added), must not
    re-assert a namespace an operator moved, and must reuse the operator's
    ``.pre-p4.bak`` snapshot rather than rewrite it.
    """
    path = v4_install.path
    await database.bootstrap_sqlite()
    assert _version(path) == 5

    backup = path.with_name(path.name + ".pre-p4.bak")
    assert backup.is_file(), "the v4 -> v5 step takes its own milestone snapshot"
    before = hashlib.sha256(backup.read_bytes()).hexdigest()

    _write(path, "UPDATE memories SET namespace = 'moved'")  # the operator moved a value
    stat_before = (path.stat().st_size, path.stat().st_mtime_ns)

    with pytest.raises(RuntimeError, match=r"unsupported SQLite schema version 5; expected 4"):
        with pytest.MonkeyPatch.context() as pin:  # the pre-P4 ladder's guard
            pin.setattr(database, "SQLITE_SCHEMA_VERSION", 4)
            await database.bootstrap_sqlite()

    assert _version(path) == 5, "the refusal must not stamp anything"
    assert _namespaces(path) == ["moved"], "the refusal must not touch rows"
    assert (path.stat().st_size, path.stat().st_mtime_ns) == stat_before
    assert hashlib.sha256(backup.read_bytes()).hexdigest() == before

    _write(path, "PRAGMA user_version = 4")  # the documented rollback: re-stamp

    await database.bootstrap_sqlite()  # roll forward

    assert _version(path) == 5
    assert _namespaces(path) == ["moved"], "a later boot re-asserted the namespace"
    assert hashlib.sha256(backup.read_bytes()).hexdigest() == before, (
        "the operator's pre-P4 snapshot was rewritten")


# ── 7. the milestone snapshots (the runbook's warning, measured) ─────────────


async def test_a_milestone_snapshot_is_not_promised_clean(_ladder_install):
    """The runbook's snapshot warning, pinned: the NAME is not the stamp.

    The ladder snapshots mid-run, and SQLite commits DDL immediately while
    `user_version` is stamped only at the very END of the whole run — so a
    one-boot upgrade writes a `.pre-p4.bak` that already CONTAINS the v4 FTS
    objects while its own stamp still reads the version the install stood at
    (3). An adopted unversioned install is the same mechanism one step earlier:
    its snapshots all read 0 while `.pre-p2.bak`/`.pre-p4.bak` already hold the
    v2 tables. Either way: check the stamp AND the objects inside.
    """
    install = await _ladder_install(version=3)
    await database.bootstrap_sqlite()
    assert _version(install.path) == 5

    pre_p2 = install.path.with_name(install.path.name + ".pre-p2.bak")
    pre_p4 = install.path.with_name(install.path.name + ".pre-p4.bak")
    assert pre_p2.is_file() and pre_p4.is_file()
    assert _version(pre_p2) == 3 and not _has_fts(pre_p2)
    assert _version(pre_p4) == 3, (
        "the pre-p4 snapshot carries the stamp of the state it was taken in")
    assert _has_fts(pre_p4), (
        "the v4 DDL was already committed when the pre-p4 snapshot was taken")
    assert "namespace" not in {
        row[1] for row in _read(pre_p4, "PRAGMA table_info(memories)")}

    adopted = await _ladder_install(version=0, with_v2=False)
    await database.bootstrap_sqlite()
    assert _version(adopted.path) == 5
    for suffix in ("pre-p1b", "pre-p2", "pre-p4"):
        snapshot = adopted.path.with_name(adopted.path.name + f".{suffix}.bak")
        assert _version(snapshot) == 0, f"{suffix} is not stamped with the version it stood at"
    tables = {row[0] for row in _read(
        adopted.path.with_name(adopted.path.name + ".pre-p4.bak"),
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "index_outbox" in tables, "the v2 objects were already committed when it was taken"


# ── 8. fail-closed: an unknown schema version (carried from the T1 review) ───


async def test_an_unknown_schema_version_fails_closed_and_leaves_the_file_untouched(env):
    """A version this binary does not know is a refusal, not a repair.

    A v8 file means a NEWER binary's data: the ladder must not boot it, must not
    stamp it back, and must not leave a milestone snapshot behind while
    inspecting it. The file's status quo is the whole assertion.
    """
    await database.bootstrap_sqlite()
    assert _version(env.db_path) == 7

    _write(env.db_path, "PRAGMA user_version = 8")
    stat_before = (env.db_path.stat().st_size, env.db_path.stat().st_mtime_ns)
    sidecars_before = _sidecars(env.tmp_path)

    with pytest.raises(RuntimeError, match=r"unsupported SQLite schema version 8; expected 7"):
        await database.bootstrap_sqlite()

    assert _version(env.db_path) == 8, "the refusal changed the stamp"
    assert (env.db_path.stat().st_size, env.db_path.stat().st_mtime_ns) == stat_before
    assert _sidecars(env.tmp_path) == sidecars_before


# ── 9. R34's failure mode (carried from the T3 review) ───────────────────────


async def test_r34_a_key_less_point_is_deleted_by_its_delete_intent(live):
    """A pre-P4 point (no ``namespace`` key) must still be deletable by its intent.

    R32 makes a key-less point personal for READS; R34 keeps the DELETE
    identity-exact (point id, never a namespace-filtered selector). The two
    rulings only work together: a selector that filtered on the key would MISS
    this point — the intent would then ride the retry loop forever while the
    point stayed served. This test is that failure mode, end to end.
    """
    generation = generation_name("memory")
    async with live.sessions() as db:
        legacy = _memory(live.alice.id, "pre-namespace point")
        db.add(legacy)
        await db.commit()
        legacy_id, legacy_revision = legacy.id, legacy.revision

    vector_backend.get_sync_client().upsert(
        collection_name=generation,
        points=[qm.PointStruct(
            id=str(legacy_id),
            vector=_vector_for(vector_store._memory_to_document(legacy)),
            # A P0/P1a payload verbatim: no `namespace` key at all.
            payload={"kind": "memory", "user_id": str(live.alice.id), "memory_id": str(legacy_id)},
        )],
    )
    assert "namespace" not in _payloads(generation)[str(legacy_id)], "the fixture must stay key-less"

    hits = await vector_store.search_memories(
        _vector_for(vector_store._memory_to_document(legacy)),
        user_id=str(live.alice.id), top_k=10)
    assert str(legacy_id) in {hit["memory_id"] for hit in hits}, (
        "R32: a key-less point is personal and IS served — the delete must reach it")

    async with live.sessions() as db:
        await outbox.enqueue_delete(db, entity_id=legacy_id.hex, tenant_id=live.alice.id.hex,
                                    revision=legacy_revision)
        await db.commit()

    assert (await outbox.drain_pending())["applied"] == 1
    assert await vector_store.get_memory_ids_present([str(legacy_id)]) == set(), (
        "the point survived its delete intent — a namespace-filtered selector would do exactly this")
    intents = [row for row in await _intents(live) if row.entity_id == legacy_id.hex]
    assert [row.status for row in intents] == ["done"]


# ── 10. the cache key (§4.3), answered honestly ──────────────────────────────


def _call_sites(functions: set[str]) -> dict[str, list[str]]:
    """Direct calls and string LITERALS naming a seam, anywhere under ``app/``.

    Two shapes, because one is not enough: a call by bare name or by attribute
    (``rc.get_cached_chunks(...)``) AND any string constant that spells one of
    the names — the ``getattr(rc, "get_cached_chunks")`` form the final review
    reproduced, which a call-node-only scan never sees. The ceiling is in the
    test's docstring; it is not "no cache exists".
    """
    hits: dict[str, list[str]] = {}
    for path in sorted(APP.rglob("*.py")):
        if path.name in {"retrieval_cache.py", "response_cache.py"}:
            continue  # the definitions themselves
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                name = next((seam for seam in functions if seam in node.value), None)
            else:
                continue
            if name in functions:
                hits.setdefault(name, []).append(f"{path.relative_to(REPO)}:{node.lineno}")
    return hits


def test_the_cache_key_question_is_answered_honestly():
    """§4.3 wants a namespace component in every cache key — there is no live cache.

    Both cache modules exist, and both are invalidation-only: nothing in ``app/``
    calls ``get_cached_chunks``/``set_cached_chunks`` or
    ``get_cached_response``/``set_cached_response``, so no key is ever built and
    the namespace component is NOT implemented in P4a (recorded as a P4b
    residual, not faked here).

    The pin's ceiling, stated as a ceiling: it sees a DIRECT call (bare name or
    attribute) and any string literal naming a seam — and nothing else. A
    ``functools.partial``, an import alias (``import get_cached_chunks as gcc``),
    a name assembled at runtime, or ``getattr(module, variable)`` slips through,
    so green here reads "no direct call and no literal reference", never "no
    cache exists". The day a directly-called producer or reader goes live this
    test fails and says what §4.3 owes; a meta-programmed one it will not catch.
    """
    from app.middleware import response_cache
    from app.retrieval import retrieval_cache

    seams = {
        "get_cached_chunks": retrieval_cache.get_cached_chunks,
        "set_cached_chunks": retrieval_cache.set_cached_chunks,
        "get_cached_response": response_cache.get_cached_response,
        "set_cached_response": response_cache.set_cached_response,
    }
    assert all(callable(seam) for seam in seams.values()), "the scan is vacuous: a seam is gone"

    call_sites = _call_sites(set(seams))
    assert call_sites == {}, (
        "a retrieval/response cache has a live producer or reader: §4.3 requires its key to "
        f"carry the authorized namespace before it can serve — implement it, then update this "
        f"gate. Call sites: {call_sites}"
    )

    # The key SHAPES are still pinned so the residual is unambiguous: the
    # retrieval key is (conversation, query hash) and the response key is
    # (path, user, params) — neither carries a namespace today.
    assert "conversation_id" in retrieval_cache.query_cache_key.__code__.co_varnames
    assert "namespace" not in retrieval_cache.query_cache_key.__code__.co_varnames
    assert "namespace" not in response_cache._get_cache_key.__code__.co_varnames


# ── 11. the CI pin (R30-style: parse the workflow, not a copy of it) ────────


def _workflow() -> dict:
    return yaml.safe_load(CI_YML.read_text())


def _step(job: dict, name: str) -> dict:
    for step in job["steps"]:
        if step.get("name") == name:
            return step
    raise AssertionError(f"CI step {name!r} is gone: {[s.get('name') for s in job['steps']]}")


def test_the_workflow_runs_this_gate_and_names_only_paths_that_exist():
    """R30-style: dropping this gate from CI (or narrowing its env, or wiring a
    path that no longer exists) fails HERE instead of silently.

    pytest exits 4 on a path that does not exist, so every ``tests/...`` token
    the workflow, the Makefile or the four P4a docs name is checked to exist.
    """
    workflow = _workflow()
    step = _step(workflow["jobs"]["test"], CI_STEP_NAME)
    run = step["run"]

    assert step["env"]["DATABASE_URL"].startswith("sqlite+aiosqlite:////tmp/"), (
        "the P4a step must pin its own disposable SQLite file, like P1a/P1b/P2/P3")
    # The step's own kill-switches. `--ignore=<this file>` lands the gate on
    # pytest's ignore list while the step still reports green (the final review
    # reproduced 12 passed with exactly that edit); `continue-on-error`/`if:`
    # neuter the whole step. A pin that only proves "the path is spelled" is
    # blind to both.
    assert "continue-on-error" not in step and "if" not in step, (
        f"the {CI_STEP_NAME!r} step can be stood down without touching its run "
        f"command: {sorted(step)}")
    for suite in P4_CI_SUITES:
        assert suite in run, f"{suite} is not wired into the {CI_STEP_NAME!r} step"
    assert "--ignore" not in run, (
        f"an --ignore in the {CI_STEP_NAME!r} step skips a suite it claims to run "
        f"(pytest exits 0 with it ignored), so the step would read green with the "
        f"gate never executed: {run!r}")
    all_runs = "\n".join(s.get("run", "") for job in workflow["jobs"].values() for s in job["steps"])
    for suite in P4_ERASURE_SUITES:
        assert suite in all_runs, f"{suite} is wired into no CI step at all"

    # The parity suite must stay wired SOMEWHERE: the lite product has no
    # Qdrant server to run it against (embedded only), so the P1b step's
    # embedded-folder run is the surviving home.
    assert "tests/retrieval/test_qdrant_parity.py" in all_runs, (
        "the Qdrant parity suite is wired into no CI step at all")

    # No dead path: pytest exits 4 on a missing target, and a stale mention in
    # the docs is the same lie told to a reader.
    sources = [CI_YML, REPO / "Makefile", *P4A_DOCS]
    pattern = __import__("re").compile(r"tests/[A-Za-z0-9_./*-]+")
    missing: dict[str, str] = {}
    for source in sources:
        for token in pattern.findall(source.read_text()):
            token = token.rstrip(".,;:)`'\"")
            if not list(REPO.glob(token)):
                missing[token] = source.relative_to(REPO).as_posix()
    assert missing == {}, f"cite tests/ paths that do not exist: {missing}"

    assert GATE_MODULE in run, "the acceptance gate itself must be in the step"


# ── hygiene: the value's one spelling ────────────────────────────────────────


def test_the_gate_pins_the_version_and_the_namespace_spelling():
    """A silent drift in either constant would leave these bullets checking nothing."""
    assert database.SQLITE_SCHEMA_VERSION == 7, "the gate's ladder bullets are written for the terminal stamp"
    assert namespaces.PERSONAL == PERSONAL == "personal"
