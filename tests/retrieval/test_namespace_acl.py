"""Namespace ACL over every REST memory surface (P4a Task 2).

Two users, one namespace each (``personal``) — and, on top of the cross-TENANT
rows this router already refused, rows of the caller's OWN account that live
outside the caller's namespace (``team`` stands in for the second namespace P4b
will bring). Every surface must serve, count and mutate personal rows only;
the public share endpoint — which deliberately has no request user — must never
hand out a row outside the public (personal) namespace.

Hermetic: a private per-test SQLite file on ``tmp_path``, bound in as the app's
engine / sessionmaker, so nothing here reads or writes whatever ``DATABASE_URL``
is ambient (pattern: ``tests/retrieval/test_visibility.py``). Only the
out-of-process vector seams are stubbed (the write-back embed and the erasure
purge/readback); every SQL statement and every HTTP hop is real.

The AST pins at the bottom are the anti-regression half: a new ``select(Memory)``
in these surfaces that forgets its namespace predicate — or a literal
``'personal'`` written back into a query — fails here. Task 5 widens the same
scan to every dormant reader under ``app/``.
"""
from __future__ import annotations

import ast
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

import app.models  # noqa: F401 — register every ORM table on Base
from app import database
from app.api.v1 import memories as memories_api
from app.database import Base, get_db
from app.main import app as asgi_app
from app.models.memory import Memory
from app.models.user import User
from app.retrieval.memory import namespaces
from app.utils.dependencies import get_current_verified_user

ACL_DB = "namespace-acl.sqlite"
TEAM = "team"  # a namespace client input can never create in P4a — seeded directly


def _user(uid: uuid.UUID) -> User:
    return User(id=uid, email=f"{uid.hex}@acl.invalid", hashed_password="x",
                display_name="Acl", is_verified=True, is_active=True)


def _mem(owner: uuid.UUID, title: str, *, namespace: str = namespaces.PERSONAL,
         tags: tuple[str, ...] = (), age: timedelta = timedelta(),
         shared: bool = False) -> Memory:
    return Memory(
        id=uuid.uuid4(), user_id=owner, title=title, content=f"{title} body",
        tags=list(tags), salience=0.5, pinned=False, is_shared=shared,
        captured_at=datetime.now(UTC) - age, extra_metadata={},
        namespace=namespace,
    )


@pytest_asyncio.fixture
async def live(tmp_path, monkeypatch):
    """The private file + an auth-switching ASGI client over the real router."""
    url = f"sqlite+aiosqlite:///{tmp_path / ACL_DB}"
    eng = create_async_engine(url, connect_args={"check_same_thread": False}, poolclass=NullPool)
    event.listen(eng.sync_engine, "connect", database._configure_sqlite_connection)
    sync_eng = create_engine(url.replace("+aiosqlite", ""),
                             connect_args={"check_same_thread": False}, poolclass=NullPool)
    event.listen(sync_eng, "connect", database._configure_sqlite_connection)
    sessions = async_sessionmaker(eng, class_=AsyncSession, expire_on_commit=False, autoflush=False)

    monkeypatch.setattr(database, "engine", eng)
    monkeypatch.setattr(database, "IS_SQLITE", True)
    monkeypatch.setattr(database, "AsyncSessionLocal", sessions)
    monkeypatch.setattr(database, "_get_sync_sessionmaker",
                        lambda: sessionmaker(bind=sync_eng, expire_on_commit=False, autoflush=False))

    async def _db_override():
        async with sessions() as session:
            yield session

    asgi_app.dependency_overrides[get_db] = _db_override

    # The out-of-process seams this suite is not about: the vector write-back
    # (index_new_memory / safe_upsert_to_index) and the erasure purge + presence
    # readback. The SQL under them and the HTTP hops above them stay real.
    async def _no_index(_memory):
        return False

    async def _no_residual(_memory_ids):
        return set()

    monkeypatch.setattr(memories_api, "index_new_memory", _no_index)
    monkeypatch.setattr(memories_api, "safe_upsert_to_index", _no_index)
    monkeypatch.setattr("app.services.erasure_service.safe_delete_from_index", _no_index)
    monkeypatch.setattr("app.services.erasure_service._vector_present_ids", _no_residual)

    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    def as_user(user_id: uuid.UUID) -> AsyncClient:
        async def _current_user():
            return _user(user_id)

        asgi_app.dependency_overrides[get_current_verified_user] = _current_user
        return AsyncClient(transport=ASGITransport(app=asgi_app), base_url="http://test")

    async def seed(*rows: object) -> None:
        async with sessions() as session:
            session.add_all(rows)
            await session.commit()

    async def read(model, pk):
        async with sessions() as session:
            return await session.get(model, pk)

    try:
        yield SimpleNamespace(as_user=as_user, seed=seed, read=read)
    finally:
        asgi_app.dependency_overrides.clear()
        await eng.dispose()
        sync_eng.dispose()


class _World:
    """Both users' rows: recent + old, personal + out-of-namespace, shared + not.

    Every row the ACL must refuse is here, so a surface that forgets the
    predicate is caught on the row itself, not on an absent one.
    """

    def __init__(self, live):
        self.live = live
        self.a, self.b = uuid.uuid4(), uuid.uuid4()
        self.a_personal = _mem(self.a, "A personal recent", tags=("alpha",), age=timedelta(hours=1))
        self.a_old = _mem(self.a, "A personal old", tags=("alpha",),
                          age=timedelta(days=365), shared=True)
        self.a_team = _mem(self.a, "A team recent", namespace=TEAM, tags=("teamtag",),
                           age=timedelta(hours=2))
        self.a_team_old = _mem(self.a, "A team old", namespace=TEAM, tags=("teamtag",),
                               age=timedelta(days=365), shared=True)
        self.b_personal = _mem(self.b, "B personal recent", tags=("beta",), age=timedelta(hours=1))
        self.b_old = _mem(self.b, "B personal old", tags=("beta",),
                          age=timedelta(days=365), shared=True)

    async def seed(self):
        await self.live.seed(
            _user(self.a), _user(self.b),
            self.a_personal, self.a_old, self.a_team, self.a_team_old,
            self.b_personal, self.b_old,
        )
        return self


@pytest_asyncio.fixture
async def world(live):
    return await _World(live).seed()


# ── readers ──────────────────────────────────────────────────────────────────


async def test_list_serves_only_the_callers_namespace(world):
    async with world.live.as_user(world.a) as client:
        body = (await client.get("/api/v1/memories")).json()

    assert {m["id"] for m in body["items"]} == {str(world.a_personal.id), str(world.a_old.id)}
    assert body["total"] == 2, "the pagination total carries the same predicate as the page"

    async with world.live.as_user(world.b) as client:
        other = (await client.get("/api/v1/memories")).json()

    assert {m["id"] for m in other["items"]} == {str(world.b_personal.id), str(world.b_old.id)}
    assert other["total"] == 2


async def test_digest_counts_only_the_callers_namespace(world):
    async with world.live.as_user(world.a) as client:
        body = (await client.get("/api/v1/memories/digest")).json()

    assert body["recent_count"] == 1, "the window count is one of the three queries here"
    assert {m["id"] for m in body["recent_memories"]} == {str(world.a_personal.id)}
    assert [t["theme"] for t in body["top_themes"]] == ["alpha"]
    assert {r["memory"]["id"] for r in body["resurfaced"]} == {str(world.a_old.id)}, \
        "'on this day' is scoped too — the team row is exactly as old and must not surface"


async def test_stats_aggregates_only_the_callers_namespace(world):
    async with world.live.as_user(world.a) as client:
        body = (await client.get("/api/v1/memories/stats")).json()

    assert body["total_memories"] == 2
    assert body["top_tags"] == [{"tag": "alpha", "count": 2}]
    assert body["recent_activity"][0]["count"] == 1


async def test_get_one_refuses_foreign_tenant_and_foreign_namespace(world):
    async with world.live.as_user(world.a) as client:
        team = await client.get(f"/api/v1/memories/{world.a_team.id}")
        tenant = await client.get(f"/api/v1/memories/{world.b_personal.id}")
        missing = await client.get(f"/api/v1/memories/{uuid.uuid4()}")
        own = await client.get(f"/api/v1/memories/{world.a_personal.id}")

    assert (team.status_code, tenant.status_code, missing.status_code) == (404, 404, 404)
    assert team.json() == missing.json(), "same body as a missing id — no existence oracle"
    assert own.status_code == 200 and own.json()["id"] == str(world.a_personal.id)


# ── writers ──────────────────────────────────────────────────────────────────


async def test_patch_refuses_foreign_tenant_and_foreign_namespace(world):
    async with world.live.as_user(world.a) as client:
        team = await client.patch(f"/api/v1/memories/{world.a_team.id}", json={"title": "hit"})
        tenant = await client.patch(f"/api/v1/memories/{world.b_personal.id}", json={"title": "hit"})

    assert (team.status_code, tenant.status_code) == (404, 404)
    assert (await world.live.read(Memory, world.a_team.id)).title == "A team recent"
    assert (await world.live.read(Memory, world.b_personal.id)).title == "B personal recent"

    async with world.live.as_user(world.a) as client:
        own = await client.patch(f"/api/v1/memories/{world.a_personal.id}", json={"title": "renamed"})

    assert own.status_code == 200 and own.json()["title"] == "renamed"
    assert (await world.live.read(Memory, world.a_personal.id)).title == "renamed"


async def test_delete_refuses_foreign_tenant_and_foreign_namespace(world):
    async with world.live.as_user(world.a) as client:
        team = await client.delete(f"/api/v1/memories/{world.a_team_old.id}")
        tenant = await client.delete(f"/api/v1/memories/{world.b_personal.id}")

    assert (team.status_code, tenant.status_code) == (404, 404)
    assert (await world.live.read(Memory, world.a_team_old.id)) is not None
    assert (await world.live.read(Memory, world.b_personal.id)) is not None

    async with world.live.as_user(world.a) as client:
        own = await client.delete(f"/api/v1/memories/{world.a_personal.id}")

    assert own.status_code == 204
    assert (await world.live.read(Memory, world.a_personal.id)) is None


async def test_create_refuses_a_parent_outside_the_callers_namespace(world):
    async with world.live.as_user(world.a) as client:
        team = await client.post("/api/v1/memories",
                                 json={"content": "child", "parent_id": str(world.a_team.id)})
        tenant = await client.post("/api/v1/memories",
                                   json={"content": "child", "parent_id": str(world.b_personal.id)})
        own = await client.post("/api/v1/memories",
                                json={"content": "child of mine", "parent_id": str(world.a_personal.id)})

    assert (team.status_code, tenant.status_code) == (404, 404)
    assert team.json()["detail"] == "Parent memory not found"
    assert own.status_code == 201

    created = await world.live.read(Memory, uuid.UUID(own.json()["id"]))
    assert created.parent_id == world.a_personal.id
    assert created.namespace == namespaces.PERSONAL, (
        "the writer lands the row in the caller's namespace, not in whatever the "
        "column default happens to be")


# ── public share (no auth dependency — the namespace clause IS the boundary) ──


async def test_share_is_scoped_to_the_public_namespace(world):
    async with world.live.as_user(world.a) as client:
        shared = await client.get(f"/api/v1/memories/{world.a_old.id}/share")
        out_of_namespace = await client.get(f"/api/v1/memories/{world.a_team_old.id}/share")
        not_shared = await client.get(f"/api/v1/memories/{world.a_personal.id}/share")
        other_tenant = await client.get(f"/api/v1/memories/{world.b_old.id}/share")

    assert shared.status_code == 200 and shared.json()["id"] == str(world.a_old.id)
    assert out_of_namespace.status_code == 404, "sharing a row never widens its namespace"
    assert not_shared.status_code == 404
    # The link is deliberately owner-agnostic (whoever holds it may read the row):
    # P4a narrows the endpoint to the public namespace, it does not re-tenant it.
    assert other_tenant.status_code == 200


# ── pins (Step 3): class-level, not line-level ───────────────────────────────

REPO = Path(__file__).resolve().parents[2]
SURFACES = (
    REPO / "app" / "api" / "v1" / "memories.py",
    REPO / "app" / "services" / "digest_service.py",
)


def _is_select(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "select")


def _namespace_aliases(tree: ast.AST) -> set[str]:
    """Names a module binds to a ``namespace_predicate(...)`` call.

    A reader may hoist the predicate into a local (``namespace = namespace_predicate(ns)``)
    and compose it into every query of the function; the alias counts as the
    predicate ONLY where a statement actually references it — a query that
    forgets the name still fails below.
    """
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        if not (isinstance(func, ast.Name) and func.id == "namespace_predicate"):
            continue
        aliases.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return aliases


def _memory_statements(path: Path) -> list[tuple[int, str, bool]]:
    """``(line, source, carries_namespace_predicate)`` per ``select(...)`` over ``Memory``.

    Statement-level: the predicate may be chained after the ``select()`` (a
    ``.where(...)``) and only the statement as a whole can be judged. Comments
    are stripped so a sentence cannot satisfy the pin.
    """
    source = path.read_text()
    tree = ast.parse(source)
    aliases = _namespace_aliases(tree)
    parents = {child: parent
               for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    found: list[tuple[int, str, bool]] = []
    for node in ast.walk(tree):
        if not _is_select(node):
            continue
        if not any(isinstance(sub, ast.Name) and sub.id == "Memory" for sub in ast.walk(node)):
            continue
        stmt: ast.AST = node
        while not isinstance(stmt, ast.stmt):
            stmt = parents[stmt]
        segment = "\n".join(line.split("#", 1)[0]
                            for line in (ast.get_source_segment(source, stmt) or "").splitlines())
        referenced = {sub.id for sub in ast.walk(stmt) if isinstance(sub, ast.Name)}
        guarded = "namespace_predicate(" in segment or bool(referenced & aliases)
        found.append((stmt.lineno, segment, guarded))
    return found


def test_every_memory_read_in_these_surfaces_carries_the_namespace_predicate():
    """A ``select(Memory)`` without ``namespace_predicate`` is a leak waiting.

    This half pins the QUERY surfaces (list + its pagination total, the three
    stats aggregates, share, all three digest queries). The primary-key surfaces
    (get / patch / delete / create-with-parent) cannot carry a predicate on a
    ``db.get``, so they are pinned behaviourally by the tests above instead.

    Scoped to the files this task owns — the dormant readers (``discovery``,
    ``entities``, ``insights``, ...) get the same treatment, over all of
    ``app/``, in Task 5.
    """
    scanned = {path: _memory_statements(path) for path in SURFACES}
    assert all(scanned.values()), "the scan must find the reads it judges (never pass vacuously)"

    missing = [f"{path.relative_to(REPO)}:{line}"
               for path, statements in scanned.items()
               for line, _segment, guarded in statements if not guarded]
    assert not missing, f"memory reads without the namespace predicate: {missing}"


def test_the_rest_surfaces_never_hardcode_the_personal_literal():
    """The value comes from ``namespaces``, so a predicate cannot drift from the rows."""
    for path in SURFACES:
        source = path.read_text()
        assert "'personal'" not in source and '"personal"' not in source, (
            f"{path.relative_to(REPO)} spells the namespace instead of routing through "
            "namespaces.personal_namespace()")
