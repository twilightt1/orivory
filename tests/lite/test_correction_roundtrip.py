"""Real-SQLite proof: JSON resolve filter + §9b self-check. No fakes."""
from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from app.database import IS_SQLITE, AsyncSessionLocal, Base, engine
from app.retrieval.memory.correction import Slot, resolve_correction

pytestmark = pytest.mark.skipif(not IS_SQLITE, reason="lite-mode tests require a sqlite DATABASE_URL")


@pytest_asyncio.fixture
async def _tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


async def test_self_check_supersede_scope_isolation(_tables):
    """Spec §9b: same triple hides old by default; other scope untouched."""
    from app.mcp_hub import tools as hub_tools
    from app.mcp_hub.identity import AgentPrincipal
    from app.models.user import User

    uid = uuid.uuid4()
    async with AsyncSessionLocal() as db:
        db.add(User(id=uid, email=f"selfcheck-{uid}@example.com", hashed_password="x",
                    display_name="SC", is_verified=True, is_active=True))
        await db.commit()
    async with AsyncSessionLocal() as db:
        r1 = await resolve_correction(db, user_id=uid, title="DB", content="Postgres",
            slot=Slot.of("proj", "db", "prod"))
        r2 = await resolve_correction(db, user_id=uid, title="DB", content="SQLite",
            slot=Slot.of("proj", "db", "demo"))
        r3 = await resolve_correction(db, user_id=uid, title="DB", content="PG16",
            slot=Slot.of("proj", "db", "prod"))
    assert r1["status"] == "added"
    assert r3["status"] == "superseded"
    assert r2["status"] == "added"  # other scope untouched
    async with AsyncSessionLocal():
        from app.mcp_hub import tools as hub_tools
        from app.mcp_hub.identity import AgentPrincipal

        p = AgentPrincipal(user_id=uid, agent_client_id=uuid.uuid4(),
                           name="SC", scopes=frozenset({"memory:read"}))
        tok = hub_tools._principal_var.set(p)
        try:
            out = await hub_tools.search_memory("db")
            ids = [r["id"] for r in out["results"]]
            assert str(r3["memory"].id) in ids
            assert str(r1["memory"].id) not in ids  # old prod hidden by default
            out2 = await hub_tools.search_memory("db", include_history=True)
            assert str(r1["memory"].id) in {r["id"] for r in out2["results"]}
            got = await hub_tools.get_memory(str(r3["memory"].id))
            assert got["supersedes"] == str(r1["memory"].id)
        finally:
            hub_tools._principal_var.reset(tok)


async def _user(uid=None):
    from app.models.user import User

    uid = uid or uuid.uuid4()
    async with AsyncSessionLocal() as db:
        db.add(User(id=uid, email=f"ku-{uid}@example.com", hashed_password="x",
                    display_name="KU", is_verified=True, is_active=True))
        await db.commit()
    return uid


def _principal(uid, write=False):
    from app.mcp_hub.identity import AgentPrincipal

    return AgentPrincipal(user_id=uid, agent_client_id=uuid.uuid4(),
                          name="KU",
                          scopes=frozenset({"memory:read", "memory:write"} if write else {"memory:read"}))


async def test_ku_correct_supersedes_via_mcp(_tables):
    """V1 hypothesis: user corrects once via correct_memory, later search
    serves the new fact, history keeps the old with its chain."""
    from app.mcp_hub import tools as hub_tools

    uid = await _user()
    tok = hub_tools._principal_var.set(_principal(uid, write=True))
    try:
        first = await hub_tools.correct_memory(subject="proj", attribute="db",
            scope="prod", title="DB", content="Postgres")
        assert first["status"] == "added"
        second = await hub_tools.correct_memory(subject="proj", attribute="db",
            scope="prod", title="DB", content="PG16")
        assert second["status"] == "superseded"
        assert second["superseded"] == [first["id"]]
    finally:
        hub_tools._principal_var.reset(tok)

    tok = hub_tools._principal_var.set(_principal(uid))
    try:
        out = await hub_tools.search_memory("db")
        ids = [r["id"] for r in out["results"]]
        assert second["id"] in ids and first["id"] not in ids
        hist = await hub_tools.search_memory("db", include_history=True)
        assert {first["id"], second["id"]} <= {r["id"] for r in hist["results"]}
        got = await hub_tools.get_memory(second["id"])
        assert got["state"] == "current" and got["supersedes"] == first["id"]
        old = await hub_tools.get_memory(first["id"])
        assert old["state"] == "superseded"
    finally:
        hub_tools._principal_var.reset(tok)


async def test_ku_ambiguous_correction_keeps_both_via_mcp(_tables):
    from app.mcp_hub import tools as hub_tools

    uid = await _user()
    tok = hub_tools._principal_var.set(_principal(uid, write=True))
    try:
        base = await hub_tools.correct_memory(subject="proj", attribute="db",
            scope="prod", title="DB", content="Postgres")
        assert base["status"] == "added"
        # empty scope + same subject/attribute elsewhere -> needs-check
        vague = await hub_tools.correct_memory(subject="proj", attribute="db",
            scope="", title="DB", content="SQLite")
        assert vague["status"] == "needs-check"
        got = await hub_tools.get_memory(vague["id"])
        assert got["state"] == "needs-check"
    finally:
        hub_tools._principal_var.reset(tok)

    tok = hub_tools._principal_var.set(_principal(uid))
    try:
        out = await hub_tools.search_memory("db")
        # nothing superseded: both stay visible
        assert {base["id"], vague["id"]} <= {r["id"] for r in out["results"]}
    finally:
        hub_tools._principal_var.reset(tok)


async def test_ku_wrong_target_and_bad_date_need_check(_tables):
    from app.mcp_hub import tools as hub_tools

    uid = await _user()
    tok = hub_tools._principal_var.set(_principal(uid, write=True))
    try:
        base = await hub_tools.correct_memory(subject="proj", attribute="db",
            scope="prod", title="DB", content="Postgres")
        other = await hub_tools.correct_memory(subject="proj", attribute="db",
            scope="prod", title="DB", content="Other")
        assert other["status"] == "superseded"  # same triple: legit correction
        unrelated = await hub_tools.correct_memory(subject="other", attribute="x",
            scope="prod", title="X", content="unrelated")
        # memory_id outside the matched triple -> ambiguous, keep both
        clash = await hub_tools.correct_memory(memory_id=unrelated["id"],
            subject="proj", attribute="db", scope="prod",
            title="DB", content="PG16")
        assert clash["status"] == "needs-check"
        # unparseable valid_from poisons only the meta
        dated = await hub_tools.correct_memory(subject="proj", attribute="db",
            scope="demo", title="DB", content="SQLite", valid_from="not-a-date")
        assert dated["status"] == "added"
        got = await hub_tools.get_memory(dated["id"])
        assert got["state"] == "needs-check"
        # the matched fact stands: latest correction wins, history kept
        cur = await hub_tools.get_memory(other["id"])
        assert cur["state"] == "current"
        old = await hub_tools.get_memory(base["id"])
        assert old["state"] == "superseded"
    finally:
        hub_tools._principal_var.reset(tok)


async def test_ku_foreign_memory_id_rejected(_tables):
    from app.mcp_hub import tools as hub_tools

    uid, victim = await _user(), await _user()
    tok = hub_tools._principal_var.set(_principal(uid, write=True))
    try:
        w_tok = hub_tools._principal_var.set(_principal(victim, write=True))
        try:
            theirs = await hub_tools.correct_memory(title="T", content="secret")
        finally:
            hub_tools._principal_var.reset(w_tok)
        out = await hub_tools.correct_memory(memory_id=theirs["id"], content="x")
        assert out == {"error": "memory not found"}
    finally:
        hub_tools._principal_var.reset(tok)
