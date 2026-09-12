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
