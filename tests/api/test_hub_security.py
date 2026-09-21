"""Behavioral REST security tests for the memory hub.

Real request/response semantics through the ASGI app against the suite's SQLite
file:
auth boundaries, cross-user isolation (no existence leaks), payload-shape
contracts — the properties the final code reviews flagged as untested.
"""
from __future__ import annotations

import hashlib
import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.main import app
from app.models.agent_client import AgentClient
from app.models.memory import Memory
from app.models.user import User
from app.utils.dependencies import get_current_user
from tests.conftest import make_async_test_engine

pytestmark = pytest.mark.api


def _agent_row(user_id: uuid.UUID, name: str) -> AgentClient:
    # uuid suffix keeps token hashes unique across test runs (unique index)
    token = f"oa_{name}_{uuid.uuid4().hex}"
    return AgentClient(
        user_id=user_id,
        name=name,
        token_hash=hashlib.sha256(token.encode()).hexdigest(),
        scopes=["memory:read", "memory:write"],
    )


def _memory_row(user_id: uuid.UUID, content: str, ref: str | None = None) -> Memory:
    return Memory(
        user_id=user_id,
        content=content,
        title=content[:60],
        source_ref=ref,
        tags=[],
        extra_metadata={},
    )


_user_ids: dict[str, uuid.UUID] = {}


async def _auth_client(user_id: uuid.UUID) -> AsyncClient:
    """ASGI client with auth + DB overridden on THIS test's loop.

    The app's global engine binds connections to whatever loop created them;
    pytest-asyncio gives each test a fresh loop, so `get_db` must be
    overridden with a per-test engine instead (loop-bound by construction —
    created inside the test's coroutine).
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker

    test_engine = make_async_test_engine()
    test_session = async_sessionmaker(test_engine, class_=AsyncSession,
                                      expire_on_commit=False)

    async def _user_override():
        return User(id=user_id, email=f"{user_id}@example.com", hashed_password="x",
                    is_verified=True, is_active=True)

    async def _db_override():
        async with test_session() as session:
            yield session

    app.dependency_overrides[get_current_user] = _user_override
    app.dependency_overrides[get_db] = _db_override
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _close(client: AsyncClient) -> None:
    await client.aclose()
    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture()
def no_chroma(monkeypatch):
    """Stub the erasure service's vector seams — behavioral tests cover DB
    + HTTP semantics, not live Chroma (which isn't running here)."""
    async def _noop_delete(memory_id):
        return True

    async def _no_residual(memory_ids):
        return set()

    monkeypatch.setattr(
        "app.services.erasure_service.safe_delete_from_index", _noop_delete
    )
    monkeypatch.setattr(
        "app.services.erasure_service._vector_present_ids", _no_residual
    )


def _session_factory():
    from sqlalchemy.ext.asyncio import async_sessionmaker

    engine = make_async_test_engine()
    return engine, async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@pytest.mark.asyncio
async def test_revoke_foreign_agent_is_404_no_leak():
    user_id = uuid.uuid4()
    foreign_id = uuid.uuid4()
    own = _agent_row(user_id, "own")
    foreign = _agent_row(foreign_id, "foreign")

    engine, Session = _session_factory()
    async with Session() as db:
        db.add_all([own, foreign])
        await db.commit()
        own_id = own.id
    await engine.dispose()

    client = await _auth_client(user_id)
    async with client as c:
        resp_foreign = await c.delete(f"/api/v1/agents/{foreign_id}")
        resp_missing = await c.delete(f"/api/v1/agents/{uuid.uuid4()}")
        assert resp_foreign.status_code == 404
        assert resp_missing.status_code == 404
        # identical bodies — no existence oracle
        assert resp_foreign.json() == resp_missing.json()

        resp_own = await c.delete(f"/api/v1/agents/{own_id}")
        assert resp_own.status_code == 204

        # re-revoke: idempotent 204, original revoked_at preserved
        engine2, Session2 = _session_factory()
        async with Session2() as db:
            row = await db.get(AgentClient, own_id)
            first_revoked_at = row.revoked_at
        await engine2.dispose()
        resp_again = await c.delete(f"/api/v1/agents/{own_id}")
        assert resp_again.status_code == 204
        async with Session2() as db:
            row = await db.get(AgentClient, own_id)
            assert row.revoked_at == first_revoked_at
        await engine2.dispose()


@pytest.mark.asyncio
async def test_agent_registration_token_shown_exactly_once():
    user_id = uuid.uuid4()
    engine, Session = _session_factory()
    async with Session() as db:
        db.add(User(id=user_id, email=f"{user_id}@example.com", hashed_password="x",
                    is_verified=True, is_active=True))
        await db.commit()
    await engine.dispose()
    client = await _auth_client(user_id)
    async with client as c:
        reg = await c.post(
            "/api/v1/agents",
            json={"name": "once-only", "scopes": ["memory:read", "memory:write"]},
        )
        assert reg.status_code == 201, reg.text
        body = reg.json()
        assert body["token"].startswith("oa_") and len(body["token"]) == 35

        listing = await c.get("/api/v1/agents")
        for item in listing.json()["items"]:
            assert "token" not in item
            assert "token_hash" not in item
    await _close(client)


@pytest.mark.asyncio
async def test_import_dedup_counts_over_http():
    user_id = uuid.uuid4()
    ref = f"imp-{uuid.uuid4().hex[:8]}"
    payload = (
        f'[{{"content": "imported fact one", "ref": "{ref}", "created_at": "2026-01-01T00:00:00Z"}},'
        '{"content": "imported fact two"}]'
    )
    engine, Session = _session_factory()
    async with Session() as db:
        db.add(User(id=user_id, email=f"{user_id}@example.com", hashed_password="x",
                    is_verified=True, is_active=True))
        await db.commit()
    await engine.dispose()

    client = await _auth_client(user_id)
    # /imports authenticates via `_optional_user` (dual human/agent auth), not
    # `get_current_user` — without this the calls below 401.
    from app.api.v1 import imports as imports_module

    async def _imports_user_override():
        return User(id=user_id, email=f"{user_id}@example.com", hashed_password="x",
                    is_verified=True, is_active=True)

    app.dependency_overrides[imports_module._optional_user] = _imports_user_override
    async with client as c:
        resp1 = await c.post(
            "/api/v1/imports",
            files={"file": ("export.json", payload.encode(), "application/json")},
        )
        assert resp1.status_code == 201, resp1.text
        body = resp1.json()
        assert (body["parsed"], body["created"], body["failed"]) == (2, 2, 0)

        resp2 = await c.post(
            "/api/v1/imports",
            files={"file": ("export.json", payload.encode(), "application/json")},
        )
        assert resp2.status_code == 201
        assert resp2.json()["skipped_duplicates"] == 1
    app.dependency_overrides.pop(imports_module._optional_user, None)
    await _close(client)


@pytest.mark.asyncio
async def test_erasure_receipt_records_targets_and_deletes(no_chroma):
    user_id = uuid.uuid4()
    m1 = _memory_row(user_id, "to erase one", "er-1")
    m2 = _memory_row(user_id, "to erase two", "er-2")

    engine, Session = _session_factory()
    async with Session() as db:
        db.add(User(id=user_id, email=f"{user_id}@example.com", hashed_password="x",
                    is_verified=True, is_active=True))
        db.add_all([m1, m2])
        await db.commit()
        m1_id, m2_id = str(m1.id), str(m2.id)
    await engine.dispose()

    client = await _auth_client(user_id)
    async with client as c:
        resp = await c.post(
            "/api/v1/erasure-receipts",
            json={"memory_ids": [m1_id, m2_id]},
        )
        assert resp.status_code == 201, resp.text
        receipt = resp.json()
        assert receipt["status"] in (
            "completed",
            "completed_with_residual",
            "completed_with_errors",
        )
        targets = receipt["detail"]["targets"]
        assert {t["memory_id"] for t in targets} == {m1_id, m2_id}
        assert all(t["status"] == "deleted" for t in targets)
        assert all(t["vector_residual_checked"] for t in targets)
        # Additive (Task 4): the receipt says what was verified and what the
        # index side still owes, and every target carries its own vector state.
        assert all(t["vector_state"] == "verified" for t in targets)
        assert receipt["detail"]["verification"] == "verified"
        assert receipt["detail"]["index_pending"] == 2  # one delete intent per memory
    await _close(client)


@pytest.mark.asyncio
async def test_erasure_ignores_foreign_memories_but_receipts_them(no_chroma):
    attacker_id = uuid.uuid4()
    victim_memory_id = uuid.uuid4()

    engine, Session = _session_factory()
    async with Session() as db:
        db.add(User(id=victim_memory_id, email=f"{victim_memory_id}@example.com",
                    hashed_password="x", is_verified=True, is_active=True))
        db.add(_memory_row(victim_memory_id, "belongs to someone else"))
        await db.commit()
    await engine.dispose()

    engine2, Session2 = _session_factory()
    async with Session2() as db2:
        db2.add(User(id=attacker_id, email=f"{attacker_id}@example.com", hashed_password="x",
                     is_verified=True, is_active=True))
        await db2.commit()
    await engine2.dispose()

    client = await _auth_client(attacker_id)
    async with client as c:
        resp = await c.post(
            "/api/v1/erasure-receipts",
            json={"memory_ids": [str(victim_memory_id)]},
        )
        assert resp.status_code == 201
        target = resp.json()["detail"]["targets"][0]
        assert target["status"] == "not_found_or_foreign"
    await _close(client)
