"""Behavioral tests for agent-token imports (the auto-capture path).

The imports endpoint accepts TWO auth modes: human JWT (default path) and
agent tokens (oa_...). These tests cover the agent-token path end-to-end
over HTTP: attribution, ledger recording, and auth boundaries.

Requires the test Postgres (ragdb_test) — same as test_hub_security.py.
"""
from __future__ import annotations

import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker as _mk_session
from sqlalchemy.ext.asyncio import create_async_engine as _mk_engine
from sqlalchemy.pool import NullPool as _NullPool

from app.database import get_db
from app.main import app
from app.models.agent_client import AgentClient
from app.models.user import User
from app.services.agent_token_service import generate_token, hash_token

pytestmark = pytest.mark.api


@pytest.fixture(autouse=True)
async def _require_test_database():
    """These suites manage their own loop-local engines (not the shared `db`
    fixture), so they probe Postgres directly and skip when it is down."""
    from tests.conftest import require_db_available

    await require_db_available()


async def _seed_agent(monkey=None, name="capture-agent",
                     scopes=("memory:read", "memory:write")):
    """Create a verified user + agent client on a loop-local engine and
    return its id/token. Each call builds its OWN engine inside the test
    coroutine — the app's global engine binds connections to whichever loop
    touched them first, so sharing it across pytest-asyncio's per-test
    loops deadlocks ("attached to a different loop")."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    eng = create_async_engine(
        "postgresql+asyncpg://postgres:password@localhost:55432/ragdb_test",
        poolclass=NullPool)
    Sess = async_sessionmaker(eng, class_=AsyncSession, expire_on_commit=False)
    user_id = uuid.uuid4()
    token = generate_token()
    client = AgentClient(
        user_id=user_id,
        name=name,
        token_hash=hash_token(token),
        scopes=list(scopes),
    )
    async with Sess() as db:
        db.add(User(id=user_id, email=f"{user_id}@example.com", hashed_password="x",
                    is_verified=True, is_active=True))
        db.add(client)
        await db.commit()
        client_id = client.id
    await eng.dispose()
    return {"user_id": user_id, "token": token, "client_id": client_id}


def _TestSession():
    eng = _mk_engine(
        "postgresql+asyncpg://postgres:password@localhost:55432/ragdb_test",
        poolclass=_NullPool,
    )
    return _mk_session(eng, class_=AsyncSession, expire_on_commit=False)()


async def _client():
    test_engine = _mk_engine(
        "postgresql+asyncpg://postgres:password@localhost:55432/ragdb_test",
        poolclass=_NullPool)
    test_session = _mk_session(test_engine, class_=AsyncSession,
                               expire_on_commit=False)

    async def _db_override():
        async with test_session() as session:
            yield session

    app.dependency_overrides[get_db] = _db_override
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_agent_token_import_succeeds_and_ledgers():
    agent_env = await _seed_agent()
    """The auto-capture path: an agent token (no JWT) can POST an import;
    the memories attribute to the agent's owner with requested_by=agent:name."""
    token = agent_env["token"]
    payload = {"session_id": "s-capture-1", "title": "OpenClaw session",
               "entries": [{"role": "user", "content": "auto-captured fact"}]}
    import json as jsonlib

    async with await _client() as c:
        resp = await c.post(
            "/api/v1/imports",
            files={"file": ("session.json", jsonlib.dumps(payload).encode(), "application/json")},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["created"] == 1

        # attribution: the memory carries requested_by = agent:<name>
        from app.models.memory import Memory

        async with _TestSession() as db:
            row = (await db.execute(
                select(Memory).where(Memory.user_id == agent_env["user_id"])
            )).scalars().first()
            assert row is not None
            assert row.extra_metadata["import"]["requested_by"] == "agent:capture-agent"


@pytest.mark.asyncio
async def test_agent_token_import_ledgers_an_import_action():
    agent_env = await _seed_agent()
    """The access ledger records agent imports — governance parity with MCP writes."""
    from app.models.memory_access_log import MemoryAccessLog

    token = agent_env["token"]
    payload = {"memory_entries": [{"session_id": "s-ledger",
                                   "entries": [{"role": "user", "content": "ledgered import"}]}]}
    import json as jsonlib

    async with await _client() as c:
        await c.post(
            "/api/v1/imports",
            files={"file": ("session.json", jsonlib.dumps(payload).encode(), "application/json")},
            headers={"Authorization": f"Bearer {token}"},
        )

    async with _TestSession() as db:
        rows = (await db.execute(
            select(MemoryAccessLog).where(
                MemoryAccessLog.agent_client_id == agent_env["client_id"]
            )
        )).scalars().all()
        assert rows, "agent import must appear in the access ledger"


@pytest.mark.asyncio
async def test_agent_token_import_requires_write_scope():
    """An agent with memory:read only must NOT be able to import (a write)."""
    user_id = uuid.uuid4()
    token = generate_token()
    client = AgentClient(
        user_id=user_id,
        name="readonly-agent",
        token_hash=hash_token(token),
        scopes=["memory:read"],  # no write
    )
    from app.database import AsyncSessionLocal as _Session2

    async with _Session2() as db:
        db.add(User(id=user_id, email=f"{user_id}@example.com", hashed_password="x",
                    is_verified=True, is_active=True))
        db.add(client)
        await db.commit()

    import json as jsonlib

    async with await _client() as c:
        resp = await c.post(
            "/api/v1/imports",
            files={"file": ("session.json", jsonlib.dumps({"memory_entries": []}).encode(), "application/json")},
            headers={"Authorization": f"Bearer {token}"},
        )
        # no JWT, invalid-for-write agent token → 401 (auth didn't resolve to a writer)
        assert resp.status_code in (401, 403), resp.text


@pytest.mark.asyncio
async def test_revoked_agent_token_import_fails():
    agent_env = await _seed_agent()
    """A revoked agent's token must stop working immediately — even for imports."""
    token = agent_env["token"]

    from app.database import AsyncSessionLocal as _Session3

    async with _Session3() as db:
        eng = _mk_engine(
        "postgresql+asyncpg://postgres:password@localhost:55432/ragdb_test",
        poolclass=_NullPool)
    Sess = _mk_session(eng, class_=AsyncSession, expire_on_commit=False)
    async with Sess() as db:
        row = await db.get(AgentClient, agent_env["client_id"])
        row.status = "revoked"
        await db.commit()
    await eng.dispose()

    import json as jsonlib

    async with await _client() as c:
        resp = await c.post(
            "/api/v1/imports",
            files={"file": ("session.json", jsonlib.dumps({"memory_entries": []}).encode(), "application/json")},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 401, resp.text
