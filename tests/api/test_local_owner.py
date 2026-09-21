"""The identity layer after account auth: ONE local owner.

Account auth is gone (register / verify / login / OAuth / JWT), so there is
nobody to authenticate: a local request IS the owner, and the API must serve
it without any token. Tokens do NOT belong on this surface — an agent token is
a credential for the MCP endpoint and ``POST /api/v1/imports``, where it
resolves to its own owner and its scopes are enforced (tests/api/test_agent_imports.py,
tests/mcp_hub/); REST refuses any Authorization header.

These tests drive the real ASGI app through the shared ``client`` fixture:
no dependency overrides, no Authorization header — exactly how a local
deployment reaches these routes.
"""
from __future__ import annotations

import uuid

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy import select

from app.config import settings
from app.models.agent_client import AgentClient
from app.models.user import User
from app.services.agent_token_service import generate_token, hash_token
from app.utils.dependencies import get_current_user

pytestmark = pytest.mark.api


def _owner_email() -> str:
    return settings.LOCAL_OWNER_EMAIL.strip().lower()


def _bearer(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


async def _agents_owner(
    db, name: str, *, active: bool = True, scopes: tuple[str, ...] = ("memory:read", "memory:write")
) -> tuple[User, str]:
    """A second user + an agent client of theirs; returns (owner, token)."""
    owner = User(
        id=uuid.uuid4(),
        email=f"agent-owner-{uuid.uuid4().hex[:8]}@example.com",
        is_verified=True,
        is_active=True,
    )
    token = generate_token()
    db.add(owner)
    db.add(AgentClient(
        user_id=owner.id,
        name=name,
        token_hash=hash_token(token),
        scopes=list(scopes),
        status="active" if active else "revoked",
    ))
    await db.commit()
    return owner, token


# ── the local owner ──────────────────────────────────────────────────────────

async def test_no_token_serves_the_local_owner(client, db):
    """No Authorization header, no 401: a local request IS the owner."""
    resp = await client.get("/api/v1/users/me")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["email"] == _owner_email()
    assert body["role"] == "admin"
    assert body["is_verified"] is True
    assert body["onboarding_done"] is True
    # a real users row, not a synthesized object
    row = await db.scalar(select(User).where(User.email == _owner_email()))
    assert row is not None
    assert str(row.id) == body["id"]


async def test_the_owner_row_is_reused_not_duplicated(client, db):
    first = await client.get("/api/v1/users/me")
    second = await client.get("/api/v1/users/me")

    assert first.json()["id"] == second.json()["id"]
    rows = (await db.execute(
        select(User).where(User.email == _owner_email())
    )).scalars().all()
    assert len(rows) == 1


async def test_an_existing_owner_row_is_left_alone(client, db):
    """Get-or-create, not reset-to-default: a row that already exists is the
    operator's data."""
    from app.services.local_owner import ensure_local_owner

    owner = await ensure_local_owner(db)
    owner.display_name, owner.role, owner.is_verified = "Keep me", "user", False
    await db.commit()

    resp = await client.get("/api/v1/users/me")

    assert resp.status_code == 200, resp.text
    assert resp.json()["id"] == str(owner.id)
    assert resp.json()["display_name"] == "Keep me"
    assert resp.json()["role"] == "user"
    assert resp.json()["is_verified"] is False
    # the rest of the suite shares this database: hand the defaults back
    owner.display_name, owner.role, owner.is_verified = "Owner", "admin", True
    await db.commit()


# ── the token trunk lives on MCP and /imports, NOT on the REST surface ───────

async def test_a_read_only_agent_token_is_refused_on_the_rest_surface(db):
    """A ``memory:read`` token must not reach REST.

    REST resolves identity without scopes, so answering with the token's owner
    would let a read-only client mint itself a write token, revoke a sibling's,
    or act as the owner. Tokens are credentials for MCP and
    ``POST /api/v1/imports``, where scopes ARE enforced — see
    tests/api/test_agent_imports.py and tests/mcp_hub/.
    """
    _, token = await _agents_owner(db, "read-only-agent", scopes=("memory:read",))

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(_bearer(token), db)

    assert exc_info.value.status_code == 401


async def test_revoked_agent_token_is_not_downgraded_to_the_owner(db):
    """A revoked token must stop working — silently serving the owner would
    make revocation meaningless."""
    _, token = await _agents_owner(db, "revoked-agent", active=False)

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(_bearer(token), db)

    assert exc_info.value.status_code == 401


async def test_a_stale_jwt_is_refused_on_the_rest_surface(db):
    """The product has no JWTs any more: one is stale config, and the honest
    answer to stale credentials is 401, not silently acting as the owner."""
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(_bearer("stale.jwt.value"), db)

    assert exc_info.value.status_code == 401


# ── the trimmed /me surface ──────────────────────────────────────────────────

async def test_retention_settings_round_trip_without_a_token(client):
    written = await client.patch(
        "/api/v1/users/me/settings",
        json={"retention_enabled": True, "retention_days": 30},
    )
    assert written.status_code == 200, written.text
    assert written.json() == {"retention_enabled": True, "retention_days": 30}

    read_back = await client.get("/api/v1/users/me")
    assert read_back.json()["retention_enabled"] is True
    assert read_back.json()["retention_days"] == 30


async def test_profile_and_password_routes_are_gone(client):
    """PATCH /me and POST /me/change-password went with account auth."""
    assert (await client.patch("/api/v1/users/me", json={"display_name": "x"})).status_code == 405
    assert (await client.post(
        "/api/v1/users/me/change-password",
        json={"current_password": "a", "new_password": "b" * 8},
    )).status_code == 404


@pytest.mark.parametrize("path", [
    "/api/v1/auth/register",
    "/api/v1/auth/login",
    "/api/v1/auth/verify-email",
    "/api/v1/auth/forgot-password",
    "/api/v1/auth/google/authorize",
])
async def test_auth_routes_are_gone(client, path):
    resp = await client.post(path, json={})

    assert resp.status_code == 404
