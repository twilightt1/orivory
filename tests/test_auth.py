"""Auth endpoint tests."""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_register_success(client: AsyncClient):
    resp = await client.post("/api/v1/auth/register", json={
        "email": "test@example.com",
        "password": "SecurePass123",
    })
    assert resp.status_code == 201
    assert "message" in resp.json()


@pytest.mark.asyncio
async def test_register_duplicate_email(client: AsyncClient):
    payload = {"email": "dup@example.com", "password": "SecurePass123"}
    await client.post("/api/v1/auth/register", json=payload)
    resp = await client.post("/api/v1/auth/register", json=payload)
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_login_unverified_when_an_email_provider_is_configured(
    client: AsyncClient, monkeypatch
):
    """SendGrid configured ⇒ the OTP gate applies: an unverified login is 403."""
    monkeypatch.setattr("app.services.auth_service.settings.SENDGRID_API_KEY", "SG.test-key")
    await client.post("/api/v1/auth/register", json={
        "email": "unverified@example.com",
        "password": "SecurePass123",
    })
    resp = await client.post("/api/v1/auth/login", json={
        "email": "unverified@example.com",
        "password": "SecurePass123",
    })
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_login_without_an_email_provider_trusts_local_registration(
    client: AsyncClient,
    monkeypatch,
):
    """No SENDGRID_API_KEY (the self-host default): there is no way to deliver
    a verification mail, so local registration is trusted and login works.

    Pinned hermetically: a developer whose .env carries any key sees the OTP
    gate instead (the sibling test above)."""
    monkeypatch.setattr("app.services.auth_service.settings.SENDGRID_API_KEY", "")
    await client.post("/api/v1/auth/register", json={
        "email": "trusted@example.com",
        "password": "SecurePass123",
    })
    resp = await client.post("/api/v1/auth/login", json={
        "email": "trusted@example.com",
        "password": "SecurePass123",
    })
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_login_wrong_password(client: AsyncClient):
    resp = await client.post("/api/v1/auth/login", json={
        "email": "test@example.com",
        "password": "wrongpassword",
    })
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_forgot_password_unknown_email(client: AsyncClient):

    resp = await client.post("/api/v1/auth/forgot-password", json={
        "email": "nobody@example.com"
    })
    assert resp.status_code == 200
