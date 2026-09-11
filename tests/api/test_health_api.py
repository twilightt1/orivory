import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app

pytestmark = pytest.mark.api


@pytest.mark.asyncio
async def test_health_returns_ok():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "1.1.0"}


@pytest.mark.asyncio
async def test_ready_returns_ok_when_dependencies_are_ready(monkeypatch):
    async def ready_payload():
        return {
            "status": "ok",
            "version": "1.0.0",
            "checks": {
                "postgres": {"status": "ok", "latency_ms": 1.0},
                "redis": {"status": "ok", "latency_ms": 1.0},
                "minio": {"status": "ok", "latency_ms": 1.0},
                "chroma": {"status": "ok", "latency_ms": 1.0},
            },
        }

    monkeypatch.setattr("app.services.health_service.check_readiness", ready_payload)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_ready_returns_503_when_dependencies_are_degraded(monkeypatch):
    async def degraded_payload():
        return {
            "status": "degraded",
            "version": "1.1.0",
            "checks": {
                "postgres": {"status": "ok", "latency_ms": 1.0},
                "redis": {"status": "ok", "latency_ms": 1.0},
                "minio": {"status": "ok", "latency_ms": 1.0},
                "chroma": {
                    "status": "failed",
                    "latency_ms": 10.0,
                    "error": "connection refused",
                },
            },
        }

    monkeypatch.setattr("app.services.health_service.check_readiness", degraded_payload)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["chroma"]["status"] == "failed"


async def test_ready_includes_mcp_hub_check_when_enabled():
    """MCP hub readiness: flag on → check present and ok (offline probe)."""
    from app.services.health_service import check_readiness

    payload = await check_readiness()
    assert "mcp_hub" in payload["checks"]
    assert payload["checks"]["mcp_hub"]["status"] == "ok"
