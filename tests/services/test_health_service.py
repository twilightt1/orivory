import pytest

from app.database import IS_SQLITE
from app.services import health_service

# These two assert the full-stack checker map (postgres + minio + …); lite mode
# deliberately exposes its own map (sqlite + storage + …), so they only apply
# under a Postgres-shaped DATABASE_URL.
pytestmark = [
    pytest.mark.service,
    pytest.mark.skipif(IS_SQLITE, reason="full-stack readiness map; lite mode exposes its own checks"),
]


@pytest.mark.asyncio
async def test_check_readiness_ok(monkeypatch):
    async def ok():
        return None

    monkeypatch.setattr(health_service, "_check_postgres", ok)
    monkeypatch.setattr(health_service, "_check_redis", ok)
    monkeypatch.setattr(health_service, "_check_minio", ok)
    monkeypatch.setattr(health_service, "_check_chroma", ok)
    monkeypatch.setattr(health_service, "_check_mcp_hub", ok)

    result = await health_service.check_readiness()

    assert result["status"] == "ok"
    assert set(result["checks"]) == {"postgres", "redis", "minio", "chroma", "mcp_hub"}
    assert all(check["status"] == "ok" for check in result["checks"].values())


@pytest.mark.asyncio
async def test_check_readiness_degraded_when_dependency_fails(monkeypatch):
    async def ok():
        return None

    async def failed():
        raise RuntimeError("connection refused to test dependency")

    monkeypatch.setattr(health_service, "_check_postgres", ok)
    monkeypatch.setattr(health_service, "_check_redis", ok)
    monkeypatch.setattr(health_service, "_check_minio", ok)
    monkeypatch.setattr(health_service, "_check_chroma", failed)
    monkeypatch.setattr(health_service, "_check_mcp_hub", ok)

    result = await health_service.check_readiness()

    assert result["status"] == "degraded"
    assert result["checks"]["chroma"]["status"] == "failed"
    assert "connection refused" in result["checks"]["chroma"]["error"]
    assert result["checks"]["postgres"]["status"] == "ok"


def test_sanitize_error_limits_length_and_removes_newlines():
    error = RuntimeError("line one\n" + "x" * 500)

    message = health_service._sanitize_error(error)

    assert "\n" not in message
    assert len(message) == 300
