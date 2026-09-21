import httpx
import pytest

from app.services import health_service

pytestmark = pytest.mark.service

@pytest.mark.asyncio
async def test_check_readiness_lite_key_set(monkeypatch):
    """The one-container key set: sqlite/storage, never postgres/minio."""
    async def ok():
        return None

    monkeypatch.setattr(health_service, "_check_sqlite", ok)
    monkeypatch.setattr(health_service, "_check_redis", ok)
    monkeypatch.setattr(health_service, "_check_storage", ok)
    monkeypatch.setattr(health_service, "_check_qdrant", ok)
    monkeypatch.setattr(health_service, "_check_mcp_hub", ok)

    result = await health_service.check_readiness()

    assert result["status"] == "ok"
    assert set(result["checks"]) == {"sqlite", "redis", "storage", "qdrant", "mcp_hub"}
    assert all(check["status"] == "ok" for check in result["checks"].values())


@pytest.mark.asyncio
async def test_check_readiness_degraded_when_dependency_fails(monkeypatch):
    async def ok():
        return None

    async def failed():
        raise RuntimeError("connection refused to test dependency")

    monkeypatch.setattr(health_service, "_check_sqlite", ok)
    monkeypatch.setattr(health_service, "_check_redis", ok)
    monkeypatch.setattr(health_service, "_check_storage", ok)
    monkeypatch.setattr(health_service, "_check_qdrant", failed)
    monkeypatch.setattr(health_service, "_check_mcp_hub", ok)

    result = await health_service.check_readiness()

    assert result["status"] == "degraded"
    assert result["checks"]["qdrant"]["status"] == "failed"
    assert "connection refused" in result["checks"]["qdrant"]["error"]
    assert result["checks"]["sqlite"]["status"] == "ok"


def test_sanitize_error_limits_length_and_removes_newlines():
    error = RuntimeError("line one\n" + "x" * 500)

    message = health_service._sanitize_error(error)

    assert "\n" not in message
    assert len(message) == 300


@pytest.mark.asyncio
async def test_check_qdrant_server_mode_probes_readyz_with_a_2s_bound(monkeypatch):
    """Server mode = GET {QDRANT_URL}/readyz with a 2s bound; non-200 fails.

    Nothing else pins that endpoint without live infra, so a future edit could
    silently retarget the probe.
    """
    from app.retrieval import vector_backend

    urls: list[str] = []
    timeouts: list[float] = []
    statuses: list[int] = [200]

    class _Response:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code != 200:
                raise httpx.HTTPStatusError(
                    "not ready",
                    request=httpx.Request("GET", urls[-1]),
                    response=httpx.Response(self.status_code),
                )

    class _Client:
        def __init__(self, timeout: float) -> None:
            timeouts.append(timeout)

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *exc_info: object) -> bool:
            return False

        async def get(self, url: str) -> _Response:
            urls.append(url)
            return _Response(statuses[0])

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    monkeypatch.setattr(vector_backend, "is_local_mode", lambda: False)
    monkeypatch.setattr(health_service.settings, "QDRANT_URL", "http://qdrant.test:6333/")

    await health_service._check_qdrant()

    assert urls == ["http://qdrant.test:6333/readyz"]
    assert timeouts == [2.0]

    statuses[0] = 500
    with pytest.raises(httpx.HTTPStatusError):
        await health_service._check_qdrant()
