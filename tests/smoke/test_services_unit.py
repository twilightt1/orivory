"""Unit tests for smoke-test assertion logic (no infra needed).

The CI smoke job runs tests/smoke/test_services.py against live compose
services, but its assertions must match the REAL vendor APIs. These tests
drive the same test functions with stubbed HTTP so the assertions are
pinned without Docker:
- Chroma v2 heartbeat returns {"nanosecond heartbeat": N} (no "success")
- Chroma v2 version returns a bare string like "1.0.0" (not {"version": …})
- MinIO S3 port answers 400 on the console route (reachable, wrong route)
- Unreachable services must SKIP, not crash (requests has no TimeoutError
  attribute — handlers must catch requests.exceptions.RequestException)
"""
import pytest

from tests.smoke import test_services as svc

pytestmark = pytest.mark.unit


class _Resp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture
def _live(monkeypatch):
    monkeypatch.setattr(svc, "DOCKER_AVAILABLE", True)


def test_chroma_heartbeat_accepts_real_payload(_live, monkeypatch):
    import requests

    monkeypatch.setattr(
        requests, "get", lambda *a, **k: _Resp(200, {"nanosecond heartbeat": 1})
    )
    svc.TestChromaDBHealth().test_chroma_heartbeat({})


def test_chroma_version_accepts_bare_string(_live, monkeypatch):
    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(200, "1.0.0"))
    svc.TestChromaDBHealth().test_chroma_version({})


def test_minio_api_accepts_400_as_reachable(_live, monkeypatch):
    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(400, None))
    svc.TestMinIOHealth().test_minio_api({})


def test_unreachable_service_skips_not_crashes(_live, monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.exceptions.ConnectTimeout("down")

    monkeypatch.setattr(requests, "get", boom)
    with pytest.raises(pytest.skip.Exception):
        svc.TestChromaDBHealth().test_chroma_heartbeat({})
    with pytest.raises(pytest.skip.Exception):
        svc.TestChromaDBHealth().test_chroma_version({})
    with pytest.raises(pytest.skip.Exception):
        svc.TestMinIOHealth().test_minio_health({})
    with pytest.raises(pytest.skip.Exception):
        svc.TestMinIOHealth().test_minio_api({})
