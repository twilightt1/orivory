"""Unit tests for smoke-test assertion logic (no infra needed).

The CI smoke job runs ``tests/smoke/test_services.py`` against the live lite
compose stack, so its assertions are driven here with stubbed HTTP and pinned
without Docker:
- a healthy ``/ready`` payload is accepted,
- a DEGRADED payload FAILS (it must never read as a skip),
- an unreachable API SKIPS, never crashes (``requests`` has no TimeoutError
  attribute — the handler must catch ``requests.exceptions.RequestException``).
"""
import pytest

from tests.smoke import test_services as svc

pytestmark = pytest.mark.unit


class _Resp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


@pytest.fixture
def _get(monkeypatch):
    def _stub(response):
        import requests

        monkeypatch.setattr(requests, "get", lambda *a, **k: response)

    return _stub


def _healthy_payload() -> dict:
    return {
        "status": "ok",
        "checks": {
            "sqlite": {"status": "ok", "latency_ms": 0.4},
            "redis": {"status": "ok", "latency_ms": 0.1},
            "storage": {"status": "ok", "latency_ms": 0.2},
            "qdrant": {"status": "ok", "latency_ms": 1.2},
            "mcp_hub": {"status": "ok", "latency_ms": 0.3},
        },
    }


def test_health_accepts_real_payload(_get):
    _get(_Resp(200, {"status": "healthy"}))
    svc.test_lite_health_endpoint({})


def test_ready_accepts_a_healthy_payload(_get):
    _get(_Resp(200, _healthy_payload()))
    svc.test_lite_ready_reports_every_dependency_ok({})


def test_degraded_readiness_fails_not_skips(_get):
    payload = _healthy_payload()
    payload["status"] = "degraded"
    payload["checks"]["qdrant"] = {"status": "failed", "error": "store down"}
    _get(_Resp(503, payload))
    with pytest.raises(AssertionError):
        svc.test_lite_ready_reports_every_dependency_ok({})


def test_a_resurrected_full_stack_check_fails(_get):
    payload = _healthy_payload()
    payload["checks"]["postgres"] = {"status": "ok", "latency_ms": 1.0}
    _get(_Resp(200, payload))
    with pytest.raises(AssertionError):
        svc.test_lite_ready_reports_every_dependency_ok({})


def test_unreachable_api_skips_not_crashes(monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.exceptions.ConnectTimeout("down")

    monkeypatch.setattr(requests, "get", boom)
    with pytest.raises(pytest.skip.Exception):
        svc.test_lite_health_endpoint({})
    with pytest.raises(pytest.skip.Exception):
        svc.test_lite_ready_reports_every_dependency_ok({})
    with pytest.raises(pytest.skip.Exception):
        svc.test_api_docs_accessible({})
