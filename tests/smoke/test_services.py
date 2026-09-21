"""Smoke tests for the lite Docker Compose stack.

The product is ONE container (`docker compose up -d`): the API with its MCP hub,
embedded Qdrant, SQLite store and filesystem uploads, published on
``localhost:8000``. These tests verify that container really serves — and SKIP
(never crash) when nothing is listening, so a bare `pytest tests/smoke` outside
a compose stack stays honest instead of red.

Mark: @pytest.mark.smoke
"""
from __future__ import annotations

import os

import pytest

API_BASE_URL = os.environ.get("ORIVORY_SMOKE_API", "http://localhost:8000")

# The lite readiness key-map (`app/services/health_service.py`, SQLite branch).
# ``mcp_hub`` joins it only while the hub is enabled — it is by default, so the
# set is checked as a floor, not as an exact equality.
CORE_READINESS_CHECKS = {"sqlite", "redis", "storage", "qdrant"}
# The dropped full-stack services: their names must never come back into the
# lite readiness payload (a rename that resurrects one is a real regression).
FULL_STACK_CHECKS = {"postgres", "minio"}


def _get(path: str, timeout: float = 5.0):
    """GET the lite API, or SKIP when it is not answering (never crash)."""
    import requests

    try:
        return requests.get(f"{API_BASE_URL}{path}", timeout=timeout)
    except requests.exceptions.RequestException:
        pytest.skip(
            f"lite API not available at {API_BASE_URL} — run 'docker compose up -d'"
        )


@pytest.mark.smoke
def test_lite_health_endpoint(docker_services):
    """API /health endpoint should respond."""
    response = _get("/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data or "healthy" in data


@pytest.mark.smoke
def test_lite_ready_reports_every_dependency_ok(docker_services):
    """``/ready`` measures each lite dependency in-process — all must be ok."""
    response = _get("/ready")
    payload = response.json()

    assert payload["status"] == "ok", payload
    checks = payload["checks"]
    assert CORE_READINESS_CHECKS <= set(checks), checks
    for name, check in checks.items():
        assert check["status"] == "ok", (name, check)
    assert FULL_STACK_CHECKS.isdisjoint(checks), (
        f"the lite stack has no {sorted(FULL_STACK_CHECKS)} service to report: {checks}")


@pytest.mark.smoke
def test_api_docs_accessible(docker_services):
    """API documentation should be accessible (development/test environments)."""
    response = _get("/docs")

    assert response.status_code in [200, 301, 302]
