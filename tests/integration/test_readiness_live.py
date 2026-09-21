from __future__ import annotations

import pytest

from app import storage
from app.database import IS_SQLITE
from app.services.health_service import check_readiness

pytestmark = [pytest.mark.integration, pytest.mark.requires_infra]

# The lite readiness key-map (app/services/health_service.py, SQLite branch).
# The Postgres shape (postgres/redis/minio/qdrant/mcp_hub) went with the
# full-stack services.
LITE_CHECKS = {"sqlite", "redis", "storage", "qdrant", "mcp_hub"}


@pytest.mark.asyncio
async def test_live_readiness_reports_all_dependencies_ok():
    if not IS_SQLITE:
        pytest.skip("the live readiness contract here is the lite (SQLite) key-map")
    await storage.ensure_bucket()

    payload = await check_readiness()

    assert payload["status"] == "ok"
    assert set(payload["checks"]) == LITE_CHECKS
    assert all(check["status"] == "ok" for check in payload["checks"].values())
