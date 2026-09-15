from __future__ import annotations

import pytest

from app import storage
from app.services.health_service import check_readiness

pytestmark = [pytest.mark.integration, pytest.mark.requires_infra]


@pytest.mark.asyncio
async def test_live_readiness_reports_all_dependencies_ok():
    await storage.ensure_bucket()

    payload = await check_readiness()

    assert payload["status"] == "ok"
    assert set(payload["checks"]) == {"postgres", "redis", "minio", "qdrant", "mcp_hub"}
    assert all(check["status"] == "ok" for check in payload["checks"].values())
