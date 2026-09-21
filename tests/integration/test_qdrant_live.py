from __future__ import annotations

import httpx
import pytest

from app.config import settings

pytestmark = [pytest.mark.integration, pytest.mark.requires_infra]


@pytest.mark.asyncio
async def test_live_qdrant_probe_is_ok_on_the_running_lite_app():
    """The vector store the app serves from is measured by the app itself.

    The lite stack owns Qdrant IN-PROCESS (QDRANT_MODE=local): there is no
    :6333 server left to probe, so the live question is the app's own
    /ready report — the embedded store must measure ``ok`` there.
    """
    url = f"{settings.API_BASE_URL.rstrip('/')}/ready"

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(url)

    assert response.status_code == 200, response.text
    payload = response.json()
    qdrant = payload["checks"]["qdrant"]
    assert qdrant["status"] == "ok", qdrant
