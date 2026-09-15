from __future__ import annotations

import httpx
import pytest

from app.config import settings

pytestmark = [pytest.mark.integration, pytest.mark.requires_infra]


@pytest.mark.asyncio
async def test_live_qdrant_readyz():
    """The store the app serves from answers its own readiness endpoint."""
    url = f"{settings.QDRANT_URL.rstrip('/')}/readyz"

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(url)

    assert response.status_code == 200
    # Real Qdrant /readyz body: plain text "all shards are ready".
    assert "ready" in response.text
