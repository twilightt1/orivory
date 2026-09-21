from __future__ import annotations

import pytest

from app import storage

pytestmark = [pytest.mark.integration, pytest.mark.requires_infra]


@pytest.mark.asyncio
async def test_live_storage_put_get_remove():
    """The live storage backend round-trips an object (filesystem on the lite
    stack: STORAGE_BACKEND=fs — the MinIO service was dropped)."""
    object_name = "integration/live-storage-check.txt"
    payload = b"supportmind live storage integration"

    await storage.ensure_bucket()
    assert await storage.bucket_exists()

    await storage.put_object(object_name, payload, "text/plain")
    try:
        assert await storage.get_object(object_name) == payload
    finally:
        await storage.remove_object(object_name)
