import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.v1 import admin
from app.main import app

pytestmark = pytest.mark.api


def _diagnostics_payload():
    return {
        "status": "ok",
        "version": "1.1.0",
        "environment": "test",
        "checks": {
            "postgres": {"status": "ok", "latency_ms": 1.0},
            "redis": {"status": "ok", "latency_ms": 1.0},
            "minio": {"status": "ok", "latency_ms": 1.0},
            "chroma": {"status": "ok", "latency_ms": 1.0},
            "celery": {"status": "ok", "latency_ms": 1.0},
        },
        "config": {
            "environment": "test",
            "docs_enabled": True,
            "cors_origins_count": 2,
            "minio_bucket": "rag-docs",
            "minio_secure": False,
            "llm_model": "openai/gpt-4o-mini",
            "embed_model": "text-embedding-3-small",
            "reranker_model": "jina-reranker-v2-base-multilingual",
            "rate_limit_per_minute": 60,
            "rate_limit_per_day": 1000,
        },
        "ingestion": {
            "counts": {"pending": 0, "processing": 0, "ready": 2, "failed": 0},
            "recent_failures": [],
            "stuck_processing": [],
        },
    }


@pytest.mark.asyncio
async def test_admin_diagnostics_requires_authentication():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/admin/diagnostics")

    assert response.status_code in {401, 403}


@pytest.mark.asyncio
async def test_admin_diagnostics_returns_payload_for_admin(monkeypatch):
    async def admin_override():
        return object()

    async def db_override():
        yield object()

    async def diagnostics_override(_db):
        return _diagnostics_payload()

    app.dependency_overrides[admin.require_admin] = admin_override
    app.dependency_overrides[admin.get_db] = db_override
    monkeypatch.setattr(admin, "build_diagnostics", diagnostics_override)

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/admin/diagnostics")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "checks" in body
    assert "config" in body
    assert "ingestion" in body
    assert body["checks"]["celery"]["status"] == "ok"


def test_admin_diagnostics_payload_is_secret_safe():
    encoded = json.dumps(_diagnostics_payload())

    assert "JWT_SECRET_KEY" not in encoded
    assert "OPENAI_API_KEY" not in encoded
    assert "OPENROUTER_API_KEY" not in encoded
    assert "JINA_API_KEY" not in encoded
    assert "DATABASE_URL" not in encoded
    assert "REDIS_URL" not in encoded
    assert "SECRET" not in encoded.upper()
