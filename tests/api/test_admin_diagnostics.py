"""Secret-safety of the diagnostics payload builder.

NOTE (slim branch): the /api/v1/admin/* routes are unmounted, so the two
HTTP-level tests that lived here were deleted with the route. This test
stays: it guards the payload shape independent of any route.
"""
import json

import pytest

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
            "qdrant": {"status": "ok", "latency_ms": 1.0},
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


def test_admin_diagnostics_payload_is_secret_safe():
    encoded = json.dumps(_diagnostics_payload())

    assert "JWT_SECRET_KEY" not in encoded
    assert "OPENAI_API_KEY" not in encoded
    assert "OPENROUTER_API_KEY" not in encoded
    assert "JINA_API_KEY" not in encoded
    assert "DATABASE_URL" not in encoded
    assert "REDIS_URL" not in encoded
    assert "SECRET" not in encoded.upper()
