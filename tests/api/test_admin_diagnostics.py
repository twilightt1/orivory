"""Secret-safety of the diagnostics payload builder.

NOTE: the admin routes went with the full-stack surface, so the two
HTTP-level tests that lived here were deleted with them. This test stays:
it guards the payload shape independent of any route.
"""
import json

import pytest

from app.config import settings
from app.services.diagnostics_service import build_config_summary

pytestmark = pytest.mark.api


def _diagnostics_payload():
    return {
        "status": "ok",
        "version": "1.1.0",
        "environment": "test",
        "checks": {
            "sqlite": {"status": "ok", "latency_ms": 1.0},
            "redis": {"status": "ok", "latency_ms": 1.0},
            "storage": {"status": "ok", "latency_ms": 1.0},
            "qdrant": {"status": "ok", "latency_ms": 1.0},
            "celery": {"status": "ok", "latency_ms": 1.0},
        },
        "config": {
            "environment": "test",
            "docs_enabled": True,
            "cors_origins_count": 2,
            "storage_backend": "fs",
            "llm_model": "openai/gpt-4o-mini",
            "embed_model": "Snowflake/snowflake-arctic-embed-xs",
            "embed_dimensions": 384,
            "reranker_model": "gte-multilingual-reranker-base (local ONNX, int8)",
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
    assert "DATABASE_URL" not in encoded
    assert "SENDGRID_API_KEY" not in encoded
    assert "SECRET" not in encoded.upper()


def test_config_summary_reports_active_local_embedding(monkeypatch):
    monkeypatch.setattr(settings, "USE_LOCAL_EMBEDDINGS", True)
    monkeypatch.setattr(settings, "LOCAL_EMBED_MODEL", "arctic")

    summary = build_config_summary()

    assert summary["embed_model"] == "Snowflake/snowflake-arctic-embed-xs"
    assert summary["embed_dimensions"] == 384
