from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from app.services import diagnostics_service

pytestmark = pytest.mark.service


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _FakeDb:
    def __init__(self, results):
        self._results = list(results)

    async def execute(self, _query):
        return self._results.pop(0)


class _StatusRow:
    def __init__(self, status, count):
        self.status = status
        self.count = count

    def __iter__(self):
        yield self.status
        yield self.count


class _DocRow:
    def __init__(self, *, status="failed", minutes_old=1, error_msg="boom"):
        now = datetime.now(UTC)
        self.id = uuid4()
        self.filename = f"{status}.md"
        self.status = status
        self.error_msg = error_msg
        self.created_at = now - timedelta(minutes=minutes_old + 1)
        self.updated_at = now - timedelta(minutes=minutes_old)


def test_build_config_summary_excludes_secret_values(monkeypatch):
    monkeypatch.setattr(diagnostics_service.settings, "JWT_SECRET_KEY", "super-secret-jwt")
    monkeypatch.setattr(diagnostics_service.settings, "OPENAI_API_KEY", "sk-secret")
    monkeypatch.setattr(diagnostics_service.settings, "DATABASE_URL", "sqlite:////data/secret.db")
    monkeypatch.setattr(diagnostics_service.settings, "SENDGRID_API_KEY", "SG.secret")

    summary = diagnostics_service.build_config_summary()
    flattened = str(summary)

    assert "super-secret-jwt" not in flattened
    assert "sk-secret" not in flattened
    assert "sqlite:////data/secret.db" not in flattened
    assert "SG.secret" not in flattened
    assert summary["llm_model"]
    assert summary["storage_backend"]


@pytest.mark.asyncio
async def test_get_document_ingestion_summary_aggregates_counts_and_recent_docs():
    failed = _DocRow(status="failed", error_msg="parse failed")
    stuck = _DocRow(status="processing", minutes_old=20, error_msg=None)
    db = _FakeDb(
        [
            _Result([_StatusRow("ready", 2), _StatusRow("failed", 1), _StatusRow("processing", 1)]),
            _Result([failed]),
            _Result([stuck]),
        ]
    )

    summary = await diagnostics_service.get_document_ingestion_summary(db)

    assert summary["counts"]["ready"] == 2
    assert summary["counts"]["failed"] == 1
    assert summary["counts"]["pending"] == 0
    assert summary["recent_failures"][0]["filename"] == "failed.md"
    assert summary["stuck_processing"][0]["filename"] == "processing.md"


@pytest.mark.asyncio
async def test_build_diagnostics_degrades_when_dependency_fails(monkeypatch):
    async def fake_readiness(extra_checkers=None):
        assert extra_checkers is None  # celery check removed: no broker on the slim branch
        return {
            "postgres": {"status": "ok", "latency_ms": 1.0},
            "redis": {"status": "failed", "latency_ms": 1.0, "error": "down"},
        }

    async def fake_ingestion(_db):
        return {"counts": {}, "recent_failures": [], "stuck_processing": []}

    async def fake_outbox(_db):
        return {"by_status": {"pending": 0, "done": 0, "blocked": 0}, "by_kind": {}}

    monkeypatch.setattr(diagnostics_service, "run_readiness_checks", fake_readiness)
    monkeypatch.setattr(diagnostics_service, "get_document_ingestion_summary", fake_ingestion)
    monkeypatch.setattr(diagnostics_service, "get_index_outbox_summary", fake_outbox)

    result = await diagnostics_service.build_diagnostics(object())

    assert result["status"] == "degraded"
    assert result["checks"]["redis"]["status"] == "failed"
    assert result["checks"]["celery"]["status"] == "dormant"
    assert result["config"]["celery_queues"] == ["default", "ingestion", "email"]
    assert result["index_outbox"]["by_status"]["blocked"] == 0  # C2: the class is in the payload
