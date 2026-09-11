from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.document import Document
from app.services.health_service import CheckPayload, run_readiness_checks

DOCUMENT_TERMINAL_FAILURE_STATUSES = ("failed", "error")
DOCUMENT_IN_FLIGHT_STATUSES = ("pending", "processing")
RECENT_DOCUMENT_LIMIT = 5
STUCK_AFTER_MINUTES = 15
VERSION = "1.1.0"


_check_celery = None  # type: ignore[assignment]  # dormant on the slim branch: no broker, see build_diagnostics.


def build_config_summary() -> dict[str, Any]:
    origins = [origin.strip() for origin in settings.ALLOWED_ORIGINS.split(",") if origin.strip()]
    return {
        "environment": settings.ENVIRONMENT,
        "docs_enabled": settings.ENVIRONMENT != "production",
        "cors_origins_count": len(origins),
        "minio_bucket": settings.MINIO_BUCKET,
        "minio_secure": settings.MINIO_SECURE,
        "llm_model": settings.LLM_MODEL,
        "embed_model": settings.EMBED_MODEL,
        "embed_dimensions": settings.EMBED_DIMENSIONS,
        "reranker_model": settings.JINA_RERANKER_MODEL,
        "reranker_top_n": settings.JINA_RERANKER_TOP_N,
        "rate_limit_per_minute": settings.RATE_LIMIT_PER_MINUTE,
        "rate_limit_per_day": settings.RATE_LIMIT_PER_DAY,
        "celery_queues": ["default", "ingestion", "email"],
    }


def _serialize_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _document_ref(row: Any) -> dict[str, Any]:
    return {
        "id": str(row.id),
        "filename": row.filename,
        "status": row.status,
        "error_msg": row.error_msg,
        "created_at": _serialize_datetime(row.created_at),
        "updated_at": _serialize_datetime(row.updated_at),
    }


def _empty_status_counts() -> dict[str, int]:
    return {
        "pending": 0,
        "processing": 0,
        "ready": 0,
        "failed": 0,
        "error": 0,
    }


async def get_document_ingestion_summary(db: AsyncSession) -> dict[str, Any]:
    counts = _empty_status_counts()
    status_result = await db.execute(select(Document.status, func.count(Document.id)).group_by(Document.status))
    for status, count in status_result.all():
        counts[str(status)] = int(count or 0)

    recent_failures_result = await db.execute(
        select(
            Document.id,
            Document.filename,
            Document.status,
            Document.error_msg,
            Document.created_at,
            Document.updated_at,
        )
        .where(Document.status.in_(DOCUMENT_TERMINAL_FAILURE_STATUSES))
        .order_by(Document.updated_at.desc())
        .limit(RECENT_DOCUMENT_LIMIT)
    )
    recent_failures = [_document_ref(row) for row in recent_failures_result.all()]

    stuck_cutoff = datetime.now(UTC) - timedelta(minutes=STUCK_AFTER_MINUTES)
    stuck_result = await db.execute(
        select(
            Document.id,
            Document.filename,
            Document.status,
            Document.error_msg,
            Document.created_at,
            Document.updated_at,
        )
        .where(Document.status.in_(DOCUMENT_IN_FLIGHT_STATUSES), Document.updated_at < stuck_cutoff)
        .order_by(Document.updated_at.asc())
        .limit(RECENT_DOCUMENT_LIMIT)
    )
    stuck_processing = [_document_ref(row) for row in stuck_result.all()]

    return {
        "counts": counts,
        "recent_failures": recent_failures,
        "stuck_processing": stuck_processing,
        "stuck_after_minutes": STUCK_AFTER_MINUTES,
    }


async def build_diagnostics(db: AsyncSession) -> dict[str, Any]:
    # ponytail: no broker on the slim branch — the celery check is a dormant
    # key so the payload shape (and the secret-safety/API tests) holds.
    checks: dict[str, CheckPayload] = await run_readiness_checks()
    status = "ok" if all(check["status"] == "ok" for check in checks.values()) else "degraded"
    checks.setdefault("celery", {"status": "dormant", "latency_ms": 0.0})
    return {
        "status": status,
        "version": VERSION,
        "environment": settings.ENVIRONMENT,
        "checks": checks,
        "config": build_config_summary(),
        "ingestion": await get_document_ingestion_summary(db),
    }
