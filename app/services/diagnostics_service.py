from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.document import Document
from app.models.index_outbox import IndexOutbox
from app.services.health_service import CheckPayload, run_readiness_checks

DOCUMENT_TERMINAL_FAILURE_STATUSES = ("failed", "error")
DOCUMENT_IN_FLIGHT_STATUSES = ("pending", "processing")
RECENT_DOCUMENT_LIMIT = 5
STUCK_AFTER_MINUTES = 15
VERSION = "1.1.0"

# The durable index outbox's own vocabularies (``models/index_outbox.py``):
# status pending|done|blocked, kind memory|chunk. Seeded so a class that
# currently has no rows is still a visible zero in the payload.
OUTBOX_STATUSES = ("pending", "done", "blocked")
OUTBOX_KINDS = ("memory", "chunk")


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
        # The per-call CAP on the reranker's own answer (`min(request top_k,
        # this)`), NOT the rerank window. The window is the retrieval pool:
        # `top_k x RETRIEVAL_RERANK_POOL_MULTIPLIER` (2.0 by default, so 20 for
        # the default top_k=10 — which is why the cap defaults to 20).
        # docs/OPERATIONS_RUNBOOK.md (P2 section) spells this out for operators.
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


async def get_index_outbox_summary(db: AsyncSession) -> dict[str, Any]:
    """The durable index outbox at a glance: what is still owed to the vector store.

    Counts by BOTH status and kind, plus the two age signals an operator needs
    (``stuck_pending`` past the same 15-minute threshold as document ingestion,
    and ``oldest_pending_at`` — the ISO timestamp of the oldest pending write's
    ``created_at``, null when nothing is pending).

    ``blocked`` is part of the summary on purpose (ruling C2): a terminally
    blocked intent (an embedding-contract mismatch, a generation the cutover
    superseded) never lands and is not pending, so the recall freshness barrier
    has nothing to wait for — a recall for that tenant can answer ``200 []``
    for a write that will never be indexed. Pending-only numbers would read as
    "still in flight" forever.
    """
    by_status: dict[str, int] = dict.fromkeys(OUTBOX_STATUSES, 0)
    status_rows = await db.execute(
        select(IndexOutbox.status, func.count())
        .select_from(IndexOutbox)
        .group_by(IndexOutbox.status)
    )
    for status, count in status_rows.all():
        by_status[str(status)] = int(count or 0)

    by_kind: dict[str, int] = dict.fromkeys(OUTBOX_KINDS, 0)
    kind_rows = await db.execute(
        select(IndexOutbox.kind, func.count())
        .select_from(IndexOutbox)
        .group_by(IndexOutbox.kind)
    )
    for kind, count in kind_rows.all():
        by_kind[str(kind)] = int(count or 0)

    stuck_cutoff = datetime.now(UTC) - timedelta(minutes=STUCK_AFTER_MINUTES)
    stuck = await db.scalar(
        select(func.count())
        .select_from(IndexOutbox)
        .where(IndexOutbox.status == "pending", IndexOutbox.created_at < stuck_cutoff)
    )
    oldest = await db.scalar(
        select(func.min(IndexOutbox.created_at)).where(IndexOutbox.status == "pending")
    )
    return {
        "by_status": by_status,
        "by_kind": by_kind,
        "stuck_pending": int(stuck or 0),
        "oldest_pending_at": _serialize_datetime(oldest),
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
        "index_outbox": await get_index_outbox_summary(db),
    }
