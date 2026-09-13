"""Admin endpoints."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.conversation import Conversation
from app.models.document import Document
from app.models.message import Message
from app.models.user import User
from app.models.user_quota import UserQuota
from app.schemas.auth import UserResponse
from app.services.audit_service import AuditService
from app.services.diagnostics_service import build_diagnostics
from app.utils.dependencies import require_admin

router = APIRouter(prefix="/admin", tags=["admin"])


class UserUpdate(BaseModel):
    # Literal: a typo like "admn" would silently strip the target's role.
    role:          Literal["user", "admin"] | None = None
    is_active:     bool | None = None
    is_deleted:    bool | None = None
    daily_limit:   int | None = Field(default=None, ge=0)
    monthly_limit: int | None = Field(default=None, ge=0)


class StatsResponse(BaseModel):
    total_users:          int
    active_users:         int
    total_documents:      int
    total_messages:       int
    pending_documents:    int

class UserActivitySummary(BaseModel):
    user_id:             UUID
    total_conversations: int
    total_messages:      int
    total_documents:     int
    last_message_at:     datetime | None
    last_document_at:    datetime | None


@router.get("/users", response_model=list[UserResponse])
async def list_users(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    include_deleted: bool = False,
    _=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    query = select(User).order_by(User.created_at.desc()).offset(skip).limit(limit)
    if not include_deleted:
        query = query.where(User.is_deleted.is_(False))

    result = await db.execute(query)
    return result.scalars().all()


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: UUID,
    admin_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(404, detail="User not found.")
    return UserResponse.model_validate(user)


@router.get("/users/{user_id}/activity", response_model=UserActivitySummary)
async def get_user_activity(
    user_id: UUID,
    admin_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(404, detail="User not found.")

    total_convs = await db.scalar(select(func.count(Conversation.id)).where(Conversation.user_id == user_id))
    total_msgs = await db.scalar(
        select(func.count(Message.id))
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(Conversation.user_id == user_id)
    )
    total_docs = await db.scalar(
        select(func.count(Document.id))
        .join(Conversation, Document.conversation_id == Conversation.id)
        .where(Conversation.user_id == user_id)
    )

    last_msg = await db.scalar(
        select(func.max(Message.created_at))
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(Conversation.user_id == user_id)
    )
    last_doc = await db.scalar(
        select(func.max(Document.created_at))
        .join(Conversation, Document.conversation_id == Conversation.id)
        .where(Conversation.user_id == user_id)
    )

    return UserActivitySummary(
        user_id=user_id,
        total_conversations=total_convs or 0,
        total_messages=total_msgs or 0,
        total_documents=total_docs or 0,
        last_message_at=last_msg,
        last_document_at=last_doc
    )


@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: UUID,
    body: UserUpdate,
    admin_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(404, detail="User not found.")

    # An admin demoting/deleting their own account can lock every admin out.
    if user_id == admin_user.id and (
        (body.role is not None and body.role != "admin")
        or body.is_active is False
        or body.is_deleted is True
    ):
        raise HTTPException(400, detail="Admins cannot demote, deactivate, or delete their own account.")

    changes = {}

    if body.role is not None and user.role != body.role:
        changes["role"] = {"old": user.role, "new": body.role}
        user.role = body.role

    if body.is_active is not None and user.is_active != body.is_active:
        changes["is_active"] = {"old": user.is_active, "new": body.is_active}
        user.is_active = body.is_active

    if body.is_deleted is not None and user.is_deleted != body.is_deleted:
        changes["is_deleted"] = {"old": user.is_deleted, "new": body.is_deleted}
        user.is_deleted = body.is_deleted

    if body.daily_limit is not None or body.monthly_limit is not None:
        quota = await db.scalar(select(UserQuota).where(UserQuota.user_id == user_id))
        if quota:
            if body.daily_limit is not None and quota.daily_limit != body.daily_limit:
                changes["daily_limit"] = {"old": quota.daily_limit, "new": body.daily_limit}
                quota.daily_limit = body.daily_limit
            if body.monthly_limit is not None and quota.monthly_limit != body.monthly_limit:
                changes["monthly_limit"] = {"old": quota.monthly_limit, "new": body.monthly_limit}
                quota.monthly_limit = body.monthly_limit

    if changes:
        await AuditService.log_action(
            db=db,
            admin_id=admin_user.id,
            action="update_user",
            target_entity_type="user",
            target_entity_id=user_id,
            changes=changes
        )

    await db.commit()
    await db.refresh(user)
    return UserResponse.model_validate(user)


@router.post("/users/{user_id}/reset-quota", status_code=200)
async def reset_quota(
    user_id: UUID,
    admin_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    quota = await db.scalar(select(UserQuota).where(UserQuota.user_id == user_id))
    if not quota:
        raise HTTPException(404, detail="Quota record not found.")

    changes = {
        "requests_today": {"old": quota.requests_today, "new": 0},
        "requests_month": {"old": quota.requests_month, "new": 0},
        "tokens_today": {"old": quota.tokens_today, "new": 0},
        "tokens_month": {"old": quota.tokens_month, "new": 0},
    }

    quota.requests_today = 0
    quota.requests_month = 0
    quota.tokens_today   = 0
    quota.tokens_month   = 0

    await AuditService.log_action(
        db=db,
        admin_id=admin_user.id,
        action="reset_quota",
        target_entity_type="user",
        target_entity_id=user_id,
        changes=changes
    )

    await db.commit()
    return {"message": "Quota reset successfully."}


class DocumentSummary(BaseModel):
    id:              UUID
    user_id:         UUID
    filename:        str
    file_size:       int | None
    mime_type:       str | None
    status:          str
    chunk_count:     int
    error_msg:       str | None
    created_at:      datetime
    updated_at:      datetime


@router.get("/documents", response_model=list[DocumentSummary])
async def list_documents(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    status: str | None = None,
    user_id: UUID | None = None,
    _=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    query = select(
        Document.id,
        Document.filename,
        Document.file_size,
        Document.mime_type,
        Document.status,
        Document.chunk_count,
        Document.error_msg,
        Document.created_at,
        Document.updated_at,
        Conversation.user_id
    ).join(Conversation).order_by(Document.created_at.desc()).offset(skip).limit(limit)

    if status:
        query = query.where(Document.status == status)
    if user_id:
        query = query.where(Conversation.user_id == user_id)

    result = await db.execute(query)

    docs = []
    for row in result.all():
        docs.append(DocumentSummary(
            id=row.id,
            user_id=row.user_id,
            filename=row.filename,
            file_size=row.file_size,
            mime_type=row.mime_type,
            status=row.status,
            chunk_count=row.chunk_count,
            error_msg=row.error_msg,
            created_at=row.created_at,
            updated_at=row.updated_at
        ))
    return docs


@router.post("/documents/{document_id}/retry", status_code=200)
async def retry_document(
    document_id: UUID,
    admin_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(404, detail="Document not found.")

    if doc.status not in ["failed", "error"]:
        raise HTTPException(400, detail="Only failed documents can be retried.")

    changes = {
        "status": {"old": doc.status, "new": "pending"},
        "error_msg": {"old": doc.error_msg, "new": None}
    }

    doc.status = "pending"
    doc.error_msg = None

    await AuditService.log_action(
        db=db,
        admin_id=admin_user.id,
        action="retry_document",
        target_entity_type="document",
        target_entity_id=document_id,
        changes=changes
    )

    await db.commit()

    from app.ingestion.pipeline import process_document_sync
    process_document_sync(str(doc.id))

    return {"message": "Document queued for retry."}


@router.delete("/documents/{document_id}", status_code=200)
async def delete_document(
    document_id: UUID,
    admin_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(404, detail="Document not found.")

    conversation = await db.get(Conversation, doc.conversation_id)
    if not conversation:
        raise HTTPException(404, detail="Conversation not found for document.")

    filename = doc.filename

    await AuditService.log_action(
        db=db,
        admin_id=admin_user.id,
        action="delete_document",
        target_entity_type="document",
        target_entity_id=document_id,
        changes={"filename": filename, "conversation_id": str(doc.conversation_id)}
    )

    from app.services.document_service import delete_document as delete_document_service
    await delete_document_service(db, doc, conversation)
    return {"message": "Document deleted successfully."}


@router.get("/diagnostics", response_model=dict[str, Any])
async def get_diagnostics(
    _=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    return await build_diagnostics(db)


@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    _=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    total_users     = await db.scalar(select(func.count(User.id)))
    active_users    = await db.scalar(select(func.count(User.id)).where(User.is_active.is_(True)))
    total_docs      = await db.scalar(select(func.count(Document.id)))
    pending_docs    = await db.scalar(select(func.count(Document.id)).where(
                            Document.status.in_(["pending", "processing"])
                        ))
    total_messages  = await db.scalar(select(func.count(Message.id)))

    return StatsResponse(
        total_users=total_users or 0,
        active_users=active_users or 0,
        total_documents=total_docs or 0,
        total_messages=total_messages or 0,
        pending_documents=pending_docs or 0,
    )


class AgentCostSummary(BaseModel):
    agent: str
    calls: int
    tokens_in: int
    tokens_out: int
    total_cost_usd: float


class AICostsResponse(BaseModel):
    window_hours: int
    total_cost_usd: float
    by_agent: list[AgentCostSummary]
    recent_calls: list[dict[str, Any]]
    note: str | None = None


@router.get("/ai-costs", response_model=AICostsResponse)
async def get_ai_costs(
    hours: int = 24,
    _=Depends(require_admin),
):
    """
    AI/ML cost summary over the last `hours` window (default 24).
    Reads from the SQLite-backed CostTracker populated by agent calls.
    """
    try:
        import asyncio

        from app.observability.cost import CostTracker, budget_window_iso

        tracker = CostTracker()
        since = budget_window_iso(hours=hours)

        def _read() -> tuple[float, dict, list]:
            return (
                tracker.total(since_iso=since),
                tracker.breakdown_by_agent(since_iso=since),
                tracker.recent(limit=20),
            )

        # SQLite reads are blocking; keep them off the event loop thread.
        total, breakdown, recent = await asyncio.to_thread(_read)
        by_agent = [
            AgentCostSummary(
                agent=agent,
                calls=stats["calls"],
                tokens_in=stats["tokens_in"],
                tokens_out=stats["tokens_out"],
                total_cost_usd=round(stats["total_cost_usd"], 6),
            )
            for agent, stats in breakdown.items()
        ]
        return AICostsResponse(
            window_hours=hours,
            total_cost_usd=round(total, 6),
            by_agent=by_agent,
            recent_calls=recent,
            note=None,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return AICostsResponse(
            window_hours=hours,
            total_cost_usd=0.0,
            by_agent=[],
            recent_calls=[],
            note=f"Cost tracker unavailable: {exc}",
        )


class ReindexRequest(BaseModel):
    user_id: UUID
    only_missing: bool = True


class ReindexResponse(BaseModel):
    queued: bool
    task_id: str | None = None
    user_id: UUID
    only_missing: bool
    note: str | None = None


@router.post("/memories/reindex", response_model=ReindexResponse)
async def reindex_memories(
    body: ReindexRequest,
    admin_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> ReindexResponse:
    """Queue a backfill that re-embeds a user's memories into ChromaDB.

    Replays the Postgres ``memories`` rows into the vector index. Use after a
    Chroma data loss, or to index memories captured before write-through
    embedding existed. ``only_missing`` (default) skips memories already
    present in the collection; set false to rebuild every vector.
    """
    target = await db.get(User, body.user_id)
    if not target:
        raise HTTPException(404, detail="User not found.")

    await AuditService.log_action(
        db=db,
        admin_id=admin_user.id,
        action="reindex_memories",
        target_entity_type="user",
        target_entity_id=body.user_id,
        changes={"only_missing": body.only_missing},
    )
    await db.commit()

    try:
        from app.retrieval.memory.reindex import reindex_user_memories_sync

        summary = reindex_user_memories_sync(str(body.user_id), only_missing=body.only_missing)
        return ReindexResponse(
            queued=True,
            task_id=None,
            user_id=body.user_id,
            only_missing=body.only_missing,
            note="scanned={scanned} reindexed={reindexed} already_indexed={already_indexed} pages={pages}".format(**summary),
        )
    except Exception as exc:  # pragma: no cover - defensive
        return ReindexResponse(
            queued=False,
            task_id=None,
            user_id=body.user_id,
            only_missing=body.only_missing,
            note=f"Reindex failed: {exc}",
        )


class QualityTrendResponse(BaseModel):
    window_hours: int
    generated_at: str | None = None
    sample_size: int
    citation_rate: float
    citation_sample: int
    grounded_rate: float
    grounded_sample: int
    hallucination_flag_rate: float
    self_correction_rate: float
    avg_grounding_confidence: float
    grounding_confidence_sample: int
    avg_answer_latency_ms: float
    latency_sample: int
    note: str | None = None


@router.get("/quality/trend", response_model=QualityTrendResponse)
async def get_quality_trend(
    hours: int = 24,
    _=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> QualityTrendResponse:
    """Quality signals aggregated over the last `hours` window.

    Reduces the per-message `agent_trace` blobs (citation presence, grounded
    verdict, self-correction retries, grounding confidence, latency) into
    rates a reviewer can watch over time. Each rate reports its own denominator
    so a flood of chitchat doesn't silently dilute the numbers.
    """
    try:
        from app.services.quality_service import build_quality_trend

        metrics = await build_quality_trend(db, hours=hours)
        return QualityTrendResponse(**metrics)
    except Exception as exc:  # pragma: no cover - defensive
        return QualityTrendResponse(
            window_hours=hours,
            sample_size=0,
            citation_rate=0.0,
            citation_sample=0,
            grounded_rate=0.0,
            grounded_sample=0,
            hallucination_flag_rate=0.0,
            self_correction_rate=0.0,
            avg_grounding_confidence=0.0,
            grounding_confidence_sample=0,
            avg_answer_latency_ms=0.0,
            latency_sample=0,
            note=f"Quality trend unavailable: {exc}",
        )
