"""
Insight Cards API — Proactive Discovery Feature

Endpoints:
    GET    /api/v1/insights              list insight cards (filter by status, type)
    POST   /api/v1/insights/generate     generate new insights for user
    GET    /api/v1/insights/{id}         get single insight card
    POST   /api/v1/insights/{id}/dismiss  dismiss an insight
    POST   /api/v1/insights/{id}/save    save an insight
    POST   /api/v1/insights/{id}/feedback  provide feedback on insight
    POST   /api/v1/insights/refresh      refresh insights based on new activity
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.insight_agent import (
    InsightGenerationRequest,
    generate_insight_batch,
    refresh_insights,
    update_user_preferences,
)
from app.database import get_db
from app.middleware.response_cache import CacheInvalidation
from app.models.insight import InsightCard, InsightStatusEnum
from app.models.user import User
from app.retrieval.memory.namespaces import personal_namespace
from app.retrieval.memory.visibility import namespace_predicate
from app.utils.dependencies import enforce_llm_quota, get_current_verified_user

log = logging.getLogger(__name__)

router = APIRouter(prefix="/insights", tags=["insights"])


# ─── Schemas ──────────────────────────────────────────────────────────────────

class InsightSourceDoc(BaseModel):
    """Source document reference."""
    document_id: str
    title: str = ""
    excerpt: str = ""
    relevance_score: float = 0.5


class InsightResponse(BaseModel):
    """Insight card API response."""
    model_config = {"from_attributes": True}

    id: UUID
    user_id: UUID
    title: str
    insight_type: str
    summary: str
    detail: str
    source_docs: list[InsightSourceDoc]
    source_count: int
    surprise_level: str
    confidence: float
    created_at: datetime
    shown_at: datetime | None
    dismissed_at: datetime | None
    status: str
    helpful: bool | None
    feedback_note: str | None
    shown_count: int
    relevance_score: float
    type_emoji: str


class InsightListResponse(BaseModel):
    """List response for insights."""
    items: list[InsightResponse]
    total: int
    limit: int
    offset: int


class InsightGenerateRequest(BaseModel):
    """Request to generate new insights."""
    # UUID items: invalid strings become 422 from pydantic, not a 500 later.
    document_ids: list[UUID] = Field(default_factory=list, description="Specific documents to analyze")
    focus_topics: list[str] = Field(default_factory=list, description="Topics user is interested in")
    max_insights: int = Field(default=5, ge=1, le=20, description="Maximum insights to generate")


class InsightGenerateResponse(BaseModel):
    """Response from insight generation."""
    insights: list[InsightResponse]
    generation_time_ms: float
    documents_analyzed: int
    error: str | None = None


class InsightFeedbackRequest(BaseModel):
    """Feedback on an insight."""
    helpful: bool
    note: str | None = None


class InsightRefreshResponse(BaseModel):
    """Response from insight refresh."""
    updated_count: int
    new_insights: list[InsightResponse]
    expired_count: int


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _to_aware_utc(dt: datetime | None) -> datetime | None:
    """Normalize a datetime to timezone-aware UTC (legacy rows may be naive)."""
    if dt is None:
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


def _insight_response(card: InsightCard) -> InsightResponse:
    """Map ORM InsightCard to API response."""
    # Parse source_docs from JSONB
    source_docs = []
    for src in card.source_docs or []:
        source_docs.append(InsightSourceDoc(
            document_id=src.get("document_id", ""),
            title=src.get("title", ""),
            excerpt=src.get("excerpt", ""),
            relevance_score=src.get("relevance_score", 0.5),
        ))

    # Get emoji for type
    emoji_map = {
        "connection": "🔗",
        "contradiction": "⚡",
        "evolution": "📈",
        "pattern": "🔄",
        "gap": "❓",
        "confirmation": "✅",
        "synthesis": "💡",
    }

    return InsightResponse(
        id=card.id,
        user_id=card.user_id,
        title=card.title,
        insight_type=card.insight_type,
        summary=card.summary,
        detail=card.detail,
        source_docs=source_docs,
        source_count=card.source_count,
        surprise_level=card.surprise_level,
        confidence=card.confidence,
        created_at=card.created_at,
        shown_at=card.shown_at,
        dismissed_at=card.dismissed_at,
        status=card.status,
        helpful=card.helpful,
        feedback_note=card.feedback_note,
        shown_count=card.shown_count,
        relevance_score=card.relevance_score,
        type_emoji=emoji_map.get(card.insight_type, "💡"),
    )


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.get("", response_model=InsightListResponse)
async def list_insights(
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    status_filter: Literal["new", "shown", "dismissed", "saved", "expired"] | None = Query(default=None, alias="status"),
    insight_type: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> InsightListResponse:
    """List insight cards for the current user with optional filters."""
    base = select(InsightCard).where(InsightCard.user_id == current_user.id)
    count_base = select(func.count(InsightCard.id)).where(InsightCard.user_id == current_user.id)

    if status_filter:
        base = base.where(InsightCard.status == status_filter)
        count_base = count_base.where(InsightCard.status == status_filter)

    if insight_type:
        base = base.where(InsightCard.insight_type == insight_type)
        count_base = count_base.where(InsightCard.insight_type == insight_type)

    # Order by relevance score, then by created date
    base = base.order_by(desc(InsightCard.relevance_score), desc(InsightCard.created_at))

    total = (await db.execute(count_base)).scalar_one()
    rows = (await db.execute(
        base.offset(offset).limit(limit)
    )).scalars().all()

    return InsightListResponse(
        items=[_insight_response(card) for card in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("/generate", response_model=InsightGenerateResponse)
async def generate_insights(
    body: InsightGenerateRequest,
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    _quota: None = Depends(enforce_llm_quota),
) -> InsightGenerateResponse:
    """Generate new insights for the user.

    Analyzes user's documents to discover hidden connections and patterns.
    """
    from app.models.memory import Memory

    # Fetch the caller's memories (as documents) — their namespace only.
    query = select(Memory).where(
        Memory.user_id == current_user.id,
        namespace_predicate(personal_namespace(current_user.id)),
    )
    if body.document_ids:
        # Filter by specific document IDs if provided
        query = query.where(Memory.id.in_(body.document_ids))

    result = await db.execute(query)
    memories = result.scalars().all()

    if not memories:
        return InsightGenerateResponse(
            insights=[],
            generation_time_ms=0,
            documents_analyzed=0,
            error="No documents found. Add some memories first.",
        )

    # Prepare documents for insight generation
    documents = [
        {
            "id": str(m.id),
            "title": m.title,
            "content": f"{m.title}\n{m.summary}\n{m.content}" if m.summary else m.content,
            "created_at": m.captured_at.isoformat() if m.captured_at else None,
            "tags": m.tags or [],
        }
        for m in memories
    ]

    # Get recent activity (recent memories by capture date)
    recent_memories = sorted(memories, key=lambda m: m.captured_at or datetime.min, reverse=True)[:20]
    recent_activity = [
        f"{m.title}: {m.summary[:100]}" if m.summary else m.title
        for m in recent_memories
    ]

    # Generate insights
    request = InsightGenerationRequest(
        user_id=str(current_user.id),
        document_ids=[str(m.id) for m in memories],
        focus_topics=body.focus_topics,
    )

    generation_result = await generate_insight_batch(
        request=request,
        documents=documents,
        recent_activity=recent_activity,
    )

    if generation_result.error:
        return InsightGenerateResponse(
            insights=[],
            generation_time_ms=generation_result.generation_time_ms,
            documents_analyzed=generation_result.documents_analyzed,
            error=generation_result.error,
        )

    # Save generated insights to database
    saved_cards = []
    for _i, insight in enumerate(generation_result.insights[:body.max_insights]):
        card = InsightCard(
            user_id=current_user.id,
            title=insight.title,
            insight_type=insight.insight_type.value,
            summary=insight.summary,
            detail=insight.detail,
            source_docs=[
                {"document_id": s.document_id, "title": s.title, "excerpt": s.excerpt, "relevance_score": s.relevance_score}
                for s in insight.sources
            ],
            source_count=insight.source_count,
            surprise_level=insight.surprise_level.value,
            confidence=insight.confidence,
            relevance_score=insight.relevance_score,
        )
        db.add(card)
        saved_cards.append(card)

    await db.commit()

    # Refresh cards to get IDs
    for card in saved_cards:
        await db.refresh(card)

    return InsightGenerateResponse(
        insights=[_insight_response(card) for card in saved_cards],
        generation_time_ms=generation_result.generation_time_ms,
        documents_analyzed=generation_result.documents_analyzed,
    )


@router.get("/{insight_id}", response_model=InsightResponse)
async def get_insight(
    insight_id: UUID,
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InsightResponse:
    """Get a single insight card."""
    card = await db.get(InsightCard, insight_id)
    if not card or card.user_id != current_user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Insight not found.")

    # Mark as shown
    if card.status == InsightStatusEnum.NEW.value:
        card.status = InsightStatusEnum.SHOWN.value
        card.shown_at = datetime.now(UTC)
    card.shown_count += 1
    await db.commit()

    return _insight_response(card)


@router.post("/{insight_id}/dismiss", response_model=InsightResponse)
async def dismiss_insight(
    insight_id: UUID,
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InsightResponse:
    """Dismiss an insight card."""
    card = await db.get(InsightCard, insight_id)
    if not card or card.user_id != current_user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Insight not found.")

    card.status = InsightStatusEnum.DISMISSED.value
    card.dismissed_at = datetime.now(UTC)
    await db.commit()
    await db.refresh(card)

    # Invalidate user's insights cache
    await CacheInvalidation.invalidate_pattern(f"response:/api/v1/insights:{current_user.id}:*")

    return _insight_response(card)


@router.post("/{insight_id}/save", response_model=InsightResponse)
async def save_insight(
    insight_id: UUID,
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InsightResponse:
    """Save an insight card for later reference."""
    card = await db.get(InsightCard, insight_id)
    if not card or card.user_id != current_user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Insight not found.")

    card.status = InsightStatusEnum.SAVED.value
    card.helpful = True
    await db.commit()
    await db.refresh(card)

    # Invalidate user's insights cache
    await CacheInvalidation.invalidate_pattern(f"response:/api/v1/insights:{current_user.id}:*")

    return _insight_response(card)


@router.post("/{insight_id}/feedback", response_model=InsightResponse)
async def feedback_insight(
    insight_id: UUID,
    body: InsightFeedbackRequest,
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> InsightResponse:
    """Provide feedback on an insight card."""
    card = await db.get(InsightCard, insight_id)
    if not card or card.user_id != current_user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Insight not found.")

    card.helpful = body.helpful
    card.feedback_note = body.note
    await db.commit()
    await db.refresh(card)

    # Update user preferences based on feedback
    feedback_list = [{
        "insight_id": str(card.id),
        "insight_type": card.insight_type,
        "helpful": body.helpful,
    }]
    await update_user_preferences(str(current_user.id), feedback_list)

    return _insight_response(card)


@router.post("/refresh", response_model=InsightRefreshResponse)
async def refresh_insights_endpoint(
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    _quota: None = Depends(enforce_llm_quota),
) -> InsightRefreshResponse:
    """Refresh insights based on new activity.

    Re-evaluates existing insights and generates new ones if warranted.
    """
    from app.models.memory import Memory

    # Get existing non-expired insights
    existing_query = select(InsightCard).where(
        InsightCard.user_id == current_user.id,
        InsightCard.status != InsightStatusEnum.EXPIRED.value,
    )
    result = await db.execute(existing_query)
    existing_cards = result.scalars().all()

    # Get recent memories for activity — the caller's namespace only.
    recent_query = select(Memory).where(
        Memory.user_id == current_user.id,
        namespace_predicate(personal_namespace(current_user.id)),
    ).order_by(desc(Memory.captured_at)).limit(10)
    recent_result = await db.execute(recent_query)
    recent_memories = recent_result.scalars().all()

    # Check for new memories since last refresh.
    # Normalize both sides to aware UTC datetimes — legacy insight cards may
    # carry naive timestamps, and comparing naive vs aware raises TypeError.
    last_insight_time = max(
        (
            aware
            for card in existing_cards
            if (aware := _to_aware_utc(card.created_at)) is not None
        ),
        default=datetime.min.replace(tzinfo=UTC),
    )
    new_memories = [
        m
        for m in recent_memories
        if _to_aware_utc(m.captured_at) is not None and _to_aware_utc(m.captured_at) > last_insight_time
    ]

    # Refresh existing insights
    updates = await refresh_insights(
        existing_insights=[{"id": str(c.id), "title": c.title, "summary": c.summary} for c in existing_cards],
        recent_activity=[m.title for m in recent_memories],
        new_documents=[{"id": str(m.id), "title": m.title} for m in new_memories],
    )

    updated_count = 0
    expired_count = 0

    # Ownership guard: update IDs come from LLM output, which may echo a
    # foreign or hallucinated ID. Only honor IDs from the caller's own
    # already-filtered card set — never trust the model for authorization.
    owned_ids = {c.id for c in existing_cards}
    for update in updates:
        insight_id = UUID(update["insight_id"])
        if insight_id not in owned_ids:
            continue
        card = await db.get(InsightCard, insight_id)
        if card and card.user_id == current_user.id:
            action = update["action"]
            if action == "expire":
                card.status = InsightStatusEnum.EXPIRED.value
                expired_count += 1
            elif action == "dismiss":
                card.status = InsightStatusEnum.DISMISSED.value
                updated_count += 1
            elif action == "boost":
                card.relevance_score = update.get("new_insight_score", card.relevance_score)
                updated_count += 1

    await db.commit()

    # Invalidate user's insights cache
    await CacheInvalidation.invalidate_pattern(f"response:/api/v1/insights:{current_user.id}:*")

    # Generate new insights if there are new documents
    new_insights = []
    if new_memories:
        documents = [
            {
                "id": str(m.id),
                "title": m.title,
                "content": f"{m.title}\n{m.summary}\n{m.content}" if m.summary else m.content,
                "created_at": m.captured_at.isoformat() if m.captured_at else None,
            }
            for m in recent_memories
        ]

        request = InsightGenerationRequest(
            user_id=str(current_user.id),
            document_ids=[str(m.id) for m in recent_memories],
        )

        result = await generate_insight_batch(
            request=request,
            documents=documents,
            recent_activity=[m.title for m in recent_memories],
        )

        for insight in result.insights[:3]:  # Limit new insights
            card = InsightCard(
                user_id=current_user.id,
                title=insight.title,
                insight_type=insight.insight_type.value,
                summary=insight.summary,
                detail=insight.detail,
                source_docs=[
                    {"document_id": s.document_id, "title": s.title, "excerpt": s.excerpt, "relevance_score": s.relevance_score}
                    for s in insight.sources
                ],
                source_count=insight.source_count,
                surprise_level=insight.surprise_level.value,
                confidence=insight.confidence,
                relevance_score=insight.relevance_score,
            )
            db.add(card)
            new_insights.append(card)

        await db.commit()
        for card in new_insights:
            await db.refresh(card)

    return InsightRefreshResponse(
        updated_count=updated_count,
        new_insights=[_insight_response(card) for card in new_insights],
        expired_count=expired_count,
    )
