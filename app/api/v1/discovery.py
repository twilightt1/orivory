"""
Discovery API - Multi-hop Discovery Experience

Q2 Growth Track: Graph visualization, guided discovery, cross-document references.

Endpoints:
    GET    /api/v1/discovery/graph              - Get document relationship graph
    POST   /api/v1/discovery/sessions          - Create discovery session
    GET    /api/v1/discovery/sessions          - List discovery sessions
    GET    /api/v1/discovery/sessions/{id}     - Get session with next step
    POST   /api/v1/discovery/sessions/{id}/advance  - Advance to next step
    POST   /api/v1/discovery/sessions/{id}/complete - Complete session
    GET    /api/v1/discovery/references        - Find cross-document references
    GET    /api/v1/discovery/metrics           - Get graph metrics
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.discovery_agent import (
    DiscoveryFlowType,
    DiscoverySession,
    DiscoveryStatus,
    advance_session,
    analyze_document_graph,
    complete_session,
    compute_graph_metrics,
    create_discovery_session,
    find_cross_references,
    get_next_discovery_step,
    synthesize_discovery,
)
from app.database import get_db
from app.middleware.response_cache import CacheInvalidation
from app.models.memory import Memory
from app.models.user import User
from app.retrieval.memory.namespaces import namespace_of, personal_namespace
from app.retrieval.memory.visibility import namespace_predicate
from app.utils.dependencies import enforce_llm_quota, get_current_verified_user

log = logging.getLogger(__name__)

router = APIRouter(prefix="/discovery", tags=["discovery"])


def _owned(memory: Memory | None, user: User) -> bool:
    """May ``user`` read ``memory``? Theirs AND in their namespace.

    The PK-surface spelling (a ``db.get`` cannot carry a predicate, so the
    loaded row is checked): same shape as ``memories._owned`` /
    ``tools._owned`` / ``erasure_service._owned``. A foreign row and a missing
    one answer the same 404 — no existence oracle, and no excerpt of a row
    outside the caller's namespace reaches an agent prompt or a response.
    """
    return (memory is not None
            and memory.user_id == user.id
            and namespace_of(memory) == personal_namespace(user.id))


# ─── Schemas ──────────────────────────────────────────────────────────────────

class DocumentNodeResponse(BaseModel):
    """Document node in graph."""
    id: str
    title: str
    entity_ids: list[str] = []
    salience: float = 0.5
    connection_count: int = 0


class RelationshipEdgeResponse(BaseModel):
    """Relationship edge in graph."""
    source_id: str
    target_id: str
    relationship_type: str
    weight: float = 0.5
    evidence: str = ""


class GraphResponse(BaseModel):
    """Document relationship graph."""
    nodes: list[DocumentNodeResponse]
    edges: list[RelationshipEdgeResponse]


class DiscoverySessionCreate(BaseModel):
    """Request to create discovery session."""
    # UUID type: invalid strings become a 422 from pydantic instead of an
    # unhandled ValueError (500) inside the handler.
    starting_doc_id: UUID = Field(description="Starting document ID")
    flow_type: str = Field(default="explore_related", description="explore_related, trace_origin, find_contradictions, synthesize, temporal_journey")
    target_doc_id: UUID | None = Field(default=None, description="Optional target document ID")


class DiscoverySessionResponse(BaseModel):
    """Discovery session response."""
    id: str
    user_id: str
    flow_type: str
    starting_doc_id: str
    target_doc_id: str | None
    path: list[str]
    current_step: int
    connections_found: list[dict]
    status: str
    created_at: datetime
    completed_at: datetime | None
    steps_taken: int
    documents_explored: int


class DiscoveryStepResponse(BaseModel):
    """Next step in discovery journey."""
    next_doc_id: str
    next_doc_title: str
    reasoning: str
    insight_preview: str
    step_goal: str
    is_complete: bool = False


class CrossReferenceResponse(BaseModel):
    """Cross-document reference."""
    source_doc_id: str
    target_doc_id: str
    source_excerpt: str
    target_excerpt: str
    reference_type: str
    relevance_score: float


class DiscoveryInsightResponse(BaseModel):
    """Synthesized discovery insight."""
    title: str
    description: str
    related_doc_ids: list[str]
    insight_type: str
    confidence: float
    evidence: list[str]


class GraphMetricsResponse(BaseModel):
    """Graph metrics."""
    total_nodes: int
    total_edges: int
    avg_connections_per_node: float
    edge_type_distribution: dict[str, int]
    avg_edge_weight: float
    graph_density: float


class SessionListResponse(BaseModel):
    """List of discovery sessions."""
    items: list[DiscoverySessionResponse]
    total: int
    limit: int
    offset: int


# ─── In-memory session store (production would use Redis/DB) ──────────────────

_session_store: dict[str, DiscoverySession] = {}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _parse_flow_type(flow_type: str) -> DiscoveryFlowType:
    """Parse flow type string to enum."""
    flow_map = {
        "explore_related": DiscoveryFlowType.EXPLORE_RELATED,
        "trace_origin": DiscoveryFlowType.TRACE_ORIGIN,
        "find_contradictions": DiscoveryFlowType.FIND_CONTRADICTIONS,
        "synthesize": DiscoveryFlowType.SYNTHESIZE,
        "temporal_journey": DiscoveryFlowType.TEMPORAL_JOURNEY,
    }
    return flow_map.get(flow_type, DiscoveryFlowType.EXPLORE_RELATED)


def _session_to_response(session: DiscoverySession) -> DiscoverySessionResponse:
    """Convert session to API response."""
    return DiscoverySessionResponse(
        id=session.id,
        user_id=session.user_id,
        flow_type=session.flow_type.value,
        starting_doc_id=session.starting_doc_id,
        target_doc_id=session.target_doc_id,
        path=session.path,
        current_step=session.current_step,
        connections_found=session.connections_found,
        status=session.status.value,
        created_at=session.created_at,
        completed_at=session.completed_at,
        steps_taken=session.steps_taken,
        documents_explored=session.documents_explored,
    )


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/graph", response_model=GraphResponse)
async def get_document_graph(
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    _quota: None = Depends(enforce_llm_quota),
    doc_ids: str | None = Query(default=None, description="Comma-separated doc IDs to include"),
) -> GraphResponse:
    """Get document relationship graph.

    Analyzes user's documents to build a graph of relationships.
    """
    # Fetch the caller's memories — their tenant AND namespace, like every
    # other memory reader (P4a/T5: this router is dormant, not exempt).
    query = select(Memory).where(
        Memory.user_id == current_user.id,
        namespace_predicate(personal_namespace(current_user.id)),
    )
    if doc_ids:
        doc_id_list = [d.strip() for d in doc_ids.split(",")]
        try:
            parsed_ids = [UUID(d) for d in doc_id_list]
        except ValueError as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"doc_ids contains an invalid UUID: {exc}",
            ) from None
        query = query.where(Memory.id.in_(parsed_ids))

    result = await db.execute(query)
    memories = result.scalars().all()

    if not memories:
        return GraphResponse(nodes=[], edges=[])

    # Prepare documents
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

    # Analyze graph
    graph = await analyze_document_graph(documents)

    # Convert to response
    nodes = [
        DocumentNodeResponse(
            id=n.id,
            title=n.title,
            entity_ids=n.entity_ids,
            salience=n.salience,
            connection_count=n.connection_count,
        )
        for n in graph.nodes
    ]

    edges = [
        RelationshipEdgeResponse(
            source_id=e.source_id,
            target_id=e.target_id,
            relationship_type=e.relationship_type,
            weight=e.weight,
            evidence=e.evidence,
        )
        for e in graph.edges
    ]

    return GraphResponse(nodes=nodes, edges=edges)


@router.post("/sessions", response_model=DiscoverySessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    body: DiscoverySessionCreate,
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DiscoverySessionResponse:
    """Create a new discovery session.

    Starts a guided discovery journey from a document.
    """
    # Verify document exists — the caller's OWN row, in their namespace (P4a/T5,
    # F2): a same-account row another namespace owns is not this surface's to
    # serve, exactly like a missing one (the REST memories surface's answer).
    doc = await db.get(Memory, body.starting_doc_id)
    if not _owned(doc, current_user):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")

    if body.target_doc_id:
        target = await db.get(Memory, body.target_doc_id)
        if not _owned(target, current_user):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Target document not found.")

    # Create session
    flow_type = _parse_flow_type(body.flow_type)
    session = create_discovery_session(
        user_id=str(current_user.id),
        starting_doc_id=str(body.starting_doc_id),
        flow_type=flow_type,
        target_doc_id=str(body.target_doc_id) if body.target_doc_id else None,
    )

    # Store session
    _session_store[session.id] = session

    # Invalidate user's sessions cache
    await CacheInvalidation.invalidate_pattern(f"response:/api/v1/discovery/sessions:{current_user.id}:*")

    return _session_to_response(session)


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    current_user: Annotated[User, Depends(get_current_verified_user)],
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status_filter: str | None = Query(default=None, alias="status"),
) -> SessionListResponse:
    """List user's discovery sessions."""
    # Filter sessions by user
    user_sessions = [
        s for s in _session_store.values()
        if s.user_id == str(current_user.id)
        and (status_filter is None or s.status.value == status_filter)
    ]

    # Sort by creation date
    user_sessions.sort(key=lambda s: s.created_at, reverse=True)

    total = len(user_sessions)
    paginated = user_sessions[offset:offset + limit]

    return SessionListResponse(
        items=[_session_to_response(s) for s in paginated],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/sessions/{session_id}", response_model=DiscoverySessionResponse)
async def get_session(
    session_id: str,
    current_user: Annotated[User, Depends(get_current_verified_user)],
) -> DiscoverySessionResponse:
    """Get a discovery session."""
    session = _session_store.get(session_id)

    if not session or session.user_id != str(current_user.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found.")

    return _session_to_response(session)


@router.get("/sessions/{session_id}/step", response_model=DiscoveryStepResponse)
async def get_next_step(
    session_id: str,
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    _quota: None = Depends(enforce_llm_quota),
) -> DiscoveryStepResponse:
    """Get the next step in a discovery journey."""
    session = _session_store.get(session_id)

    if not session or session.user_id != str(current_user.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found.")

    if session.status == DiscoveryStatus.COMPLETED:
        return DiscoveryStepResponse(
            next_doc_id="",
            next_doc_title="Journey complete",
            reasoning="You have completed this discovery journey.",
            insight_preview="",
            step_goal="",
            is_complete=True,
        )

    # Fetch user's documents
    result = await db.execute(
        select(Memory).where(
            Memory.user_id == current_user.id,
            namespace_predicate(personal_namespace(current_user.id)),
        )
    )
    memories = result.scalars().all()

    documents = [
        {
            "id": str(m.id),
            "title": m.title,
            "content": f"{m.title}\n{m.summary}\n{m.content}" if m.summary else m.content,
        }
        for m in memories
    ]

    # Build graph
    graph = await analyze_document_graph(documents)

    # Get next step
    step = await get_next_discovery_step(session, documents, graph)

    return DiscoveryStepResponse(
        next_doc_id=step["next_doc_id"],
        next_doc_title=step["next_doc_title"],
        reasoning=step["reasoning"],
        insight_preview=step["insight_preview"],
        step_goal=step["step_goal"],
        is_complete=not step["next_doc_id"],
    )


@router.post("/sessions/{session_id}/advance", response_model=DiscoverySessionResponse)
async def advance_discovery(
    session_id: str,
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    _quota: None = Depends(enforce_llm_quota),
) -> DiscoverySessionResponse:
    """Advance to the next step in discovery journey."""
    session = _session_store.get(session_id)

    if not session or session.user_id != str(current_user.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found.")

    if session.status == DiscoveryStatus.COMPLETED:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Session already completed.")

    # Get next step
    result = await db.execute(
        select(Memory).where(
            Memory.user_id == current_user.id,
            namespace_predicate(personal_namespace(current_user.id)),
        )
    )
    memories = result.scalars().all()

    documents = [
        {"id": str(m.id), "title": m.title, "content": m.content}
        for m in memories
    ]

    graph = await analyze_document_graph(documents)
    step = await get_next_discovery_step(session, documents, graph)

    if not step["next_doc_id"]:
        # No more steps, complete session
        session = complete_session(session)
        _session_store[session_id] = session
        return _session_to_response(session)

    # Advance session
    connection = {
        "from_doc": session.path[-1] if session.path else None,
        "to_doc": step["next_doc_id"],
        "insight": step["insight_preview"],
        "step_goal": step["step_goal"],
    }

    session = advance_session(session, step["next_doc_id"], connection)
    _session_store[session_id] = session

    # Invalidate user's sessions cache
    await CacheInvalidation.invalidate_pattern(f"response:/api/v1/discovery/sessions:{current_user.id}:*")

    return _session_to_response(session)


@router.post("/sessions/{session_id}/complete", response_model=DiscoverySessionResponse)
async def complete_discovery(
    session_id: str,
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    _quota: None = Depends(enforce_llm_quota),
) -> DiscoverySessionResponse:
    """Complete a discovery session and get synthesized insight."""
    session = _session_store.get(session_id)

    if not session or session.user_id != str(current_user.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found.")

    # Synthesize findings
    result = await db.execute(
        select(Memory).where(
            Memory.user_id == current_user.id,
            namespace_predicate(personal_namespace(current_user.id)),
        )
    )
    memories = result.scalars().all()

    documents = [
        {"id": str(m.id), "title": m.title, "content": m.content}
        for m in memories
    ]

    synthesis = await synthesize_discovery(session, documents)

    # Add synthesis as final connection
    session.connections_found.append({
        "type": "synthesis",
        "title": synthesis.title,
        "description": synthesis.description,
        "confidence": synthesis.confidence,
    })

    session = complete_session(session)
    _session_store[session_id] = session

    # Invalidate user's sessions cache
    await CacheInvalidation.invalidate_pattern(f"response:/api/v1/discovery/sessions:{current_user.id}:*")

    return _session_to_response(session)


@router.get("/references", response_model=list[CrossReferenceResponse])
async def find_references(
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    doc1_id: UUID = Query(description="First document ID"),
    doc2_id: UUID = Query(description="Second document ID"),
) -> list[CrossReferenceResponse]:
    """Find cross-document references between two documents."""
    # Both rows must be the caller's AND in their namespace (P4a/T5, F2): the
    # titles/bodies below feed an analysis and come back as excerpts, so a row
    # outside the namespace must be refused before it is ever read into one.
    doc1 = await db.get(Memory, doc1_id)
    doc2 = await db.get(Memory, doc2_id)

    if not _owned(doc1, current_user):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "First document not found.")
    if not _owned(doc2, current_user):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Second document not found.")

    doc1_dict = {
        "id": str(doc1.id),
        "title": doc1.title,
        "content": f"{doc1.title}\n{doc1.summary}\n{doc1.content}" if doc1.summary else doc1.content,
    }
    doc2_dict = {
        "id": str(doc2.id),
        "title": doc2.title,
        "content": f"{doc2.title}\n{doc2.summary}\n{doc2.content}" if doc2.summary else doc2.content,
    }

    references = await find_cross_references(doc1_dict, doc2_dict)

    return [
        CrossReferenceResponse(
            source_doc_id=r.source_doc_id,
            target_doc_id=r.target_doc_id,
            source_excerpt=r.source_excerpt,
            target_excerpt=r.target_excerpt,
            reference_type=r.reference_type,
            relevance_score=r.relevance_score,
        )
        for r in references
    ]


@router.get("/metrics", response_model=GraphMetricsResponse)
async def get_graph_metrics(
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    _quota: None = Depends(enforce_llm_quota),
) -> GraphMetricsResponse:
    """Get metrics for user's document graph."""
    result = await db.execute(
        select(Memory).where(
            Memory.user_id == current_user.id,
            namespace_predicate(personal_namespace(current_user.id)),
        )
    )
    memories = result.scalars().all()

    documents = [
        {
            "id": str(m.id),
            "title": m.title,
            "content": f"{m.title}\n{m.summary}\n{m.content}" if m.summary else m.content,
        }
        for m in memories
    ]

    graph = await analyze_document_graph(documents)
    metrics = compute_graph_metrics(graph)

    return GraphMetricsResponse(**metrics)


@router.post("/sessions/{session_id}/synthesis", response_model=DiscoveryInsightResponse)
async def get_synthesis(
    session_id: str,
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DiscoveryInsightResponse:
    """Get synthesized insight from a discovery session."""
    session = _session_store.get(session_id)

    if not session or session.user_id != str(current_user.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found.")

    result = await db.execute(
        select(Memory).where(
            Memory.user_id == current_user.id,
            namespace_predicate(personal_namespace(current_user.id)),
        )
    )
    memories = result.scalars().all()

    documents = [
        {"id": str(m.id), "title": m.title, "content": m.content}
        for m in memories
    ]

    synthesis = await synthesize_discovery(session, documents)

    return DiscoveryInsightResponse(
        title=synthesis.title,
        description=synthesis.description,
        related_doc_ids=synthesis.related_doc_ids,
        insight_type=synthesis.insight_type,
        confidence=synthesis.confidence,
        evidence=synthesis.evidence,
    )
