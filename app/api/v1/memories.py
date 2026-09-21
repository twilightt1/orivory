"""
Memory API — second-brain personal memory storage.

Endpoints:
    POST   /api/v1/memories           create a memory (manual note, or from any source)
    GET    /api/v1/memories           list memories (filter by source_type, tag, query)
    GET    /api/v1/memories/{id}      fetch one memory with entity links
    PATCH  /api/v1/memories/{id}      update fields (title, summary, tags, salience, pinned)
    DELETE /api/v1/memories/{id}      remove a memory (cascades to entity + source links)

Note: This endpoint is for direct user/agent memory writes. The bulk
ingestion path (file upload, sync from a Source) lives in the
ingestion service and is wired up in Phase 2.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import String, cast, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.memory import Memory
from app.models.user import User
from app.retrieval.memory.correction import client_metadata, state_of
from app.retrieval.memory.namespaces import PERSONAL, namespace_of, personal_namespace
from app.retrieval.memory.outbox import bump_revision, enqueue_upsert, mark_done
from app.retrieval.memory.retriever import MemoryRetriever
from app.retrieval.memory.visibility import (
    namespace_predicate,
    not_dirty_predicate,
    state_expression,
)
from app.retrieval.memory.write_back import index_new_memory, safe_upsert_to_index
from app.schemas.Orivory import (
    DigestResponse,
    MemoryCreate,
    MemoryListResponse,
    MemoryResponse,
    MemoryUpdate,
    RecallRequest,
    RecallResponse,
)
from app.services.erasure_service import erase_memories
from app.utils.dependencies import enforce_llm_quota, get_current_user

log = logging.getLogger(__name__)

router = APIRouter(prefix="/memories", tags=["memories"])


def _memory_response(
    memory: Memory,
    *,
    indexing: Literal["ready", "pending"] | None = None,
    state: str | None = None,
) -> MemoryResponse:
    """Map ORM Memory.extra_metadata to API field `metadata`.

    ``indexing`` is set by write paths only (POST/PATCH); read paths leave it
    None so a response never claims an index state it did not observe.

    ``state`` is the row's lifecycle label; when the caller already selected it
    in SQL the SQL value is passed through, otherwise ``state_of`` labels here.
    """
    return MemoryResponse(
        id=memory.id,
        user_id=memory.user_id,
        parent_id=memory.parent_id,
        source_type=memory.source_type,
        source_ref=memory.source_ref,
        source_url=memory.source_url,
        title=memory.title,
        content=memory.content,
        summary=memory.summary,
        tags=memory.tags or [],
        salience=memory.salience,
        pinned=memory.pinned,
        recall_count=memory.recall_count,
        last_used_at=memory.last_used_at,
        captured_at=memory.captured_at,
        indexed_at=memory.indexed_at,
        updated_at=memory.updated_at,
        revision=memory.revision or 1,  # unsaved/detached rows carry the column default
        indexing=indexing,
        state=state if state is not None else state_of(memory),
        metadata=memory.extra_metadata or {},
    )


async def _owned(memory: Memory | None, user: User) -> bool:
    """May ``user`` read/write ``memory``? Theirs AND in their namespace.

    The one spelling of the boundary for a row already in hand: a primary-key
    get (``db.get``) cannot carry a predicate, so the check is on the loaded row
    — against the same ``namespaces`` value the SQL predicate is built from. A
    foreign row and a missing one answer the same 404.
    """
    return (memory is not None
            and memory.user_id == user.id
            and namespace_of(memory) == personal_namespace(user.id))


@router.post("", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
async def create_memory(
    body: MemoryCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MemoryResponse:
    """Create a new memory. The owning user is taken from the auth context."""
    if body.parent_id is not None:
        # Parent must exist and belong to the caller — otherwise a client could
        # parent into (and later cascade-delete into) another user's subtree, or
        # into a namespace it cannot read.
        if not await _owned(await db.get(Memory, body.parent_id), current_user):
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Parent memory not found")
    # Compression-before-storage (feature-flagged, best-effort): long
    # bodies get an AI summary + compressed body before persisting. Any
    # failure stores the original content unchanged.
    content, summary = body.content, body.summary
    if body.auto_compress:
        from app.services.compression_service import compress_memory

        compressed = await compress_memory(body.content)
        if compressed is not None:
            content, summary = compressed[1], (summary or compressed[0])

    memory = Memory(
        user_id=current_user.id,
        namespace=personal_namespace(current_user.id),
        title=body.title,
        content=content,
        summary=summary,
        source_type=body.source_type,
        source_ref=body.source_ref,
        source_url=body.source_url,
        tags=body.tags,
        captured_at=body.captured_at or datetime.now(UTC),
        parent_id=body.parent_id,
        pinned=body.pinned,
        extra_metadata=client_metadata(body.metadata),
    )
    db.add(memory)
    # Durable index intent in the SAME commit as the row: if the process dies
    # before the vector write, the drain replays it. The write-through below
    # stays the fast path; the intent is the backstop.
    bump_revision(memory)
    await enqueue_upsert(db, memory)
    await db.commit()
    await db.refresh(memory)
    # Post-persist indexing (embed + graph) — best-effort, Postgres is truth.
    indexed = await index_new_memory(memory)
    if indexed:
        # The fast path has this revision in the index: ack the intent it
        # enqueued so a boot drain does not re-embed it.
        await mark_done(db, entity_id=memory.id, revision=memory.revision)
    return _memory_response(memory, indexing="ready" if indexed else "pending")


@router.get("", response_model=MemoryListResponse)
async def list_memories(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    source_type: Literal["manual_note", "file_upload", "google_drive", "notion",
                          "gmail", "web_clipper", "rss", "conversation_excerpt",
                          "chatgpt_import", "claude_import", "gemini_import", "copilot_import", "openclaw_import", "generic_import", "other"] | None = None,
    tag: str | None = Query(default=None, description="Filter by tag (exact match)"),
    query: str | None = Query(default=None, description="Substring search in title/content"),
    pinned: bool | None = None,
    sort: Literal["newest", "salience", "last_used"] = Query(default="newest"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> MemoryListResponse:
    """List memories for the current user with optional filters.

    Dirty rows are never listed (their derived view is stale); superseded rows
    are listed with ``state="superseded"`` — history stays readable, labeled.
    Only the caller's namespace is listed, and the pagination ``total`` carries
    the same predicate as the page.
    """
    namespace = namespace_predicate(personal_namespace(current_user.id))
    base = select(Memory, state_expression()).where(
        Memory.user_id == current_user.id, namespace, not_dirty_predicate())
    count_base = select(func.count(Memory.id)).where(
        Memory.user_id == current_user.id, namespace, not_dirty_predicate())

    if source_type:
        base = base.where(Memory.source_type == source_type)
        count_base = count_base.where(Memory.source_type == source_type)
    if pinned is not None:
        base = base.where(Memory.pinned == pinned)
        count_base = count_base.where(Memory.pinned == pinned)
    if tag:
        # tags is a JSON column (was a Postgres ARRAY) — cast to text and
        # match the quoted value; works identically on Postgres and SQLite.
        tag_match = f'%"{tag}"%'
        base = base.where(cast(Memory.tags, String).like(tag_match))
        count_base = count_base.where(cast(Memory.tags, String).like(tag_match))
    if query:
        # Case-insensitive substring match in title OR content
        pattern = f"%{query.lower()}%"
        title_match  = func.lower(Memory.title).like(pattern)
        content_match = func.lower(Memory.content).like(pattern)
        base = base.where(or_(title_match, content_match))
        count_base = count_base.where(or_(title_match, content_match))

    order_by = {
        "salience": (Memory.pinned.desc(), Memory.salience.desc(), Memory.captured_at.desc()),
        "last_used": (Memory.last_used_at.desc().nullslast(), Memory.captured_at.desc()),
        "newest": (Memory.captured_at.desc(),),
    }[sort]

    total = (await db.execute(count_base)).scalar_one()
    rows  = (await db.execute(
        base.order_by(*order_by).offset(offset).limit(limit)
    )).all()

    return MemoryListResponse(
        items=[_memory_response(m, state=state) for m, state in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


# Declared BEFORE /{memory_id} so "digest" isn't parsed as a memory UUID.
@router.get("/digest", response_model=DigestResponse)
async def memory_digest(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    window_days: int = Query(default=7, ge=1, le=90),
) -> DigestResponse:
    """Proactive surfacing: what you saved recently + 'on this day' from the past.

    Pull-based for now (the UI can render it on a home screen); a future
    scheduled job can push it via email using the same builder.
    """
    from app.services.digest_service import build_digest

    return await build_digest(db, current_user.id, window_days=window_days)


# Declared BEFORE /{memory_id} so "stats"/"digest" aren't parsed as UUIDs.
@router.get("/stats")
async def memory_stats(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """Aggregate counts for the memory dashboard (the caller's namespace only).

    Visible evidence only: dirty is "never served" and invalidated is history,
    so neither is counted here — the dashboard must agree with the surfaces.
    """
    namespace = namespace_predicate(personal_namespace(current_user.id))
    rows = (await db.execute(
        select(Memory.source_type, func.count(Memory.id))
        .where(Memory.user_id == current_user.id, namespace, not_dirty_predicate())
        .group_by(Memory.source_type)
    )).all()

    counts: dict[str, int] = {source: count for source, count in rows}
    total = sum(counts.values())

    week_ago = datetime.now(UTC) - timedelta(days=7)
    recent = (await db.execute(
        select(func.count(Memory.id)).where(
            Memory.user_id == current_user.id,
            namespace,
            not_dirty_predicate(),
            Memory.captured_at >= week_ago,
        )
    )).scalar_one()

    tags_rows = (await db.execute(
        select(Memory.tags).where(
            Memory.user_id == current_user.id, namespace, not_dirty_predicate()
        )
    )).scalars().all()
    tag_counts: dict[str, int] = {}
    for tags in tags_rows:
        for tag in tags or []:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    top_tags = sorted(
        ({"tag": t, "count": c} for t, c in tag_counts.items()),
        key=lambda x: x["count"], reverse=True,
    )[:10]

    return {
        "total_memories": total,
        "entities": counts.get("file_upload", 0),
        "relationships": counts.get("conversation_excerpt", 0),
        "observations": total,
        "concepts": counts.get("manual_note", 0),
        "recent_activity": [{"date": str(week_ago.date()), "count": recent}],
        "top_tags": top_tags,
    }


@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MemoryResponse:
    memory = await db.get(Memory, memory_id)
    if not await _owned(memory, current_user):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Memory not found.")
    return _memory_response(memory)


@router.patch("/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id: UUID,
    body: MemoryUpdate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MemoryResponse:
    memory = await db.get(Memory, memory_id)
    if not await _owned(memory, current_user):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Memory not found.")

    data = body.model_dump(exclude_unset=True)
    # Pydantic alias `metadata` maps to ORM attribute `extra_metadata` (the
    # underlying column is named "metadata", reserved by SQLAlchemy).
    if "metadata" in data:
        # Client metadata replaces only the CLIENT's own keys: the server-owned
        # ``cm_*`` entries already on the row ride through untouched — a PATCH
        # carrying ``{}`` (or forged markers) must never clear
        # ``cm_invalidated`` (un-forget) or fake lifecycle state.
        incoming = client_metadata(data.pop("metadata"))
        reserved = {key: value for key, value in (memory.extra_metadata or {}).items()
                    if isinstance(key, str) and key.startswith("cm_")}
        data["extra_metadata"] = {**incoming, **reserved}
    for field, value in data.items():
        setattr(memory, field, value)

    if data:
        bump_revision(memory)
        await enqueue_upsert(db, memory)
    await db.commit()
    await db.refresh(memory)
    # Write-through to the vector store (best-effort)
    indexed = await safe_upsert_to_index(memory)
    if indexed:
        await mark_done(db, entity_id=memory.id, revision=memory.revision)
    if not data and not indexed:
        # A no-op PATCH enqueues no intent, so a failed write-through must not
        # claim a durable 'pending' backstop that does not exist.
        return _memory_response(memory, indexing=None)
    return _memory_response(memory, indexing="ready" if indexed else "pending")


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(
    memory_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Delete one owned memory through the durable erasure path.

    Same response shape as before (404 for foreign/missing ids — no existence
    leak); the erase is one closure transaction that leaves a receipt and a
    durable delete intent per affected id.
    """
    if not await _owned(await db.get(Memory, memory_id), current_user):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Memory not found.")
    await erase_memories(db, current_user.id, [memory_id], requested_by="rest_api")


# ── Phase 3.7: recall endpoint ──────────────────────────────────────────────


@router.post("/recall", response_model=RecallResponse)
async def recall_memory(
    body: RecallRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    _quota: None = Depends(enforce_llm_quota),
) -> RecallResponse:
    """Personal-context recall: find memories matching the query.

    Pipeline (see :class:`MemoryRetriever` for details):

        1. Fetch personal context (pinned + recent).
        2. LLM rewrite the query + extract entities.
        3. Vector search in Qdrant.
        4. Hydrate + apply entity boost + time decay.
        5. Return top_k with trace (rewritten query, entities, latency).

    Every step degrades gracefully (empty ``results`` plus a ``trace``),
    with three exceptions: an embedding contract mismatch, an unreachable
    vector store, or a recall that waited out its freshness budget for a write
    still in flight answer 503 with a typed body (``embedding_contract_mismatch``
    / ``vector_unavailable`` / ``index_freshness_timeout``) — never a silent
    empty recall. A vector outage reaches that 503 only where the deployment
    has no lexical index (ruling R19): on SQLite the recall answers from the
    FTS5 lexical leg instead, with ``trace.counts["lexical"]`` set, no
    ``dense`` key, and ``retrieval.vector_unavailable`` counted.
    """
    retriever = MemoryRetriever(
        db=db,
        user_id=current_user.id,
    )
    return await retriever.recall(
        query=body.query,
        top_k=body.top_k,
        include_personal_context=body.include_personal_context,
    )


# ── Public Share Endpoint ────────────────────────────────────────────


class SharedMemoryResponse(BaseModel):
    id: str
    title: str
    content: str
    summary: str | None
    tags: list[str]
    created_at: str
    source_type: str


@router.get("/{memory_id}/share", response_model=SharedMemoryResponse)
async def get_shared_memory(
    memory_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SharedMemoryResponse:
    """Get a shared memory publicly (no auth required).

    Only memories with is_shared=True are accessible — and only in the public
    (personal) namespace: sharing a row never widens it into a namespace the
    public has no claim to. The link stays owner-agnostic.
    """
    memory = (await db.execute(
        select(Memory).where(
            Memory.id == memory_id,
            Memory.is_shared.is_(True),
            namespace_predicate(PERSONAL),
            not_dirty_predicate(),
        )
    )).scalar_one_or_none()
    if memory is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Memory not found or not shared")

    return SharedMemoryResponse(
        id=str(memory.id),
        title=memory.title,
        content=memory.content,
        summary=memory.summary,
        tags=memory.tags or [],
        created_at=memory.captured_at.isoformat() if memory.captured_at else "",
        source_type=memory.source_type,
    )
