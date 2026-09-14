"""
Chat router — conversations, messages, documents (nested), SSE streaming.
All document operations are scoped to the parent conversation.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.state import AgentState
from app.api.v1.sse import format_sse
from app.config import settings
from app.database import get_db
from app.middleware.rate_limiter import check_rate_limit
from app.models.conversation import Conversation
from app.models.message import Message
from app.schemas.conversation import (
    ChatRequest,
    ConversationCreate,
    ConversationDetail,
    ConversationResponse,
    ConversationUpdate,
    DocumentResponse,
    MessageResponse,
)
from app.services import document_service
from app.services.quota_service import check_and_increment
from app.utils.dependencies import get_current_active_user

router = APIRouter(prefix="/chat", tags=["chat"])
log    = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Root chat endpoint (POST /chat) - creates conversation if needed
# ─────────────────────────────────────────────────────────────────────────────


class RootChatRequest(BaseModel):
    """Request for root /chat endpoint."""
    query: str | None = None  # Backend prefers "query"
    content: str | None = None  # Frontend sends "content"
    session_id: str | None = None
    # Defaults must mirror ChatRequest (True): the frontend never sends these
    # fields, so False silently disabled personal-memory retrieval and graph
    # context for every chat request — documents were embedded but never used
    # to ground answers, forcing the LLM to hallucinate.
    include_personal_context: bool = True
    include_graph_context: bool = True
    personal_memory_top_k: int = Field(default=5, ge=0, le=10)

    def get_query(self) -> str:
        """Get query from either field."""
        return (self.query or self.content or "").strip()


@router.post("", response_class=StreamingResponse)
async def root_chat(
    body: RootChatRequest,
    current_user=Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Root chat endpoint - creates session if needed and streams response.

    Frontend calls /api/v1/chat with {query, session_id}.
    This creates a conversation if session_id not provided.
    """
    import uuid

    # Get or create conversation
    query = body.get_query()

    # Validate before touching the DB so bad requests don't leave an
    # orphaned empty conversation behind (they also 500'd on ChatRequest).
    if not query:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Message content is required.")

    if body.session_id:
        try:
            conv_id = uuid.UUID(body.session_id)
            conv = await db.scalar(
                select(Conversation).where(
                    Conversation.id == conv_id,
                    Conversation.user_id == current_user.id,
                )
            )
        except ValueError:
            conv = None
    else:
        conv = None

    if not conv:
        # Create new conversation
        title = query[:50] + "..." if len(query) > 50 else query
        conv = Conversation(
            user_id=current_user.id,
            title=title,
            document_count=0,
        )
        db.add(conv)
        await db.commit()
        await db.refresh(conv)

    # Redirect to the conversation message endpoint
    # Reuse the send_message logic
    chat_body = ChatRequest(
        query=query,
        include_personal_context=body.include_personal_context,
        include_graph_context=body.include_graph_context,
        personal_memory_top_k=body.personal_memory_top_k,
    )

    return await send_message(chat_body, conv, current_user, db)


# ─────────────────────────────────────────────────────────────────────────────
# Session aliases (frontend uses /sessions, backend uses /conversations)
# ─────────────────────────────────────────────────────────────────────────────

SessionResponse = ConversationResponse


@router.get("/sessions", response_model=list[SessionResponse])
async def list_sessions(
    current_user=Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Alias for /conversations - frontend compatibility."""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
    )
    return result.scalars().all()


@router.post("/sessions", response_model=SessionResponse, status_code=201)
async def create_session(
    body: ConversationCreate | None = None,
    current_user=Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Alias for /conversations - frontend compatibility."""
    title = body.title if body else "New Chat"
    conv = Conversation(user_id=current_user.id, title=title, document_count=0)
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return conv


@router.get("/sessions/{session_id}", response_model=ConversationDetail)
async def get_session(
    session_id: UUID,
    current_user=Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Get session details with messages and documents."""
    conv = await db.scalar(
        select(Conversation).where(
            Conversation.id == session_id,
            Conversation.user_id == current_user.id,
        )
    )
    if not conv:
        raise HTTPException(404, detail="Session not found.")
    docs = await document_service.list_documents(db, conv.id)
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conv.id)
        .order_by(Message.created_at.asc())
    )
    messages = result.scalars().all()
    return ConversationDetail(
        **ConversationResponse.model_validate(conv).model_dump(),
        documents=[DocumentResponse.model_validate(d) for d in docs],
        messages=[MessageResponse.model_validate(m) for m in messages],
    )


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: UUID,
    current_user=Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Get messages for a session."""
    conv = await db.scalar(
        select(Conversation).where(
            Conversation.id == session_id,
            Conversation.user_id == current_user.id,
        )
    )
    if not conv:
        raise HTTPException(404, detail="Session not found.")
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conv.id)
        .order_by(Message.created_at.asc())
    )
    messages = result.scalars().all()
    return {"messages": [MessageResponse.model_validate(m) for m in messages]}


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: UUID,
    current_user=Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a session."""
    conv = await db.scalar(
        select(Conversation).where(
            Conversation.id == session_id,
            Conversation.user_id == current_user.id,
        )
    )
    if not conv:
        raise HTTPException(404, detail="Session not found.")
    await db.delete(conv)
    await db.commit()


# LangGraph agents dropped — chat endpoints that need the graph raise 501.
# ``rag_graph`` stays None so tests can still monkeypatch(chat, "rag_graph", fake).
rag_graph = None


def _load_rag_graph():
    raise RuntimeError("RAG graph removed with LangGraph agents")


def _get_rag_graph():
    """Return the current rag_graph (allows monkeypatching in tests)."""
    return rag_graph


def _source_event_payload(chunk: dict[str, Any]) -> dict[str, Any]:
    metadata = chunk.get("metadata") or {}
    return {
        "content": chunk.get("content", "")[:200],
        "filename": metadata.get("filename", ""),
        "score": round(chunk.get("rerank_score", chunk.get("score", 0)), 4),
        "source_type": metadata.get("source_type"),
        "memory_id": metadata.get("memory_id"),
        "entity_names": metadata.get("entity_names"),
    }



async def _get_conversation(
    conversation_id: UUID,
    current_user=Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Conversation:
    conv = await db.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
    )
    if not conv:
        raise HTTPException(404, detail="Conversation not found.")
    return conv



@router.get("/conversations", response_model=list[ConversationResponse])
async def list_conversations(
    current_user=Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
    )
    return result.scalars().all()


@router.post("/conversations", response_model=ConversationResponse, status_code=201)
async def create_conversation(
    body: ConversationCreate,
    current_user=Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    conv = Conversation(user_id=current_user.id, title=body.title, document_count=0)
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return conv


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation: Conversation = Depends(_get_conversation),
    db: AsyncSession = Depends(get_db),
):
    docs = await document_service.list_documents(db, conversation.id)
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at.asc())
    )
    messages = result.scalars().all()
    return ConversationDetail(
        **ConversationResponse.model_validate(conversation).model_dump(),
        documents=[DocumentResponse.model_validate(d) for d in docs],
        messages=[MessageResponse.model_validate(m) for m in messages],
    )


@router.patch("/conversations/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    body: ConversationUpdate,
    conversation: Conversation = Depends(_get_conversation),
    db: AsyncSession = Depends(get_db),
):
    conversation.title = body.title
    await db.commit()
    await db.refresh(conversation)
    return conversation


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation: Conversation = Depends(_get_conversation),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import select

    from app.ingestion.document_memory import delete_document_memories_async
    from app.models.document import Document
    from app.retrieval.bm25_retriever import bm25_retriever
    from app.retrieval.memory.vector_store import (
        delete_memories as delete_memory_vectors,
    )
    from app.retrieval.vector_retriever import delete_conversation_collection

    conv_id = str(conversation.id)

    # Unify (P1.1): the conversation cascade-deletes its documents, so first
    # remove the cross-conversation memories those documents projected.
    doc_ids = (
        await db.execute(select(Document.id).where(Document.conversation_id == conversation.id))
    ).scalars().all()
    removed_memory_ids: list[str] = []
    for doc_id in doc_ids:
        removed_memory_ids.extend(
            await delete_document_memories_async(db, str(doc_id), user_id=conversation.user_id)
        )

    await db.delete(conversation)
    await db.commit()

    if removed_memory_ids:
        await delete_memory_vectors(removed_memory_ids)
    await delete_conversation_collection(conv_id)
    await bm25_retriever.publish_invalidate_async(conv_id)



@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=list[MessageResponse],
)
async def list_messages(
    conversation: Conversation = Depends(_get_conversation),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at.asc())
    )
    return result.scalars().all()



@router.post("/conversations/{conversation_id}/message")
async def send_message(
    body: ChatRequest,
    conversation: Conversation = Depends(_get_conversation),
    current_user=Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):

    await check_rate_limit(str(current_user.id), window_seconds=60, limit=settings.RATE_LIMIT_PER_MINUTE)
    await check_and_increment(current_user.id, db)

    state = AgentState(
        user_id=str(current_user.id),
        conversation_id=str(conversation.id),
        query=body.query,
        query_type="",
        history=[],
        bm25_results=[],
        vector_results=[],
        fused_chunks=[],
        reranked_chunks=[],
        response="",
        token_count=0,
        agent_trace={},
        error=None,
        should_stream=True,
        has_documents=conversation.document_count > 0,
        document_count=conversation.document_count,
        personal_memory_enabled=body.include_personal_context,
        graph_context_enabled=body.include_graph_context,
        personal_memory_top_k=body.personal_memory_top_k,
        doc_context_chunks=[],
        personal_memory_chunks=[],
        graph_context_chunks=[],
        grounding_context_chunks=[],
    )

    async def event_stream():
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        final_state: dict[str, Any] = {}
        final_response_emitted = False

        async def emit(data: dict[str, Any], event: str | None = None) -> None:
            await queue.put({"event": event, "data": data})

        async def stream_token(delta: str) -> None:
            return

        state["_stream_callback"] = stream_token

        async def run_graph() -> None:
            nonlocal final_response_emitted
            try:
                graph = _get_rag_graph()
                if graph is None:
                    await emit(
                        {"type": "error", "message": "Chat RAG graph is unavailable (LangGraph agents removed)."},
                        event="error",
                    )
                    return
                await emit({"type": "status", "stage": "started"}, event="status")
                async for event in graph.astream(state):
                    node, data = next(iter(event.items()))
                    if isinstance(data, dict):
                        final_state.update(data)

                    retry_count = final_state.get("retry_count", 0)
                    is_retry_stage = node.startswith("retry_") or node.startswith("record_")
                    await emit(
                        {
                            "type": "status",
                            "stage": node,
                            "retry_count": retry_count,
                            "attempt": retry_count + 1,
                            "category": "retry" if is_retry_stage else "progress",
                        },
                        event="status",
                    )

                    if node == "save":
                        response = final_state.get("response", "")
                        if response and not final_response_emitted:
                            final_response_emitted = True
                            await emit(
                                {
                                    "type": "token",
                                    "content": response,
                                    "retry_count": final_state.get("retry_count", 0),
                                    "mode": "final_evaluated_response",
                                },
                                event="token",
                            )

                        sources = [
                            _source_event_payload(c)
                            for c in final_state.get("reranked_chunks", [])
                        ]
                        await emit({"type": "sources", "sources": sources}, event="sources")
                        await emit(
                            {
                                "type": "trace",
                                "agent_trace": final_state.get("agent_trace", {}),
                            },
                            event="trace",
                        )
                        await emit(
                            {
                                "type": "done",
                                "sources": sources,
                                "token_count": final_state.get("token_count", 0),
                                "retry_count": final_state.get("retry_count", 0),
                                "grounding": final_state.get("agent_trace", {}).get("grounding"),
                            },
                            event="done",
                        )
                if "response" not in final_state:
                    await emit({"type": "done", "sources": []}, event="done")
            except Exception as e:
                log.error("Stream error", extra={"error": str(e)})
                await emit(
                    {"type": "error", "message": "An error occurred."},
                    event="error",
                )
            finally:
                await queue.put(None)

        graph_task = asyncio.create_task(run_graph())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield format_sse(item["data"], event=item.get("event"))
        finally:
            if not graph_task.done():
                graph_task.cancel()
                with suppress(asyncio.CancelledError):
                    await graph_task

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )



@router.get(
    "/conversations/{conversation_id}/documents",
    response_model=list[DocumentResponse],
)
async def list_documents(
    conversation: Conversation = Depends(_get_conversation),
    db: AsyncSession = Depends(get_db),
):
    return await document_service.list_documents(db, conversation.id)


@router.post(
    "/conversations/{conversation_id}/documents",
    response_model=DocumentResponse,
    status_code=202,
)
async def upload_document(
    file: UploadFile = File(...),
    conversation: Conversation = Depends(_get_conversation),
    db: AsyncSession = Depends(get_db),
):
    return await document_service.upload_document(db, conversation, file)


@router.get(
    "/conversations/{conversation_id}/documents/{document_id}",
    response_model=DocumentResponse,
)
async def get_document_status(
    document_id: UUID,
    conversation: Conversation = Depends(_get_conversation),
    db: AsyncSession = Depends(get_db),
):
    return await document_service.get_document(db, document_id, conversation.id)


@router.delete(
    "/conversations/{conversation_id}/documents/{document_id}",
    status_code=204,
)
async def delete_document(
    document_id: UUID,
    conversation: Conversation = Depends(_get_conversation),
    db: AsyncSession = Depends(get_db),
):
    doc = await document_service.get_document(db, document_id, conversation.id)
    await document_service.delete_document(db, doc, conversation)


# ─────────────────────────────────────────────────────────────────────────────
# Root document delete (DELETE /chat/documents/{id})
# ─────────────────────────────────────────────────────────────────────────────

@router.delete(
    "/documents/{document_id}",
    status_code=204,
)
async def delete_document_root(
    document_id: UUID,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a document without requiring conversation ID."""
    from sqlalchemy import select

    from app.models.document import Document

    # Find the document
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.conversation.has(user_id=current_user.id)
        )
    )
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(404, detail="Document not found")

    # Get conversation for deletion
    conv_result = await db.execute(
        select(Conversation).where(Conversation.id == doc.conversation_id)
    )
    conversation = conv_result.scalar_one()

    await document_service.delete_document(db, doc, conversation)


# ─────────────────────────────────────────────────────────────────────────────
# Root document upload (POST /chat/documents) - creates default session if needed
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/documents",
    response_model=DocumentResponse,
    status_code=202,
)
async def upload_document_root(
    file: UploadFile = File(...),
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload a document without requiring a conversation ID.
    Creates a default conversation for the user if needed.
    """
    from app.models.conversation import Conversation

    # Get or create default conversation
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.created_at.desc())
        .limit(1)
    )
    conversation = result.scalar_one_or_none()

    if not conversation:
        # Create a new conversation
        conversation = Conversation(
            user_id=current_user.id,
            title="Default",
        )
        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)

    return await document_service.upload_document(db, conversation, file)
