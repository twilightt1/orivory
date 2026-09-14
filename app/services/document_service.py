"""Document service — scoped to conversation."""
from __future__ import annotations

import logging
import uuid
from uuid import UUID

from fastapi import HTTPException, UploadFile
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Conversation
from app.models.document import Document

log = logging.getLogger(__name__)

ALLOWED_MIME = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/markdown",
}
# Some browsers upload Markdown with an octet-stream / missing MIME — accept
# these by extension as well (the ingestion connector already parses .md).
ALLOWED_EXTENSIONS = (".md", ".txt", ".pdf", ".docx")
MAX_SIZE = 50 * 1024 * 1024
MAX_DOCS = 20


async def upload_document(db: AsyncSession, conversation: Conversation, file: UploadFile) -> Document:
    name = (file.filename or "").lower()
    if file.content_type not in ALLOWED_MIME and not name.endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(
            400,
            detail="Only PDF, DOCX, TXT, and Markdown (MD) files are supported.",
        )
    content = await file.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(413, detail="File exceeds 50 MB limit.")
    if conversation.document_count >= MAX_DOCS:
        raise HTTPException(400, detail=f"Maximum {MAX_DOCS} documents per conversation.")

    doc_id    = str(uuid.uuid4())
    file_path = f"{conversation.id}/{doc_id}_{file.filename}"

    from app import storage
    await storage.put_object(file_path, content, file.content_type)

    doc = Document(
        id=doc_id,
        conversation_id=conversation.id,
        filename=file.filename,
        file_path=file_path,
        file_size=len(content),
        mime_type=file.content_type,
        status="pending",
    )
    db.add(doc)
    conversation.document_count += 1
    await db.commit()
    await db.refresh(doc)

    from app.ingestion.pipeline import process_document_sync
    from app.retrieval.retrieval_cache import invalidate_query_cache

    await invalidate_query_cache(str(conversation.id))
    process_document_sync(str(doc.id))

    log.info("Document uploaded", extra={"doc_id": doc_id, "conversation_id": str(conversation.id)})
    return doc


async def list_documents(db: AsyncSession, conversation_id: UUID) -> list[Document]:
    result = await db.execute(
        select(Document)
        .where(Document.conversation_id == conversation_id)
        .order_by(Document.created_at.desc())
    )
    return list(result.scalars().all())


async def get_document(db: AsyncSession, document_id: UUID, conversation_id: UUID) -> Document:
    doc = await db.scalar(
        select(Document).where(and_(
            Document.id == document_id,
            Document.conversation_id == conversation_id,
        ))
    )
    if not doc:
        raise HTTPException(404, detail="Document not found.")
    return doc


async def delete_document(db: AsyncSession, document: Document, conversation: Conversation) -> None:
    from app import storage
    from app.ingestion.document_memory import delete_document_memories_async
    from app.models.document_chunk import DocumentChunk
    from app.retrieval.bm25_retriever import bm25_retriever
    from app.retrieval.memory.outbox import enqueue_chunk_delete
    from app.retrieval.memory.vector_store import (
        delete_memories as delete_memory_vectors,
    )
    from app.retrieval.retrieval_cache import invalidate_query_cache
    from app.retrieval.vector_retriever import delete_document_chunks

    document_id = str(document.id)

    try:
        await storage.remove_object(document.file_path)
    except Exception as e:
        log.warning("MinIO delete failed", extra={"error": str(e)})

    # Unify (P1.1): also remove the cross-conversation memories derived from
    # this document. Rows AND their durable delete intents ride the commit
    # below; the vector purges (chunk + memory) come after it — never before
    # it — so a failed delete can never take the DB rows with it.
    memory_ids = await delete_document_memories_async(db, document_id, user_id=conversation.user_id)

    # The chunk rows cascade with the document: their ids must be captured —
    # with a durable delete intent each — BEFORE the rows go (spec §5.4).
    chunk_ids = (
        await db.execute(
            select(DocumentChunk.id).where(DocumentChunk.document_id == document.id)
        )
    ).scalars().all()
    if chunk_ids:
        await enqueue_chunk_delete(db, chunk_ids=chunk_ids, tenant_id=conversation.user_id)

    await db.delete(document)
    conversation.document_count = max(0, conversation.document_count - 1)
    await db.commit()

    # Immediate attempt; the intents above are the durable proof each point goes.
    await delete_document_chunks(
        str(conversation.id), document_id, user_id=str(conversation.user_id)
    )

    if memory_ids:
        await delete_memory_vectors(memory_ids)

    await bm25_retriever.publish_rebuild_async(db, str(conversation.id))
    await invalidate_query_cache(str(conversation.id))
    log.info("Document deleted", extra={"doc_id": document_id, "memories_removed": len(memory_ids)})
