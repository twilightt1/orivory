"""Document ingestion pipeline — synchronous in-process function.

Changes vs. original:
  - Uses build_parent_child_chunks() from smart chunker
  - Inserts PARENT chunks into document_chunks (DB) — returned to LLM
  - Inserts CHILD chunks into document_chunks with parent_id metadata
  - Embeds only CHILD chunks into ChromaDB
  - Caches PARENT chunks in Redis via parent_store
  - BM25 index built on PARENT content (better semantic units)
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class IngestionStageError(RuntimeError):
    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(message)


def _stage_error(stage: str, exc: Exception) -> IngestionStageError:
    message = str(exc) or exc.__class__.__name__
    if stage == "chroma_upsert":
        try:
            from app.config import settings

            message = (
                f"ChromaDB unavailable at {settings.CHROMA_HOST}:{settings.CHROMA_PORT}. "
                "Start docker compose service chromadb and retry ingestion. "
                f"Original error: {message}"
            )
        except Exception:
            message = f"ChromaDB unavailable. Original error: {message}"
    return IngestionStageError(stage, message)


def process_document_sync(document_id: str) -> None:
    """Run the ingestion pipeline for one document, synchronously.

    Never raises: on failure the document row is marked ``failed`` (same
    terminal state the old eager Celery task left behind) and the error is
    logged. Single attempt — there is no worker to back off to.
    """
    from app.database import sync_session

    with sync_session() as db:
        try:
            _ingest(db, document_id)
        except Exception as exc:
            db.rollback()
            stage = getattr(exc, "stage", "unknown")
            log.warning(
                "Ingestion failed",
                extra={"doc_id": document_id, "stage": stage, "error": str(exc)},
            )
            _fail(db, document_id, str(exc))
            log.error(
                "Ingestion failed permanently",
                extra={"doc_id": document_id, "stage": stage, "error": str(exc)},
            )


def _ingest(db, document_id: str) -> None:
    from sqlalchemy import delete

    from app import storage as minio
    from app.models.document import Document
    from app.models.document_chunk import DocumentChunk
    from app.retrieval.bm25_retriever import bm25_retriever
    from app.retrieval.parent_store import store_parents_sync
    from app.retrieval.retrieval_cache import invalidate_query_cache_sync
    from app.retrieval.vector_retriever import (
        delete_document_chunks_sync,
        upsert_chunks_sync,
    )
    from app.utils.chunker import build_parent_child_chunks, extract_text

    doc = db.get(Document, document_id)
    if not doc:
        log.error("Document not found", extra={"doc_id": document_id})
        return

    conversation_id = str(doc.conversation_id)
    doc.status = "processing"
    doc.error_msg = None
    db.commit()

    try:
        file_bytes = minio.get_object_sync(doc.file_path)
    except Exception as exc:
        raise _stage_error("minio_read", exc) from exc

    try:
        text = extract_text(file_bytes, doc.mime_type)
        if not text.strip():
            raise ValueError("Could not extract text content from file.")
    except Exception as exc:
        raise _stage_error("text_extraction", exc) from exc

    try:
        parents, children = build_parent_child_chunks(
            text=text,
            document_id=document_id,
            conversation_id=conversation_id,
            filename=doc.filename,
        )
        if not children:
            raise ValueError("No chunks produced from document.")
    except Exception as exc:
        raise _stage_error("chunking", exc) from exc

    try:
        delete_document_chunks_sync(conversation_id, document_id)
        db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
        db.flush()
    except Exception as exc:
        raise _stage_error("cleanup_existing_chunks", exc) from exc

    try:
        for parent in parents:
            db.add(
                DocumentChunk(
                    id=parent.id,
                    document_id=document_id,
                    content=parent.content,
                    chunk_index=parent.index,
                    chunk_metadata=parent.metadata,
                )
            )
        db.flush()

        for child in children:
            db.add(
                DocumentChunk(
                    id=child.id,
                    document_id=document_id,
                    content=child.content,
                    chunk_index=child.index,
                    chunk_metadata=child.metadata,
                )
            )
        db.flush()
    except Exception as exc:
        raise _stage_error("db_insert_chunks", exc) from exc

    try:
        store_parents_sync(
            conversation_id,
            [{"id": p.id, "content": p.content, "metadata": p.metadata} for p in parents],
        )
    except Exception as exc:
        raise _stage_error("redis_parent_cache", exc) from exc

    child_dicts = [
        {"id": child.id, "content": child.content, "metadata": child.metadata}
        for child in children
    ]
    try:
        upsert_chunks_sync(conversation_id, child_dicts)
    except Exception as exc:
        raise _stage_error("chroma_upsert", exc) from exc

    parent_dicts = [
        {"id": parent.id, "content": parent.content, "metadata": parent.metadata}
        for parent in parents
    ]
    try:
        bm25_retriever.publish_build_sync(conversation_id, parent_dicts)
    except Exception as exc:
        raise _stage_error("bm25_build", exc) from exc

    doc.status = "ready"
    doc.error_msg = None
    doc.chunk_count = len(parents)
    try:
        db.commit()
    except Exception as exc:
        raise _stage_error("db_commit", exc) from exc

    try:
        invalidate_query_cache_sync(conversation_id)
    except Exception as exc:
        log.warning(
            "Retrieval query cache invalidation failed",
            extra={"doc_id": document_id, "conversation_id": conversation_id, "error": str(exc)},
        )

    # Unify (roadmap P1.1): project this document into the user's cross-
    # conversation memory. Best-effort — the document is already "ready" and
    # the per-conversation path works regardless; a failure here is replayable
    # via the reindex helper and must not fail ingestion.
    try:
        _project_document_to_memories(db, document_id, parents)
    except Exception as exc:
        log.warning(
            "Doc→memory projection failed",
            extra={"doc_id": document_id, "conversation_id": conversation_id, "error": str(exc)},
        )

    log.info(
        "Ingestion complete",
        extra={
            "doc_id": document_id,
            "conversation_id": conversation_id,
            "parents": len(parents),
            "children": len(children),
        },
    )


def _project_document_to_memories(db, document_id: str, parents) -> None:
    """Create + embed cross-conversation memories for an ingested document.

    Commits the new Memory rows — with their durable index intents — then
    embeds them into the shared memory collection. Embedding is best-effort
    (replayable via reindex); the rows are the durable source of truth.
    """
    from sqlalchemy import select

    from app.ingestion.document_memory import build_document_memories_sync
    from app.models.conversation import Conversation
    from app.models.document import Document
    from app.models.memory import Memory
    from app.retrieval.memory.outbox import mark_done_sync
    from app.retrieval.memory.vector_store import (
        delete_memories_sync,
        upsert_memories_sync,
    )

    # The owner is the conversation's owner (never a caller-supplied id): the
    # projection queries below must not read or embed another user's rows.
    doc = db.get(Document, document_id)
    conversation = db.get(Conversation, doc.conversation_id) if doc is not None else None
    if conversation is None:
        log.warning(
            "Doc→memory projection skipped: document or conversation not found",
            extra={"doc_id": document_id},
        )
        return
    user_id = conversation.user_id

    result = build_document_memories_sync(db, document_id, parents, user_id=user_id)
    db.commit()

    # Purge vectors from a prior projection whose Postgres rows were just
    # deleted (re-ingest with different/fewer chunks → different ids).
    if result.stale_vector_ids:
        delete_memories_sync(result.stale_vector_ids)

    if not result.all_ids:
        return

    rows = (
        db.execute(
            select(Memory).where(
                Memory.source_ref == document_id,
                Memory.user_id == user_id,
            )
        )
        .scalars()
        .all()
    )
    if rows and upsert_memories_sync(rows):
        # The vectors are in: ack the intents committed with the rows, so a
        # boot drain does not re-embed the whole projection.
        for memory in rows:
            mark_done_sync(db, entity_id=memory.id, revision=memory.revision)


def _fail(db, document_id: str, error: str) -> None:
    try:
        from app.models.document import Document

        doc = db.get(Document, document_id)
        if doc:
            doc.status = "failed"
            doc.error_msg = error[:500]
            conversation_id = str(doc.conversation_id)
            db.commit()
            from app.retrieval.retrieval_cache import invalidate_query_cache_sync

            invalidate_query_cache_sync(conversation_id)
    except Exception:
        pass
