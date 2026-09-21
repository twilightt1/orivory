"""Document ingestion pipeline — synchronous in-process function.

Changes vs. original:
  - Uses build_parent_child_chunks() from smart chunker
  - Inserts PARENT chunks into document_chunks (DB) — returned to LLM
  - Inserts CHILD chunks into document_chunks with parent_id metadata
  - Embeds only CHILD chunks, into the ACTIVE chunk generation (Qdrant)
  - The chunk rows and their durable index intents (delete for the ids
    leaving, upsert for the children entering) share ONE transaction; the
    vector write is the post-commit attempt, never the record of truth
  - Caches PARENT chunks in Redis via parent_store
  - BM25 index built on PARENT content (better semantic units)
"""
from __future__ import annotations

import hashlib
import logging
import uuid

log = logging.getLogger(__name__)


class IngestionStageError(RuntimeError):
    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(message)


def _stage_error(stage: str, exc: Exception) -> IngestionStageError:
    message = str(exc) or exc.__class__.__name__
    return IngestionStageError(stage, message)


def _document_owner(db, doc) -> str:
    """The document's conversation owner — the tenant its chunks are indexed under.

    Never a caller-supplied id: the chunk payload's tenant clause and the
    intents' ``tenant_id`` both come from here.
    """
    from app.models.conversation import Conversation

    conversation = db.get(Conversation, doc.conversation_id)
    if conversation is None:
        raise ValueError(f"document {doc.id} has no conversation owner")
    return str(conversation.user_id)


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
    from sqlalchemy import delete, select

    from app import storage as minio
    from app.models.document import Document
    from app.models.document_chunk import DocumentChunk
    from app.retrieval.bm25_retriever import bm25_retriever
    from app.retrieval.memory.outbox import (
        enqueue_chunk_delete_sync,
        enqueue_chunk_upsert_sync,
    )
    from app.retrieval.parent_store import store_parents_sync
    from app.retrieval.retrieval_cache import invalidate_query_cache_sync
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
        raise _stage_error("storage_read", exc) from exc
    # The projection's content hash comes from THESE bytes (R38): read once,
    # hash once, hand it down — a second storage read could fail on its own and
    # leave the re-upload guard blind.
    upload_content_hash = hashlib.sha256(file_bytes).hexdigest()

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

    # Canonical chunk transaction (spec §5.1): every id leaving SQL carries a
    # durable delete intent, every child row carries a durable upsert intent,
    # and all of it commits with the rows below. Indexing never happens before
    # that commit — a crash leaves a replayable intent, never an orphan point.
    try:
        user_id = _document_owner(db, doc)
        old_ids = db.execute(
            select(DocumentChunk.id).where(DocumentChunk.document_id == doc.id)
        ).scalars().all()
        db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc.id))
        db.flush()
        quarantined = enqueue_chunk_delete_sync(db, chunk_ids=old_ids, tenant_id=user_id)

        # A reingest mints NEW chunk ids (the chunker mints a uuid4 per chunk),
        # so the old points are removed by the delete intents above and the new
        # rows start at revision 1 — no in-place bump is needed (ruling R16).
        for chunk in (*parents, *children):
            db.add(
                DocumentChunk(
                    id=uuid.UUID(str(chunk.id)),
                    document_id=doc.id,
                    content=chunk.content,
                    chunk_index=chunk.index,
                    revision=1,
                    chunk_metadata=chunk.metadata,
                )
            )
        db.flush()
        for child in children:
            quarantined += enqueue_chunk_upsert_sync(
                db, chunk_id=child.id, tenant_id=user_id, revision=1,
                conversation_id=conversation_id,
            )
        if quarantined:
            log.warning(
                "Chunk ids quarantined (not UUIDs): never indexed",
                extra={"doc_id": document_id, "n": len(quarantined)},
            )
    except Exception as exc:
        raise _stage_error("chunk_transaction", exc) from exc

    try:
        store_parents_sync(
            conversation_id,
            [{"id": p.id, "content": p.content, "metadata": p.metadata} for p in parents],
        )
    except Exception as exc:
        raise _stage_error("redis_parent_cache", exc) from exc

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

    # Immediate attempt after the commit (spec §5.2): the intents above are the
    # durable proof, so a vector outage must not fail an ingested document.
    try:
        _index_document_chunks(db, children=children, old_ids=old_ids,
                               user_id=user_id, conversation_id=conversation_id)
    except Exception as exc:
        log.warning(
            "Immediate chunk index attempt failed; the intents stay pending",
            extra={"doc_id": document_id, "error": str(exc)},
        )

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
        _project_document_to_memories(db, document_id, parents,
                                      content_hash=upload_content_hash)
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


def _index_document_chunks(db, *, children, old_ids, user_id: str,
                           conversation_id: str) -> None:
    """The post-commit fast path: write the new children, ack, then forget the old.

    The ORDER is the fence (ruling R9): the purge names ONLY the ids whose rows
    left SQL in this same commit, so it can never delete a point a concurrent
    drain has just written and acked for a row that is still live — that ack is
    final, nothing is left pending to replay it. Each child's ack still follows
    its own vector write; a purge that cannot be confirmed only leaves the
    delete intents pending (the durable proof), never a live row without a point.
    """
    from app.models.document_chunk import DocumentChunk
    from app.retrieval.memory.outbox import KIND_CHUNK, mark_done_sync
    from app.retrieval.vector_retriever import delete_chunks_by_ids, upsert_chunks_sync

    rows = [db.get(DocumentChunk, uuid.UUID(str(child.id))) for child in children]
    rows = [row for row in rows if row is not None]
    if rows and upsert_chunks_sync(rows, user_id=user_id):
        for row in rows:
            mark_done_sync(db, entity_id=row.id, revision=row.revision, kind=KIND_CHUNK)

    delete_chunks_by_ids(
        [str(chunk_id) for chunk_id in old_ids],
        user_id=user_id,
        conversation_id=conversation_id,
    )


def _project_document_to_memories(db, document_id: str, parents,
                                  content_hash: str | None = None) -> None:
    """Create + embed cross-conversation memories for an ingested document.

    Commits the new Memory rows — with their durable index intents — then
    embeds them into the shared memory collection. Embedding is best-effort
    (replayable via reindex); the rows are the durable source of truth.

    R38 (P4b/T4): the projection records the sha256 of the UPLOADED BYTES in its
    metadata. That is what lets a later forget pin the bytes — not only the doc
    id — in the suppression ledger, and so catch a re-upload of the same file
    (a new document id can never match on ``source_ref``).

    ``content_hash`` is handed down by the ingest stage (which already read the
    exact same bytes for text extraction, so the hash cannot fail independently
    of the ingest itself). When None — a direct caller that never read the
    bytes, tests — the hash is read from the stored object; an unavailable hash
    then leaves the projection intact with the key absent ("unknown").
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
    if doc is None or conversation is None:
        log.warning(
            "Doc→memory projection skipped: document or conversation not found",
            extra={"doc_id": document_id},
        )
        return
    user_id = conversation.user_id

    if content_hash is None:
        try:
            # Fallback only (direct callers/tests): the ingest stage hands the
            # hash down from the bytes it already read — no second object read.
            from app import storage

            content_hash = hashlib.sha256(storage.get_object_sync(doc.file_path)).hexdigest()
        except Exception as exc:
            # The hash is the re-upload guard's key, not the projection's truth:
            # an unreadable object leaves the rows (with an absent hash) in place.
            log.warning(
                "Projection content hash unavailable",
                extra={"doc_id": document_id, "error": str(exc)},
            )

    result = build_document_memories_sync(db, document_id, parents, user_id=user_id,
                                          content_hash=content_hash)
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
