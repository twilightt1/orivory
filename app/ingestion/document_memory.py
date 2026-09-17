"""Turn an ingested document into cross-conversation personal memories.

Part of the "unify the two worlds" work (roadmap P1.1). A document uploaded
into one conversation only lives in that conversation's per-conversation
vector index. To make it part of the user's second brain — recallable from
*any* conversation — we also project it into the ``memories`` table and the
shared ``Orivory_memories`` vector collection.

Granularity = **hybrid** (1 document + N passages):

    Memory(kind="document")            title=filename, content=summary
      ├─ Memory(kind="passage")        parent_id=doc, content=parent chunk
      ├─ Memory(kind="passage")
      └─ ...                            one per parent chunk

The document-level row is a single handle the user sees in their memory list;
the passage rows give fine-grained, citable recall. ``document_chunks`` remains
the high-fidelity per-conversation citation layer — this is additive.

Linkage is by ``Memory.source_ref == document_id`` (there is no FK from Memory
to Document). ``source_ref`` is caller-supplied data, so every cleanup /
projection query is scoped by ``(source_ref, user_id)``: a foreign memory that
happens to share a ``source_ref`` is never read, deleted or re-embedded. See
``delete_document_memories_sync`` and the async variant used by API deletes.

Every removed/new projection row leaves one durable ``index_outbox`` intent in
the caller's transaction (spec §5.1), so a crash before the vector write is
replayable. ``memory_suppressions`` pins a forgotten identity: re-ingest skips
it instead of resurrecting the memory (spec §5.4/§12.3).

Runs in the **synchronous** Celery ingestion context (the async faces below
exist for the API delete / forget paths).
"""
from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.models.conversation import Conversation
from app.models.document import Document
from app.models.memory import Memory, MemorySuppression
from app.retrieval.memory.outbox import (
    bump_revision,
    enqueue_delete,
    enqueue_delete_sync,
    enqueue_upsert_sync,
)
from app.utils.chunker import ParentChunk

log = logging.getLogger(__name__)

DOC_MEMORY_SOURCE_TYPE = "file_upload"
_SUMMARY_MAX_CHARS = 1500


@dataclass
class DocMemoryResult:
    document_id: str
    doc_memory_id: str | None
    passage_memory_ids: list[str]
    # Ids of memories from a *prior* projection of this document that were
    # deleted (re-ingest). Callers purge their stale vectors from the index.
    removed_memory_ids: list[str] = field(default_factory=list)

    @property
    def all_ids(self) -> list[str]:
        ids = [self.doc_memory_id] if self.doc_memory_id else []
        ids.extend(self.passage_memory_ids)
        return ids

    @property
    def stale_vector_ids(self) -> list[str]:
        """Prior-projection ids that no longer have a Postgres row."""
        current = set(self.all_ids)
        return [i for i in self.removed_memory_ids if i not in current]


def _document_summary(parents: list[ParentChunk]) -> str:
    """Cheap summary = the first parent chunk, truncated. No extra LLM cost."""
    if not parents:
        return ""
    head = parents[0].content.strip()
    return head[:_SUMMARY_MAX_CHARS]


# ── suppression ledger: a forgotten identity stays forgotten ────────────────

#: The projection metadata key carrying the upload-time content hash (R38).
#: Computed from the uploaded BYTES at ingest, never backfilled onto rows that
#: predate it — an absent key is the honest "unknown", not an invented hash.
CONTENT_HASH_KEY = "content_hash"


def _ledger_stmt(column, user_id):
    """One ledger column, scoped to one owner — the probe's shared half."""
    return select(column).where(MemorySuppression.user_id == user_id)


def _suppression_stmt(user_id, source_ref: str):
    return _ledger_stmt(MemorySuppression.id, user_id).where(
        MemorySuppression.source_ref == source_ref)


def is_suppressed(db: Session, *, user_id, source_ref: str) -> bool:
    """True when this (user, source_ref) was forgotten and must not come back."""
    return db.execute(_suppression_stmt(user_id, source_ref)).first() is not None


async def is_suppressed_async(db: AsyncSession, *, user_id, source_ref: str) -> bool:
    """Async face of :func:`is_suppressed`."""
    return (await db.execute(_suppression_stmt(user_id, source_ref))).first() is not None


async def is_content_suppressed_async(db: AsyncSession, *, user_id,
                                      content_hash: str | None) -> bool:
    """True when this user forgot a source with these BYTES (R38/T4).

    The re-upload guard's probe: a re-upload mints a NEW document id, so a
    ``source_ref`` match cannot see it — the ledger's ``content_hash`` (written
    when the source was forgotten) is the only key that survives the new id.
    ``None`` never matches: an unknown hash is not a forgotten identity.
    """
    if not content_hash:
        return False
    stmt = _ledger_stmt(MemorySuppression.id, user_id).where(
        MemorySuppression.content_hash == content_hash)
    return (await db.execute(stmt)).first() is not None


async def suppressed_refs_async(db: AsyncSession, *, user_id,
                                source_refs: Iterable[str]) -> set[str]:
    """Which of ``source_refs`` this user forgot — ONE batched ledger read.

    The import guard's probe (never one query per item): the same shape as the
    import path's dedup select, against the same ``(user_id, source_ref)`` key.
    The ledger's unique index answers it in one pass.
    """
    refs = [ref for ref in source_refs if ref]
    if not refs:
        return set()
    stmt = _ledger_stmt(MemorySuppression.source_ref, user_id).where(
        MemorySuppression.source_ref.in_(refs))
    return {row for row in (await db.execute(stmt)).scalars().all() if row}


def projection_content_hash(row) -> str | None:
    """The upload-time content hash a projection row carries, or ``None``.

    Read by the forget path to pin BYTES in the ledger, not only the doc id
    (R38). A row that predates the hash, or a non-document memory, reads None.
    """
    value = (getattr(row, "extra_metadata", None) or {}).get(CONTENT_HASH_KEY)
    return str(value) if value else None


def suppress_source(db: Session, *, user_id, source_ref: str, reason: str = "forgotten",
                    namespace: str | None = None, content_hash: str | None = None) -> None:
    """Record that this identity was forgotten (idempotent, caller commits).

    The unique ``(user_id, source_ref)`` keeps exactly one suppression row, so
    replaying the forget is a no-op instead of an integrity error. ``namespace``
    is the boundary the suppression was written in; ``content_hash`` is filled
    at UPLOAD time by the P4b/T4 guards — a caller that does not know one passes
    nothing, and the column stays NULL (R38: never backfilled).
    """
    if is_suppressed(db, user_id=user_id, source_ref=source_ref):
        return
    db.add(MemorySuppression(id=uuid.uuid4().hex, user_id=user_id,
                             source_ref=source_ref, reason=reason,
                             namespace=namespace, content_hash=content_hash))
    # Flush so a replay inside the same transaction sees the row (the session
    # does not autoflush) instead of racing the unique constraint.
    db.flush()


async def suppress_source_async(db: AsyncSession, *, user_id, source_ref: str,
                                reason: str = "forgotten", namespace: str | None = None,
                                content_hash: str | None = None) -> None:
    """Async face of :func:`suppress_source`."""
    if await is_suppressed_async(db, user_id=user_id, source_ref=source_ref):
        return
    db.add(MemorySuppression(id=uuid.uuid4().hex, user_id=user_id,
                             source_ref=source_ref, reason=reason,
                             namespace=namespace, content_hash=content_hash))
    await db.flush()


# ── projection queries: always scoped to (source_ref, owner) ────────────────


def _projection_rows(db, document_id: str, user_id) -> list[Memory]:
    """This document's projected rows *for one owner* — the tenant boundary."""
    return list(
        db.execute(
            select(Memory).where(
                Memory.source_ref == document_id,
                Memory.user_id == user_id,
            )
        ).scalars().all()
    )


def delete_document_memories_sync(db: Session, document_id: str, *, user_id) -> list[str]:
    """Delete one owner's memories derived from a document (sync).

    Deletes by ``(source_ref == document_id, user_id == user_id)`` so both the
    document-level row and its passages are removed regardless of hierarchy,
    while a foreign memory sharing the same ``source_ref`` survives. Caller
    commits.
    """
    rows = _projection_rows(db, document_id, user_id)
    ids = [str(m.id) for m in rows]
    for mem in rows:
        db.delete(mem)
    return ids


async def delete_document_memories_async(db: AsyncSession, document_id: str, *, user_id) -> list[str]:
    """Async variant of :func:`delete_document_memories_sync`.

    Used by the API delete paths (document delete, conversation delete). Deletes
    the rows in Postgres, enqueues one durable delete intent per row in the SAME
    transaction, and returns the ids so the caller can also purge the vector
    store after committing. Caller commits.
    """
    rows = (
        await db.execute(
            select(Memory).where(
                Memory.source_ref == document_id,
                Memory.user_id == user_id,
            )
        )
    ).scalars().all()
    ids = [str(m.id) for m in rows]
    for mem in rows:
        # Durable intent in the same transaction as the row delete: a crash
        # before the caller's vector purge is replayable by the drain.
        await enqueue_delete(db, entity_id=str(mem.id), tenant_id=str(user_id),
                             revision=int(mem.revision or 1))
        await db.delete(mem)
    return ids


def build_document_memories_sync(
    db: Session,
    document_id: str,
    parents: list[ParentChunk],
    *,
    user_id,
    content_hash: str | None = None,
) -> DocMemoryResult:
    """Create the document + passage memories for one ingested document.

    Idempotent: this owner's prior projection of the document is deleted first,
    so re-ingestion replaces rather than duplicates. Every removed row gets a
    durable delete intent and every new row a durable upsert intent, all in the
    caller's transaction — the caller commits the session, then embeds the
    returned ids into the vector store.

    A suppressed identity (user forgot this source) is skipped outright: no
    rows, no intent, nothing to re-ingest.

    ``content_hash`` is the sha256 of the UPLOADED BYTES, handed down by the
    ingest seam (R38/T4). It travels with the projection so a later forget can
    pin the bytes — not only the doc id — in the suppression ledger, which is
    what catches a re-upload (a new document id). ``None`` keeps the key out of
    the metadata: an unknown hash must not be invented.
    """
    if is_suppressed(db, user_id=user_id, source_ref=document_id):
        log.info("Doc→memory skipped: source suppressed",
                 extra={"doc_id": document_id, "user_id": str(user_id)})
        return DocMemoryResult(document_id=document_id, doc_memory_id=None, passage_memory_ids=[])

    doc = db.get(Document, document_id)
    if doc is None:
        log.warning("Doc→memory skipped: document not found", extra={"doc_id": document_id})
        return DocMemoryResult(document_id=document_id, doc_memory_id=None, passage_memory_ids=[])

    conversation = db.get(Conversation, doc.conversation_id)
    if conversation is None:
        log.warning("Doc→memory skipped: conversation not found", extra={"doc_id": document_id})
        return DocMemoryResult(document_id=document_id, doc_memory_id=None, passage_memory_ids=[])
    if conversation.user_id != user_id:
        # The projection is stored under the passed owner, so a mismatched
        # caller would write (and later erase) another tenant's memories.
        raise ValueError(
            f"document {document_id} belongs to a conversation owned by another user; "
            f"refusing to project memories for {user_id}"
        )

    # Idempotency: replace this owner's prior projection. A delete intent per
    # removed row rides the caller's transaction, so the vectors cannot be
    # stranded by a crash between the SQL commit and the index purge.
    removed_ids: list[str] = []
    for mem in _projection_rows(db, document_id, user_id):
        enqueue_delete_sync(db, entity_id=str(mem.id), tenant_id=str(user_id), revision=mem.revision)
        db.delete(mem)
        removed_ids.append(str(mem.id))

    base_meta = {
        "document_id": document_id,
        "conversation_id": str(doc.conversation_id),
        "filename": doc.filename,
    }
    if content_hash:
        # R38: the upload-time hash rides the projection, so forgetting this
        # source pins the BYTES and a re-upload of the same file is caught.
        base_meta[CONTENT_HASH_KEY] = content_hash

    doc_memory = Memory(
        id=uuid.uuid4(),
        user_id=user_id,
        source_type=DOC_MEMORY_SOURCE_TYPE,
        source_ref=document_id,
        title=doc.filename,
        content=_document_summary(parents) or doc.filename,
        summary=None,
        tags=[],
        extra_metadata={**base_meta, "kind": "document"},
    )
    db.add(doc_memory)
    bump_revision(doc_memory)  # revision 1, explicit before the INSERT
    db.flush()  # need doc_memory.id for passage parent_id

    passages: list[Memory] = []
    for parent in parents:
        passage = Memory(
            id=uuid.uuid4(),
            user_id=user_id,
            parent_id=doc_memory.id,
            source_type=DOC_MEMORY_SOURCE_TYPE,
            source_ref=document_id,
            title=doc.filename,
            content=parent.content,
            summary=None,
            tags=[],
            extra_metadata={
                **base_meta,
                "kind": "passage",
                "parent_chunk_id": parent.id,
                "parent_index": parent.index,
            },
        )
        db.add(passage)
        bump_revision(passage)
        passages.append(passage)

    db.flush()
    passage_ids = [str(p.id) for p in passages]
    for mem in (doc_memory, *passages):
        # Durable index intent in the SAME commit as the row (spec §5.1).
        enqueue_upsert_sync(db, mem)
    log.info(
        "Built document memories",
        extra={"doc_id": document_id, "passages": len(passage_ids)},
    )
    return DocMemoryResult(
        document_id=document_id,
        doc_memory_id=str(doc_memory.id),
        passage_memory_ids=passage_ids,
        removed_memory_ids=removed_ids,
    )


__all__ = [
    "CONTENT_HASH_KEY",
    "DocMemoryResult",
    "build_document_memories_sync",
    "delete_document_memories_sync",
    "delete_document_memories_async",
    "is_content_suppressed_async",
    "is_suppressed",
    "is_suppressed_async",
    "projection_content_hash",
    "suppress_source",
    "suppress_source_async",
    "suppressed_refs_async",
    "DOC_MEMORY_SOURCE_TYPE",
]
