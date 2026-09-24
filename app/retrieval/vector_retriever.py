"""Qdrant-backed store for document CHILD chunks (spec §4.2).

One PHYSICAL collection per generation — never one per conversation: the
per-conversation ``rag_conv_<id>`` collections are gone, and a conversation is
a payload filter inside the generation's collection instead. The tenant clause
is the security boundary and is built once, in
:mod:`app.retrieval.qdrant_filter`; nothing here hand-rolls a ``Filter``.

The SQL ``DocumentChunk`` row stays the source of truth. The payload is derived,
verifiable index data (see :func:`_chunk_payload` for the contract); a write
verifies the generation's contract first (the collection's dim/metric AND the
manifest row's fingerprint), exactly like the memory store.

An upsert/delete intent is enqueued in the SAME transaction as the SQL write
(:mod:`app.retrieval.memory.outbox`); the post-commit attempts here are the fast
path, never the record of truth.
"""
from __future__ import annotations

import logging
from typing import Any

from qdrant_client import models as qm

from app.models.document_chunk import DocumentChunk
from app.retrieval import vector_backend
from app.retrieval.embedder import (
    EmbeddingDimensionMismatch,
    check_generation_contract,
    embed_query,
    embed_texts,
    embed_texts_sync,
)
from app.retrieval.embedding_fingerprint import canonical_fingerprint, current_fingerprint
from app.retrieval.qdrant_filter import build_chunk_filter

log = logging.getLogger(__name__)

KIND_CHUNK = "chunk"


class VectorUnavailableError(Exception):
    """The vector backend itself is unreachable (not: empty collection, no matches).

    Raised instead of returning [] so callers can distinguish "vector search
    is down" (degrade to BM25-only + flag it) from "no vectors".
    """


def _current_dim() -> int:
    """The ACTIVE embedding dimension (384 local / configured API dimension)."""
    return int(current_fingerprint()["dim"])


# ── chunk <-> payload helpers ───────────────────────────────────────────────


def _chunk_metadata(chunk: DocumentChunk) -> dict[str, Any]:
    return dict(chunk.chunk_metadata or {})


def _chunk_payload(chunk: DocumentChunk, *, user_id: str) -> dict[str, Any]:
    """The payload written next to the vector — the chunk contract (§4.2).

    Identity and scope come from SQL (never from a caller-supplied id), the
    revision is the row's monotonic counter, and ``fingerprint`` is the
    embedding contract the vector was produced under. The embedded text rides
    the payload so a result can carry it; consumers still hydrate content from
    SQL. ``parent_id`` is omitted when absent (a payload never carries a null
    for a consumer to re-interpret).
    """
    metadata = _chunk_metadata(chunk)
    payload: dict[str, Any] = {
        "kind": KIND_CHUNK,
        "user_id": str(user_id),
        "conversation_id": _chunk_conversation_id(chunk, metadata),
        "document_id": str(chunk.document_id),
        "chunk_id": str(chunk.id),
        "revision": int(chunk.revision or 1),
        "fingerprint": canonical_fingerprint(current_fingerprint()),
        "child_index": int(metadata.get("child_index", chunk.chunk_index)),
        "content": chunk.content,
    }
    parent_id = metadata.get("parent_id")
    if parent_id:
        payload["parent_id"] = str(parent_id)
    return payload


def _chunk_conversation_id(chunk: DocumentChunk, metadata: dict[str, Any]) -> str:
    """The conversation the point is scoped to: the metadata, else the row.

    ``""`` is not a valid scope — a point written with an empty
    ``conversation_id`` is unreachable by every conversation-scoped filter and
    delete, i.e. an orphan no erasure can name. Fall back to the document row;
    if that is absent too, refuse loudly instead of indexing an unowned point.
    """
    conversation_id = metadata.get("conversation_id")
    if not conversation_id:
        conversation_id = getattr(chunk.document, "conversation_id", None)
    if not conversation_id:
        raise ValueError(
            f"chunk {chunk.id} has no conversation_id (neither chunk_metadata "
            "nor its document row): refusing to index an unowned point"
        )
    return str(conversation_id)


def _point(chunk: DocumentChunk, vector: list[float], *, user_id: str) -> qm.PointStruct:
    """One point: the chunk's UUID as the point id (Qdrant's own id type)."""
    return qm.PointStruct(
        id=str(chunk.id), vector=vector, payload=_chunk_payload(chunk, user_id=user_id)
    )


# ── generation + contract (the manifest is the only pointer) ────────────────


def _generation_sync() -> tuple[str, str | None]:
    """(generation, manifest fingerprint) for kind=chunk, sync face."""
    from app.retrieval.memory import outbox  # local: outbox imports this module

    return outbox.active_generation_sync(kind=outbox.KIND_CHUNK)


async def _generation() -> tuple[str, str | None]:
    """(generation, manifest fingerprint) for kind=chunk, async face."""
    from app.retrieval.memory import outbox

    return await outbox.active_generation(kind=outbox.KIND_CHUNK)


def _open_collection_sync(embedding_dim: int) -> tuple[Any, str, str | None]:
    """Sync face: client + the active generation, created if missing."""
    client = vector_backend.get_sync_client()
    generation, manifest_fingerprint = _generation_sync()
    vector_backend.ensure_collection(KIND_CHUNK, generation, embedding_dim)
    return client, generation, manifest_fingerprint


async def _open_collection(embedding_dim: int) -> tuple[Any, str, str | None]:
    """Async face: the same three facts, without blocking the event loop."""
    client = vector_backend.get_async_client()
    generation, manifest_fingerprint = await _generation()
    await vector_backend.ensure_collection_async(KIND_CHUNK, generation, embedding_dim)
    return client, generation, manifest_fingerprint


def _checked_collection_sync(embedding_dim: int) -> tuple[Any, str, int]:
    """Sync face plus the contract guard — verified before anything is written."""
    client, generation, manifest_fingerprint = _open_collection_sync(embedding_dim)
    count = int(client.count(generation).count)
    check_generation_contract(
        vector_backend.collection_info(KIND_CHUNK, generation),
        embedding_dim,
        manifest_fingerprint=manifest_fingerprint,
        collection_is_empty=count == 0,
    )
    return client, generation, count


async def _checked_collection(embedding_dim: int) -> tuple[Any, str, int]:
    """Async face plus the contract guard."""
    client, generation, manifest_fingerprint = await _open_collection(embedding_dim)
    count = int((await client.count(generation)).count)
    check_generation_contract(
        await vector_backend.collection_info_async(KIND_CHUNK, generation),
        embedding_dim,
        manifest_fingerprint=manifest_fingerprint,
        collection_is_empty=count == 0,
    )
    return client, generation, count


# ── public API: writes ──────────────────────────────────────────────────────


async def upsert_chunks(chunks: list[DocumentChunk], *, user_id: str) -> int:
    """Embed child chunks and write them to the active generation. Returns count.

    ``user_id`` is the owner resolved from SQL by the caller (never a
    caller-supplied tenant). A contract mismatch propagates as
    :class:`EmbeddingDimensionMismatch` — indexing into a generation whose
    contract cannot be verified is an integrity failure, not a retry.
    """
    if not chunks:
        return 0
    vectors = await embed_texts([chunk.content for chunk in chunks])
    client, generation, _ = await _checked_collection(len(vectors[0]))
    await client.upsert(
        collection_name=generation,
        points=[
            _point(chunk, vectors[index], user_id=user_id)
            for index, chunk in enumerate(chunks)
        ],
    )
    log.info("Upserted chunks into Qdrant", extra={"n": len(chunks), "user_id": str(user_id)})
    return len(chunks)


def upsert_chunks_sync(chunks: list[DocumentChunk], *, user_id: str) -> int:
    """Synchronous variant of :func:`upsert_chunks` (ingestion / CLI face)."""
    if not chunks:
        return 0
    vectors = embed_texts_sync([chunk.content for chunk in chunks])
    client, generation, _ = _checked_collection_sync(len(vectors[0]))
    client.upsert(
        collection_name=generation,
        points=[
            _point(chunk, vectors[index], user_id=user_id)
            for index, chunk in enumerate(chunks)
        ],
    )
    log.info("Upserted chunks into Qdrant (sync)", extra={"n": len(chunks)})
    return len(chunks)


# ── public API: deletes ─────────────────────────────────────────────────────


async def delete_chunks(chunk_ids: list[str]) -> bool:
    """Remove points by chunk id; ``True`` only when absence was read back.

    Deliberately UNGUARDED by the contract check: erasure must still work when
    a generation's contract is stale, and the durable outbox acks a delete
    intent ``done`` from this result — ``False`` keeps it pending and retried.
    """
    if not chunk_ids:
        return True
    try:
        client, generation, _ = await _open_collection(_current_dim())
        await client.delete(
            collection_name=generation,
            points_selector=qm.PointIdsList(points=list(chunk_ids)),
        )
        survivors = await client.retrieve(
            collection_name=generation, ids=list(chunk_ids), with_payload=False
        )
        if survivors:
            log.warning("Chunk delete not confirmed", extra={"still_present": len(survivors)})
            return False
        log.info("Deleted chunks from Qdrant", extra={"n": len(chunk_ids)})
        return True
    except Exception as e:
        log.warning(
            "Failed to delete chunks from Qdrant",
            extra={"n": len(chunk_ids), "error": str(e)},
        )
        return False


async def _delete_filtered(chunk_filter: qm.Filter, *, what: str) -> bool:
    """Server-side filtered delete + count readback (no get-then-delete)."""
    try:
        client, generation, _ = await _open_collection(_current_dim())
        await client.delete(collection_name=generation, points_selector=chunk_filter)
        remaining = await client.count(collection_name=generation, count_filter=chunk_filter)
        if int(remaining.count):
            log.warning("Filtered chunk delete not confirmed", extra={"what": what})
            return False
        log.info("Deleted chunks from Qdrant", extra={"scope": what})
        return True
    except Exception as e:
        log.warning(
            "Failed to delete chunks by filter from Qdrant",
            extra={"scope": what, "error": str(e)},
        )
        return False


def _delete_filtered_sync(chunk_filter: qm.Filter, *, what: str) -> bool:
    """Synchronous twin of :func:`_delete_filtered` (ingestion / CLI face)."""
    try:
        client, generation, _ = _open_collection_sync(_current_dim())
        client.delete(collection_name=generation, points_selector=chunk_filter)
        remaining = client.count(collection_name=generation, count_filter=chunk_filter)
        if int(remaining.count):
            log.warning("Filtered chunk delete not confirmed", extra={"what": what})
            return False
        return True
    except Exception as e:
        log.warning(
            "Failed to delete chunks by filter from Qdrant (sync)",
            extra={"scope": what, "error": str(e)},
        )
        return False


async def delete_document_chunks(
    conversation_id: str, document_id: str, *, user_id: str
) -> bool:
    """Delete one document's points with a server-side filtered delete.

    A TRUE document delete only — every point of the document belongs to rows
    being deleted with it. A REINGEST must purge by id instead
    (:func:`delete_chunks_by_ids`): a document-wide sweep can take a point a
    concurrent drain has just written and acked for a still-live row.
    """
    return await _delete_filtered(
        build_chunk_filter(user_id, conversation_id, document_id=document_id),
        what=f"document:{document_id}",
    )


def delete_document_chunks_sync(
    conversation_id: str, document_id: str, *, user_id: str
) -> bool:
    """Synchronous variant of :func:`delete_document_chunks` — a true delete only."""
    return _delete_filtered_sync(
        build_chunk_filter(user_id, conversation_id, document_id=document_id),
        what=f"document:{document_id}",
    )


def delete_chunks_by_ids(chunk_ids: list[str], *, user_id: str, conversation_id: str) -> int:
    """Delete exactly the NAMED ids, inside the caller's own scope (ruling R8/R9).

    The reingest purge. Id-scoped by construction: only the ids whose rows left
    SQL are named, so this can never sweep away a point a concurrent drain has
    just written and acked for a row that is STILL live (that ack is final —
    no pending intent is left to replay the point). The selector is the id set
    ANDed into the tenant-scoped chunk filter, so naming a foreign point's id
    reaches nothing.

    Returns how many of the named ids this call confirmed deleted; a partial or
    unconfirmed delete is logged and reported as a short count (``0`` when the
    store itself failed) — the caller's durable delete intents stay pending and
    the drain owns the retry. The readback counts by ID ALONE: the tenant clause
    protects the DELETE, and an id outside the caller's scope must report
    honestly as not deleted rather than count as a deletion this call made.
    """
    ids = [str(chunk_id) for chunk_id in chunk_ids]
    if not ids:
        return 0
    try:
        client, generation, _ = _open_collection_sync(_current_dim())
        client.delete(
            collection_name=generation,
            points_selector=qm.Filter(
                must=[
                    *(build_chunk_filter(user_id, conversation_id).must or []),
                    qm.HasIdCondition(has_id=ids),
                ]
            ),
        )
        survivors = client.count(
            collection_name=generation,
            count_filter=qm.Filter(must=[qm.HasIdCondition(has_id=ids)]),
        )
        deleted = len(ids) - int(survivors.count)
        if deleted < len(ids):
            log.warning(
                "Chunk delete by id not confirmed",
                extra={"n": len(ids), "survivors": int(survivors.count)},
            )
        return deleted
    except Exception as e:
        log.warning(
            "Failed to delete chunks by id from Qdrant (sync)",
            extra={"n": len(ids), "error": str(e)},
        )
        return 0


async def delete_conversation_chunks(conversation_id: str, *, user_id: str) -> bool:
    """Delete a whole conversation's points (the session/document cascade)."""
    return await _delete_filtered(
        build_chunk_filter(user_id, conversation_id), what=f"conversation:{conversation_id}"
    )


# ── public API: search ──────────────────────────────────────────────────────


async def search(
    query: str,
    top_k: int,
    conversation_id: str,
    hyde_text: str | None = None,
    *,
    user_id: str,
) -> list[dict]:
    """Semantic search over the active chunk generation.

    Returns the pinned shape, best first:
        ``{content, score, source, rank, metadata, child_id, parent_id}``

    ``score`` is the cosine SIMILARITY the store reports (never ``1 - dist``),
    and the list is sorted by ``(-score, child_id)`` so equal scores have a
    stable order. Scope is the conversation — plus the tenant clause when
    the REQUIRED ``user_id`` (conversation scope alone is not a tenant
    boundary, so there is no unscoped call form).

    Raises :class:`VectorUnavailableError` when the generation cannot be
    acquired or the count/query calls themselves fail — a vector outage is a
    readiness signal, never an empty list — and
    :class:`EmbeddingDimensionMismatch` on a contract mismatch. Only a
    genuinely empty generation (or a query matching nothing) yields ``[]``.
    """
    chunk_filter = build_chunk_filter(user_id, conversation_id)
    try:
        # The read path opens the generation at the CONTRACT dim (the active
        # fingerprint), never the query's.
        client, generation, manifest_fingerprint = await _open_collection(_current_dim())
    except Exception as e:
        log.warning("Qdrant unavailable for chunk search", extra={"error": str(e)})
        raise VectorUnavailableError(f"Qdrant unreachable for chunk search: {e}") from e

    try:
        count = int((await client.count(generation)).count)
    except EmbeddingDimensionMismatch:
        raise
    except Exception as e:
        log.warning("Qdrant count failed for chunk search", extra={"error": str(e)})
        raise VectorUnavailableError(f"Qdrant chunk search failed at count: {e}") from e

    # Fail loud on backend/dim switches; never stamp/write the manifest here
    # (read path). An empty generation has nothing to verify and no points.
    if count == 0 or top_k <= 0:
        return []

    embed_input = hyde_text if hyde_text else query
    embedding = await embed_query(embed_input)
    try:
        check_generation_contract(
            await vector_backend.collection_info_async(KIND_CHUNK, generation),
            len(embedding),
            manifest_fingerprint=manifest_fingerprint,
            collection_is_empty=False,
        )
    except EmbeddingDimensionMismatch:
        raise
    except Exception as e:
        # Reading the contract needs the store too: an unreadable contract is
        # an outage, while a READABLE mismatch stays the integrity error above.
        log.warning("Qdrant contract check failed for chunk search", extra={"error": str(e)})
        raise VectorUnavailableError(
            f"Qdrant chunk search failed at contract check: {e}"
        ) from e

    try:
        response = await client.query_points(
            collection_name=generation,
            query=list(embedding),
            query_filter=chunk_filter,
            limit=min(top_k, count),
        )
    except EmbeddingDimensionMismatch:
        raise
    except Exception as e:
        log.warning("Qdrant query failed for chunk search", extra={"error": str(e)})
        raise VectorUnavailableError(f"Qdrant chunk search failed at query: {e}") from e

    items: list[dict] = []
    for point in response.points:
        payload = dict(point.payload or {})
        content = payload.pop("content", None)
        items.append(
            {
                "content": content,
                "score": float(point.score),
                "source": "vector",
                "rank": 0,
                "metadata": payload,
                "child_id": str(payload.get("chunk_id") or ""),
                "parent_id": str(payload.get("parent_id") or ""),
            }
        )
    items.sort(key=lambda item: (-item["score"], item["child_id"]))
    for rank, item in enumerate(items):
        item["rank"] = rank
    return items


__all__ = [
    "KIND_CHUNK",
    "VectorUnavailableError",
    "delete_chunks",
    "delete_chunks_by_ids",
    "delete_conversation_chunks",
    "delete_document_chunks",
    "delete_document_chunks_sync",
    "search",
    "upsert_chunks",
    "upsert_chunks_sync",
]
