"""Qdrant-backed vector store for the personal Memory collection (spec §4.2).

One PHYSICAL collection per generation, named by the ``index_generations``
manifest — the cutover is a pointer flip, never a data copy — with a
``user_id`` payload filter inside it: that filter is the security boundary, so
the tenant clause is built once, in :mod:`app.retrieval.qdrant_filter`, and a
caller's ``where`` can never widen it.

Every operation verifies the generation's contract before it touches data
(:func:`app.retrieval.embedder.check_generation_contract`): the collection's
own dim/metric AND the manifest row's fingerprint. A populated generation with
no manifest row is quarantined — served as neither results nor a promise.

The SQL ``Memory`` row stays the source of truth. The vector payload is
derived, verifiable index data (see ``_memory_to_metadata`` for the contract);
consumers hydrate content from SQL rather than trusting the vector copy
(:mod:`app.retrieval.memory.retriever`).
"""
from __future__ import annotations

import logging
from typing import Any

from qdrant_client import models as qm

from app.models.memory import Memory
from app.retrieval import vector_backend
from app.retrieval.embedder import (
    EmbeddingDimensionMismatch,
    active_backend_name,
    check_generation_contract,
    embed_texts,
    embed_texts_sync,
)
from app.retrieval.embedding_fingerprint import (
    canonical_fingerprint,
    current_fingerprint,
    fingerprint_generation,
)
from app.retrieval.qdrant_filter import build_filter
from app.retrieval.vector_retriever import VectorUnavailableError

log = logging.getLogger(__name__)

# Transitional generation name: the collection this install served before the
# P1b cutover seeded a manifest row. It stays the fallback inside
# ``outbox.active_generation`` (a P1a test pins the constant), so nothing here
# re-derives a name.
COLLECTION_NAME = "Orivory_memories"

KIND_MEMORY = "memory"


# ── memory <-> payload helpers ──────────────────────────────────────────────


def _memory_to_document(memory: Memory) -> str:
    """The text that gets embedded: title (when present) prepended to content.

    Title is prepended so the embedding captures the topic; the same text is
    kept in the payload as ``content`` so a result can carry it without a
    second source of truth.
    """
    parts: list[str] = []
    if memory.title:
        parts.append(f"Title: {memory.title}")
    parts.append(memory.content)
    return "\n".join(parts)


def _memory_to_metadata(memory: Memory, *, embedding_dim: int | None = None) -> dict[str, Any]:
    """The payload written next to the vector — the memory contract (§4.2).

    Provenance keys keep the spelling the P0/P1a corpus already persisted
    (``orivory_embed_fingerprint`` / ``orivory_embed_generation`` /
    ``orivory_memory_revision``): the brief's short ``fingerprint`` /
    ``revision`` names ARE those keys, and a second alias would be a second
    source of truth for one value.

    An absent value is OMITTED, never null — Qdrant rejects null payload
    values, and an absent key is exactly "unknown" to a filter. The user's
    ``extra_metadata`` is deliberately not copied into the index payload.
    """
    # Local import: correction -> outbox -> this module is a real cycle.
    from app.retrieval.memory.correction import state_of

    fingerprint = current_fingerprint()
    canonical = canonical_fingerprint(fingerprint)
    metadata: dict[str, Any] = {
        "kind": KIND_MEMORY,
        "user_id": str(memory.user_id),
        "memory_id": str(memory.id),
        # Derived from the ONE lifecycle authority (correction.state_of, the
        # Python rule app.retrieval.memory.visibility mirrors in SQL): a
        # dirty/superseded row must be identifiable from the payload itself.
        "visibility_state": state_of(memory),
        "source_type": memory.source_type,
        "salience": float(memory.salience),
        "pinned": bool(memory.pinned),
        "orivory_embed_backend": active_backend_name(),
        "orivory_embed_dim": int(
            embedding_dim if embedding_dim is not None else fingerprint["dim"]
        ),
        "orivory_embed_fingerprint": canonical,
        "orivory_embed_generation": fingerprint_generation(canonical),
    }
    if memory.captured_at is not None:
        metadata["captured_at"] = memory.captured_at.isoformat()
    model_revision = fingerprint.get("model_revision") or fingerprint.get("revision")
    metadata["orivory_embed_model_revision"] = (
        model_revision if isinstance(model_revision, str) and model_revision else "unavailable"
    )
    memory_revision = getattr(memory, "revision", None)
    if isinstance(memory_revision, (int, float, str)) and not isinstance(memory_revision, bool):
        metadata["orivory_memory_revision"] = memory_revision
    else:
        # Memory has no canonical revision column yet; make that limitation
        # visible to backfill/audit tooling instead of guessing from a clock.
        metadata["orivory_memory_revision"] = "unavailable"
    if memory.tags:
        metadata["tags"] = list(memory.tags)
    return metadata


def _memory_payload(memory: Memory, document: str, embedding_dim: int) -> dict[str, Any]:
    """The point payload: contract metadata + the embedded document text."""
    return {**_memory_to_metadata(memory, embedding_dim=embedding_dim), "content": document}


def _point(memory: Memory, vector: list[float], document: str) -> qm.PointStruct:
    """One point: the memory's UUID as the point id (Qdrant's own id type)."""
    return qm.PointStruct(
        id=str(memory.id),
        vector=vector,
        payload=_memory_payload(memory, document, len(vector)),
    )


# ── generation + contract (the manifest is the only pointer) ────────────────


def _generation_sync() -> tuple[str, str | None]:
    """(generation, manifest fingerprint) — P1a's manifest, sync face."""
    from app.retrieval.memory import outbox  # local: outbox imports this module

    return outbox.active_generation_sync()


async def _generation() -> tuple[str, str | None]:
    """(generation, manifest fingerprint) — P1a's manifest, async face."""
    from app.retrieval.memory import outbox

    return await outbox.active_generation()


def _open_collection_sync(embedding_dim: int) -> tuple[Any, str, str | None]:
    """Sync face: client + the active generation, created if missing."""
    client = vector_backend.get_sync_client()
    generation, manifest_fingerprint = _generation_sync()
    vector_backend.ensure_collection(KIND_MEMORY, generation, embedding_dim)
    return client, generation, manifest_fingerprint


async def _open_collection(embedding_dim: int) -> tuple[Any, str, str | None]:
    """Async face: the same three facts, without blocking the event loop."""
    client = vector_backend.get_async_client()
    generation, manifest_fingerprint = await _generation()
    await vector_backend.ensure_collection_async(KIND_MEMORY, generation, embedding_dim)
    return client, generation, manifest_fingerprint


def _checked_collection_sync(embedding_dim: int) -> tuple[Any, str, int]:
    """Sync face plus the contract guard — verified before anything is written."""
    client, generation, manifest_fingerprint = _open_collection_sync(embedding_dim)
    count = int(client.count(generation).count)
    check_generation_contract(
        vector_backend.collection_info(KIND_MEMORY, generation),
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
        await vector_backend.collection_info_async(KIND_MEMORY, generation),
        embedding_dim,
        manifest_fingerprint=manifest_fingerprint,
        collection_is_empty=count == 0,
    )
    return client, generation, count


# ── public API ──────────────────────────────────────────────────────────────


async def upsert_memory(memory: Memory) -> None:
    """Embed a memory and write it to the active generation.

    Best-effort: logs and re-raises. Callers should wrap in try/except so a
    vector outage doesn't fail a CRUD request — the SQL ``Memory`` row is the
    source of truth. A contract mismatch is NOT absorbed: it propagates as
    :class:`EmbeddingDimensionMismatch`, because indexing into a generation
    whose contract cannot be verified is an integrity failure.
    """
    document = _memory_to_document(memory)
    embedding = (await embed_texts([document]))[0]
    client, generation, _ = await _checked_collection(len(embedding))
    await client.upsert(
        collection_name=generation, points=[_point(memory, embedding, document)]
    )
    log.info(
        "Upserted memory into Qdrant",
        extra={"memory_id": str(memory.id), "user_id": str(memory.user_id)},
    )


def upsert_memory_sync(memory: Memory) -> None:
    """Synchronous variant — used by Celery / CLI contexts."""
    document = _memory_to_document(memory)
    embedding = embed_texts_sync([document])[0]
    client, generation, _ = _checked_collection_sync(len(embedding))
    client.upsert(collection_name=generation, points=[_point(memory, embedding, document)])
    log.info(
        "Upserted memory into Qdrant (sync)",
        extra={"memory_id": str(memory.id), "user_id": str(memory.user_id)},
    )


def upsert_memories_sync(memories: list[Memory]) -> int:
    """Batch-embed and upsert many memories (sync). Returns count written.

    Embeds all documents in one batched pass (respecting ``EMBED_BATCH_SIZE``
    inside ``embed_texts_sync``) and writes them in a single upsert. Used by
    the reindex/backfill task.
    """
    if not memories:
        return 0
    documents = [_memory_to_document(memory) for memory in memories]
    embeddings = embed_texts_sync(documents)
    client, generation, _ = _checked_collection_sync(len(embeddings[0]))
    client.upsert(
        collection_name=generation,
        points=[
            _point(memory, embeddings[index], documents[index])
            for index, memory in enumerate(memories)
        ],
    )
    return len(memories)


def get_existing_memory_ids_sync(memory_ids: list[str]) -> set[str]:
    """Return the subset of ``memory_ids`` already present in the generation.

    Used by the reindex task to compute which memories are missing their
    vector without re-embedding everything. Presence is only trusted when the
    generation contract matches the active embedding contract.
    """
    if not memory_ids:
        return set()
    client, generation, _ = _checked_collection_sync(int(current_fingerprint()["dim"]))
    records = client.retrieve(
        collection_name=generation, ids=list(memory_ids), with_payload=False
    )
    return {str(record.id) for record in records}


async def get_memory_ids_present(memory_ids: list[str]) -> set[str]:
    """Return the subset of ``memory_ids`` that still exist in the generation.

    Verification seam for erasure receipts: after deletion the erasure service
    re-queries the store through this helper and records any ids still present
    as residuals. Async counterpart of :func:`get_existing_memory_ids_sync`.
    """
    if not memory_ids:
        return set()
    client, generation, _ = await _checked_collection(int(current_fingerprint()["dim"]))
    records = await client.retrieve(
        collection_name=generation, ids=list(memory_ids), with_payload=False
    )
    return {str(record.id) for record in records}


async def delete_memory(memory_id: str) -> bool:
    """Remove a memory's vector from the active generation.

    Deliberately UNGUARDED by the contract check: erasure must still work when
    a generation's contract is stale, and the durable outbox acks a delete
    intent ``done`` from this result — ``False`` keeps it pending and retried.
    """
    try:
        client, generation, _ = await _open_collection(int(current_fingerprint()["dim"]))
        await client.delete(
            collection_name=generation, points_selector=qm.PointIdsList(points=[memory_id])
        )
        log.info("Deleted memory from Qdrant", extra={"memory_id": memory_id})
        return True
    except Exception as e:
        log.warning(
            "Failed to delete memory from Qdrant",
            extra={"memory_id": memory_id, "error": str(e)},
        )
        return False


async def delete_memories(memory_ids: list[str]) -> bool:
    """Remove many memories' vectors (see :func:`delete_memory`)."""
    if not memory_ids:
        return True
    try:
        client, generation, _ = await _open_collection(int(current_fingerprint()["dim"]))
        await client.delete(
            collection_name=generation,
            points_selector=qm.PointIdsList(points=list(memory_ids)),
        )
        log.info("Deleted memories from Qdrant", extra={"n": len(memory_ids)})
        return True
    except Exception as e:
        log.warning(
            "Failed to batch-delete memories from Qdrant",
            extra={"n": len(memory_ids), "error": str(e)},
        )
        return False


def delete_memories_sync(memory_ids: list[str]) -> None:
    """Synchronous batch delete — used by Celery ingestion (best-effort)."""
    if not memory_ids:
        return
    try:
        client, generation, _ = _open_collection_sync(int(current_fingerprint()["dim"]))
        client.delete(
            collection_name=generation,
            points_selector=qm.PointIdsList(points=list(memory_ids)),
        )
        log.info("Deleted memories from Qdrant (sync)", extra={"n": len(memory_ids)})
    except Exception as e:
        log.warning(
            "Failed to batch-delete memories from Qdrant (sync)",
            extra={"n": len(memory_ids), "error": str(e)},
        )


async def search_memories(
    query_embedding: list[float],
    *,
    user_id: str,
    top_k: int = 10,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Vector search restricted to a single user.

    Returns a list of dicts, best first:
        {memory_id, content, score, metadata, rank, source="vector"}

    ``score`` is the cosine SIMILARITY the store reports (never ``1 - dist``),
    and the list is sorted by ``(-score, memory_id)`` so equal scores have a
    stable, reproducible order. The tenant clause is applied by the store's
    own filter builder; ``where`` can only narrow it.

    Raises :class:`VectorUnavailableError` when the generation cannot be
    acquired, or when the ``count``/``query`` calls themselves fail — a vector
    outage is a readiness signal, never an empty list — and
    :class:`EmbeddingDimensionMismatch` on a contract mismatch. Only a
    genuinely empty generation (or a query matching nothing) yields ``[]``.
    """
    user_filter = build_filter(user_id, where)
    try:
        client, generation, manifest_fingerprint = await _open_collection(
            len(query_embedding)
        )
    except Exception as e:
        # An outage is a typed readiness signal, never an empty result: a
        # silent [] here is a false "no memories matched".
        log.warning("Qdrant unavailable for search", extra={"error": str(e)})
        raise VectorUnavailableError(f"Qdrant unreachable for memory search: {e}") from e

    try:
        count = int((await client.count(generation)).count)
    except EmbeddingDimensionMismatch:
        # A contract mismatch is never re-typed as an outage.
        raise
    except Exception as e:
        # The store can also die after acquisition; type the failure at the
        # step it happened so it is never read as an empty (no-match) result.
        log.warning("Qdrant count failed for memory search", extra={"error": str(e)})
        raise VectorUnavailableError(
            f"Qdrant memory search failed at count: {e}"
        ) from e

    # Fail loud on backend/dim switches; never stamp/write the manifest here
    # (read path). The count must be known first so an unclaimed populated
    # generation cannot be mistaken for a new empty collection.
    try:
        check_generation_contract(
            await vector_backend.collection_info_async(KIND_MEMORY, generation),
            len(query_embedding),
            manifest_fingerprint=manifest_fingerprint,
            collection_is_empty=count == 0,
        )
    except EmbeddingDimensionMismatch:
        raise
    except Exception as e:
        # Reading the contract needs the store too: an unreadable contract is
        # an outage, while a READABLE mismatch stays the integrity error above.
        log.warning("Qdrant contract check failed for memory search", extra={"error": str(e)})
        raise VectorUnavailableError(
            f"Qdrant memory search failed at contract check: {e}"
        ) from e
    if count == 0 or top_k <= 0:
        return []

    try:
        response = await client.query_points(
            collection_name=generation,
            query=list(query_embedding),
            query_filter=user_filter,
            limit=min(top_k, count),
        )
    except EmbeddingDimensionMismatch:
        raise
    except Exception as e:
        log.warning("Qdrant query failed for memory search", extra={"error": str(e)})
        raise VectorUnavailableError(
            f"Qdrant memory search failed at query: {e}"
        ) from e

    items: list[dict[str, Any]] = []
    for point in response.points:
        payload = dict(point.payload or {})
        items.append(
            {
                "memory_id": str(payload.get("memory_id") or point.id),
                "content": payload.pop("content", None),
                "score": float(point.score),
                "metadata": payload,
                "rank": 0,
                "source": "vector",
            }
        )
    items.sort(key=lambda item: (-item["score"], item["memory_id"]))
    for rank, item in enumerate(items):
        item["rank"] = rank
    return items
