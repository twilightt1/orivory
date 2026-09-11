"""
Phase 3 — ChromaDB vector store for the personal Memory collection.

A single shared collection ``Orivory_memories`` is partitioned by
``user_id`` via ChromaDB ``where={...}`` filter. We keep one collection
(not one per user) to avoid expensive create/destroy churn; the
``user_id`` filter is the security boundary.

Mirrors the structure of :mod:`app.retrieval.vector_retriever` but
operates on the ``Memory`` table instead of document chunks.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import httpx

from app.config import settings
from app.models.memory import Memory
from app.retrieval.embedder import (
    astamp_collection_dim,
    check_collection_dim,
    embed_texts,
    embed_texts_sync,
    stamp_collection_dim,
)

# Lazily imported so this module is importable in test/CLI contexts
# that don't have ChromaDB running.
if TYPE_CHECKING:
    import chromadb

log = logging.getLogger(__name__)

COLLECTION_NAME = "Orivory_memories"

_async_client: chromadb.AsyncHttpClient | None = None
_sync_client: chromadb.HttpClient | None = None


# ── retry decorator (mirrors app.retrieval.vector_retriever.with_retry) ─────


def _with_retry(retries: int = 3, base_delay: float = 1.0):
    """Decorator that retries ChromaDB calls on connection errors."""

    def decorator(func: Callable):
        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                last_exc: Exception | None = None
                for i in range(retries):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        last_exc = e
                        if any(
                            msg in str(e)
                            for msg in ["Could not connect", "connection", "Refused"]
                        ) or isinstance(e, (ValueError, httpx.ConnectError)):
                            delay = base_delay * (2 ** i)
                            log.warning(
                                "Chroma connection failed (attempt %d/%d). "
                                "Retrying in %.1fs...",
                                i + 1, retries, delay,
                                extra={"error": str(e)},
                            )
                            await asyncio.sleep(delay)
                        else:
                            raise
                log.error(
                    "Failed to connect to Chroma after all retries.",
                    extra={"error": str(last_exc)},
                )
                raise last_exc  # type: ignore[misc]

            return async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                last_exc: Exception | None = None
                for i in range(retries):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exc = e
                        if any(
                            msg in str(e)
                            for msg in ["Could not connect", "connection", "Refused"]
                        ) or isinstance(e, (ValueError, httpx.ConnectError)):
                            delay = base_delay * (2 ** i)
                            log.warning(
                                "Chroma connection failed (attempt %d/%d). "
                                "Retrying in %.1fs...",
                                i + 1, retries, delay,
                                extra={"error": str(e)},
                            )
                            time.sleep(delay)
                        else:
                            raise
                log.error(
                    "Failed to connect to Chroma after all retries.",
                    extra={"error": str(last_exc)},
                )
                raise last_exc  # type: ignore[misc]

            return sync_wrapper

    return decorator


# ── client singletons ────────────────────────────────────────────────────────


@_with_retry()
async def _get_async_client():
    """Async client for HTTP mode; in local (lite) mode the sync
    PersistentClient is wrapped to satisfy await-less call sites — lite is
    single-user so the blocking call is acceptable."""
    global _async_client
    if _async_client is None:
        import chromadb  # lazy: only needed at runtime
        if settings.CHROMA_MODE == "local":
            client = _get_sync_client()

            class _SyncAsAsync:
                def __init__(self, inner):
                    self._inner = inner

                def __getattr__(self, name):
                    attr = getattr(self._inner, name)

                    async def _call(*args, **kwargs):
                        return attr(*args, **kwargs)

                    return _call

            _async_client = _SyncAsAsync(client)
        else:
            _async_client = await chromadb.AsyncHttpClient(
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT,
            )
    return _async_client


@_with_retry()
def _get_sync_client():
    global _sync_client
    if _sync_client is None:
        import chromadb  # lazy: only needed at runtime
        if settings.CHROMA_MODE == "local":
            _sync_client = chromadb.PersistentClient(path=settings.CHROMA_LOCAL_PATH)
        else:
            _sync_client = chromadb.HttpClient(
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT,
            )
    return _sync_client


@_with_retry()
async def _get_collection():
    cli = await _get_async_client()
    collection = await cli.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    # Local (lite) mode: the client wrapper only covers client-level calls —
    # the collection itself is the raw sync object, and call sites await
    # collection.count()/query()/upsert(). Route the collection through the
    # same sync→async adapter so every call site works in both modes.
    if settings.CHROMA_MODE == "local":
        sync_inner = getattr(collection, "_inner", None)
        target = sync_inner if sync_inner is not None else collection

        class _SyncAsAsync:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                attr = getattr(self._inner, name)

                async def _call(*args, **kwargs):
                    return attr(*args, **kwargs)

                return _call

        return _SyncAsAsync(target)
    return collection


# ── memory <-> document helpers ─────────────────────────────────────────────


def _memory_to_document(memory: Memory) -> str:
    """Concatenate title + content for the ChromaDB document text.

    Title is prepended (when present) so embedding captures the topic;
    content carries the body.
    """
    parts: list[str] = []
    if memory.title:
        parts.append(f"Title: {memory.title}")
    parts.append(memory.content)
    return "\n".join(parts)


def _memory_to_metadata(memory: Memory) -> dict[str, Any]:
    """Build the metadata dict stored alongside the vector.

    All values must be scalar or list-of-str (ChromaDB constraint).
    """
    captured_iso = memory.captured_at.isoformat() if memory.captured_at else None
    metadata: dict[str, Any] = {
        "user_id":     str(memory.user_id),
        "memory_id":   str(memory.id),
        "source_type": memory.source_type,
        "captured_at": captured_iso,
        "salience":    float(memory.salience),
        "pinned":      bool(memory.pinned),
    }
    # ChromaDB rejects empty-list metadata values ("Expected metadata list
    # value ... to be non-empty"), which silently broke the doc→memory
    # projection for every untagged memory. Only store `tags` when non-empty.
    if memory.tags:
        metadata["tags"] = list(memory.tags)
    return metadata


# ── public API ──────────────────────────────────────────────────────────────


async def upsert_memory(memory: Memory) -> None:
    """Embed a memory and write it to the ChromaDB collection.

    Best-effort: logs and re-raises. Callers should wrap in try/except
    so a ChromaDB outage doesn't fail a CRUD request — the Postgres
    ``Memory`` row is the source of truth.
    """
    collection = await _get_collection()
    document = _memory_to_document(memory)
    metadata = _memory_to_metadata(memory)
    embedding = (await embed_texts([document]))[0]
    await astamp_collection_dim(collection, len(embedding))
    await collection.upsert(
        ids=[str(memory.id)],
        documents=[document],
        embeddings=[embedding],
        metadatas=[metadata],
    )
    log.info(
        "Upserted memory into ChromaDB",
        extra={"memory_id": str(memory.id), "user_id": str(memory.user_id)},
    )


def upsert_memory_sync(memory: Memory) -> None:
    """Synchronous variant — used by Celery / CLI contexts."""
    cli = _get_sync_client()
    collection = cli.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    document = _memory_to_document(memory)
    metadata = _memory_to_metadata(memory)
    embedding = embed_texts_sync([document])[0]
    stamp_collection_dim(collection, len(embedding))
    collection.upsert(
        ids=[str(memory.id)],
        documents=[document],
        embeddings=[embedding],
        metadatas=[metadata],
    )
    log.info(
        "Upserted memory into ChromaDB (sync)",
        extra={"memory_id": str(memory.id), "user_id": str(memory.user_id)},
    )


def upsert_memories_sync(memories: list[Memory]) -> int:
    """Batch-embed and upsert many memories (sync). Returns count written.

    Embeds all documents in one batched embedding pass (respecting
    ``EMBED_BATCH_SIZE`` inside ``embed_texts_sync``) and writes them in a
    single Chroma upsert. Used by the reindex/backfill task.
    """
    if not memories:
        return 0
    cli = _get_sync_client()
    collection = cli.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    documents = [_memory_to_document(m) for m in memories]
    metadatas = [_memory_to_metadata(m) for m in memories]
    ids = [str(m.id) for m in memories]
    embeddings = embed_texts_sync(documents)
    stamp_collection_dim(collection, len(embeddings[0]))
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(ids)


def get_existing_memory_ids_sync(memory_ids: list[str]) -> set[str]:
    """Return the subset of ``memory_ids`` already present in the collection.

    Used by the reindex task to compute which memories are missing their
    vector without re-embedding everything.
    """
    if not memory_ids:
        return set()
    cli = _get_sync_client()
    collection = cli.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    found = collection.get(ids=memory_ids, include=[])
    return set(found.get("ids", []) or [])


async def get_memory_ids_present(memory_ids: list[str]) -> set[str]:
    """Return the subset of ``memory_ids`` that still exist in the collection.

    Verification seam for erasure receipts: after deletion the erasure
    service re-queries the collection through this helper and records any
    ids still present as residuals. Async counterpart of
    :func:`get_existing_memory_ids_sync`.
    """
    if not memory_ids:
        return set()
    collection = await _get_collection()
    found = await collection.get(ids=memory_ids, include=[])
    return {str(i) for i in (found.get("ids") or [])}


async def delete_memory(memory_id: str) -> None:
    """Remove a memory's vector from the collection (best-effort)."""
    try:
        collection = await _get_collection()
        await collection.delete(ids=[memory_id])
        log.info("Deleted memory from ChromaDB", extra={"memory_id": memory_id})
    except Exception as e:
        log.warning(
            "Failed to delete memory from ChromaDB",
            extra={"memory_id": memory_id, "error": str(e)},
        )


async def delete_memories(memory_ids: list[str]) -> None:
    """Remove many memories' vectors from the collection (best-effort)."""
    if not memory_ids:
        return
    try:
        collection = await _get_collection()
        await collection.delete(ids=memory_ids)
        log.info("Deleted memories from ChromaDB", extra={"n": len(memory_ids)})
    except Exception as e:
        log.warning(
            "Failed to batch-delete memories from ChromaDB",
            extra={"n": len(memory_ids), "error": str(e)},
        )


def delete_memories_sync(memory_ids: list[str]) -> None:
    """Synchronous batch delete — used by Celery ingestion (best-effort)."""
    if not memory_ids:
        return
    try:
        cli = _get_sync_client()
        collection = cli.get_or_create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        collection.delete(ids=memory_ids)
        log.info("Deleted memories from ChromaDB (sync)", extra={"n": len(memory_ids)})
    except Exception as e:
        log.warning(
            "Failed to batch-delete memories from ChromaDB (sync)",
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

    Returns a list of dicts:
        {memory_id, content, score, metadata, rank, source="vector"}
    """
    try:
        collection = await _get_collection()
    except Exception as e:
        log.warning("Chroma unavailable for search", extra={"error": str(e)})
        return []

    # Fail loud on backend/dim switches; never stamp here (read path).
    check_collection_dim(collection, len(query_embedding))

    count = await collection.count()
    if count == 0:
        return []

    user_filter: dict[str, Any] = {"user_id": {"$eq": user_id}}
    if where:
        user_filter.update(where)

    results = await collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        where=user_filter,
    )

    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]

    if not docs:
        return []

    out: list[dict[str, Any]] = []
    for i, (doc, dist, meta, mid) in enumerate(
        zip(docs, distances, metadatas, ids, strict=False)
    ):
        out.append(
            {
                "memory_id": mid,
                "content":   doc,
                "score":     1.0 - dist,  # cosine distance -> similarity
                "metadata":  meta,
                "rank":      i,
                "source":    "vector",
            }
        )
    return out
