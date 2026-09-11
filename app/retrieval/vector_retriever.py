from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

import chromadb
import httpx

from app.config import settings
from app.retrieval.embedder import (
    astamp_collection_dim,
    check_collection_dim,
    embed_query,
    embed_texts,
    embed_texts_sync,
    stamp_collection_dim,
)

log = logging.getLogger(__name__)
_async_client: chromadb.AsyncHttpClient | None = None
_sync_client: chromadb.HttpClient | None = None


class VectorUnavailableError(Exception):
    """ChromaDB itself is unreachable (not: empty collection, no matches).

    Raised instead of returning [] so callers can distinguish "vector
    search is down" (degrade to BM25-only + flag it) from "no vectors".
    """


def with_retry(retries: int = 3, base_delay: float = 1.0):
    def decorator(func: Callable):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                last_exc = None
                for i in range(retries):
                    try:
                        return await func(*args, **kwargs)
                    except (ValueError, httpx.ConnectError, httpx.HTTPError, Exception) as e:
                        last_exc = e

                        if any(msg in str(e) for msg in ["Could not connect", "connection", "Refused"]) or \
                           isinstance(e, (ValueError, httpx.ConnectError)):
                            delay = base_delay * (2 ** i)
                            log.warning(f"Chroma connection failed (attempt {i+1}/{retries}). Retrying in {delay}s...", extra={"error": str(e)})
                            await asyncio.sleep(delay)
                        else:
                            raise e
                log.error("Failed to connect to Chroma after all retries.", extra={"error": str(last_exc)})
                raise last_exc
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                last_exc = None
                for i in range(retries):
                    try:
                        return func(*args, **kwargs)
                    except (ValueError, httpx.ConnectError, httpx.HTTPError, Exception) as e:
                        last_exc = e
                        if any(msg in str(e) for msg in ["Could not connect", "connection", "Refused"]) or \
                           isinstance(e, (ValueError, httpx.ConnectError)):
                            delay = base_delay * (2 ** i)
                            log.warning(f"Chroma connection failed (attempt {i+1}/{retries}). Retrying in {delay}s...", extra={"error": str(e)})
                            time.sleep(delay)
                        else:
                            raise e
                log.error("Failed to connect to Chroma after all retries.", extra={"error": str(last_exc)})
                raise last_exc
            return sync_wrapper
    return decorator


@with_retry()
async def _get_async_client() -> chromadb.AsyncHttpClient:
    global _async_client
    if _async_client is None:
        _async_client = await chromadb.AsyncHttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
        )
    return _async_client


@with_retry()
def _get_sync_client() -> chromadb.HttpClient:
    global _sync_client
    if _sync_client is None:
        _sync_client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
        )
    return _sync_client


def _col_name(conversation_id: str) -> str:
    return f"rag_conv_{conversation_id}"




async def upsert_chunks(conversation_id: str, chunks: list[dict]) -> None:
    if not chunks:
        return
    cli        = await _get_async_client()
    collection = await cli.get_or_create_collection(
        _col_name(conversation_id),
        metadata={"hnsw:space": "cosine"},
    )
    embeddings = await embed_texts([c["content"] for c in chunks])
    await astamp_collection_dim(collection, len(embeddings[0]))
    await collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["content"] for c in chunks],
        embeddings=embeddings,
        metadatas=[c["metadata"] for c in chunks],
    )
    log.info("Upserted child chunks", extra={"conversation_id": conversation_id, "n": len(chunks)})


def upsert_chunks_sync(conversation_id: str, chunks: list[dict]) -> None:
    if not chunks:
        return
    cli        = _get_sync_client()
    collection = cli.get_or_create_collection(
        _col_name(conversation_id),
        metadata={"hnsw:space": "cosine"},
    )
    embeddings = embed_texts_sync([c["content"] for c in chunks])
    stamp_collection_dim(collection, len(embeddings[0]))
    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["content"] for c in chunks],
        embeddings=embeddings,
        metadatas=[c["metadata"] for c in chunks],
    )




async def search(
    query: str,
    top_k: int,
    conversation_id: str,
    hyde_text: str | None = None,
) -> list[dict]:
    try:
        cli        = await _get_async_client()
        collection = await cli.get_collection(_col_name(conversation_id))
    except Exception as exc:
        raise VectorUnavailableError(f"ChromaDB unreachable: {exc}") from exc

    count = await collection.count()
    if count == 0:
        return []


    embed_input = hyde_text if hyde_text else query
    embedding   = await embed_query(embed_input)

    # Fail loud on backend/dim switches; never stamp here (read path).
    check_collection_dim(collection, len(embedding))

    results = await collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, count),
    )

    if not results["documents"] or not results["documents"][0]:
        return []

    return [
        {
            "content":   doc,
            "score":     1 - dist,
            "source":    "vector",
            "rank":      i,
            "metadata":  meta,
            "child_id":  meta.get("child_id", ""),
            "parent_id": meta.get("parent_id", ""),
        }
        for i, (doc, dist, meta) in enumerate(zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0], strict=False,
        ))
    ]




async def delete_document_chunks(conversation_id: str, document_id: str) -> None:
    try:
        cli        = await _get_async_client()
        collection = await cli.get_collection(_col_name(conversation_id))
        results    = await collection.get(where={"document_id": {"$eq": document_id}})
        if results["ids"]:
            await collection.delete(ids=results["ids"])
    except Exception as e:
        log.warning("Failed to delete chunks", extra={"error": str(e)})


def delete_document_chunks_sync(conversation_id: str, document_id: str) -> None:
    try:
        cli = _get_sync_client()
        collection = cli.get_collection(_col_name(conversation_id))
        results = collection.get(where={"document_id": {"$eq": document_id}})
        if results["ids"]:
            collection.delete(ids=results["ids"])
    except Exception as e:
        log.warning("Failed to delete chunks sync", extra={"error": str(e)})


async def delete_conversation_collection(conversation_id: str) -> None:
    try:
        cli = await _get_async_client()
        await cli.delete_collection(_col_name(conversation_id))
    except Exception:
        pass
