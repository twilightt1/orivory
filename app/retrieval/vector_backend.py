"""One owner for the Qdrant client(s) (spec §3.1).

Local (lite) mode: ONE ``QdrantClient(path=QDRANT_LOCAL_PATH)`` owns the storage
folder — qdrant-client locks it and a second client on the same folder raises
``RuntimeError: ... already accessed by another instance``. The async face is
therefore a thin awaitable shim (:class:`_SyncAsAsync`) over that same client,
submitting every call to one dedicated executor thread; nothing here ever
creates a second client on one folder.

Server (scale) mode: an ``AsyncQdrantClient`` for request paths plus a
``QdrantClient`` for Celery/CLI callers.

Sync callers drive the sync client on their own thread by design: ownership of
the folder is per PROCESS, and the offline maintenance CLI runs with the app
stopped (spec §3.1), so it cannot race the serving process for the lock.
"""
from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import settings

_LOCAL_THREAD_PREFIX = "qdrant-local"


def _new_local_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix=_LOCAL_THREAD_PREFIX)


_LOCAL_EXECUTOR = _new_local_executor()

_sync_client: QdrantClient | None = None
_async_client: AsyncQdrantClient | _SyncAsAsync | None = None


def is_local_mode() -> bool:
    """True when this process owns an embedded Qdrant storage folder."""
    return settings.QDRANT_MODE == "local"


class _SyncAsAsync:
    """Awaitable face over the ONE local sync client — it never opens one."""

    def __init__(self, inner: QdrantClient) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        async def _call(*args: Any, **kwargs: Any) -> Any:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                _LOCAL_EXECUTOR, functools.partial(attr, *args, **kwargs)
            )

        return _call


def _open_sync_client() -> QdrantClient:
    if is_local_mode():
        return QdrantClient(path=settings.QDRANT_LOCAL_PATH)
    return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None)


def get_sync_client() -> QdrantClient:
    """The process's sync client, opened on first use."""
    global _sync_client
    if _sync_client is None:
        _sync_client = _open_sync_client()
    return _sync_client


def get_async_client() -> AsyncQdrantClient | _SyncAsAsync:
    """Async client: a real ``AsyncQdrantClient`` (server) or the local shim."""
    global _async_client
    if _async_client is None:
        if is_local_mode():
            _async_client = _SyncAsAsync(get_sync_client())
        else:
            _async_client = AsyncQdrantClient(
                url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None
            )
    return _async_client


async def close_clients() -> None:
    """Close the client(s) and stop the local executor. Idempotent.

    Releasing the folder lock here is what lets the next process — or the
    offline migration CLI — open the same path.
    """
    global _sync_client, _async_client, _LOCAL_EXECUTOR

    async_client, sync_client = _async_client, _sync_client
    _async_client, _sync_client = None, None
    if isinstance(async_client, AsyncQdrantClient):
        await async_client.close()
    if sync_client is not None:
        sync_client.close()
    _LOCAL_EXECUTOR.shutdown(wait=True)
    _LOCAL_EXECUTOR = _new_local_executor()


def _collection_name(kind: str, generation: str) -> str:
    """Physical collection name for ``kind`` (spec §4.2).

    The generation token from ``index_generations`` IS the physical name
    (``orivory_memories__<fp8>`` / ``orivory_chunks__<fp8>``): a cutover is a
    pointer flip in SQLite, never a rename in the store. ``kind`` labels the
    family the per-kind payload indexes hang off.
    """
    return generation


def ensure_collection(kind: str, generation: str, dim: int) -> None:
    """Create ``kind``'s collection for ``generation`` if it is missing.

    ``dim`` is the ACTIVE embedding dimension supplied by the caller from its
    fingerprint (384 local / 1024 Jina / 1536 OpenAI) — never a constant here.
    An existing collection is left exactly as it is: its dim/metric belong to
    the contract guard, not to a silent overwrite.
    """
    client = get_sync_client()
    name = _collection_name(kind, generation)
    if client.collection_exists(name):
        return
    try:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=int(dim), distance=Distance.COSINE),
        )
    except Exception:
        # Two server workers can race this check-then-create; only the loser of
        # that race sees an error, so re-raise anything else.
        if not client.collection_exists(name):
            raise


def collection_info(kind: str, generation: str) -> dict:
    """The collection's actual contract as the store reports it: dim/metric."""
    info = get_sync_client().get_collection(_collection_name(kind, generation))
    vectors = info.config.params.vectors
    return {
        "dim": int(vectors.size),
        "distance": str(getattr(vectors.distance, "value", vectors.distance)),
    }
