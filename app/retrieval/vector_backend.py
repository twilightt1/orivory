"""One owner for the Qdrant client(s) (spec §3.1).

Local (lite) mode: ONE ``QdrantClient(path=QDRANT_LOCAL_PATH)`` owns the storage
folder. Per spec §3.1 it is CONSTRUCTED on the dedicated ``qdrant-local``
executor thread and every operation — from either face — runs on that one
thread: a second client on the same folder raises ``RuntimeError: ... already
accessed``, and the embedded store keeps the thread affinity of whatever it
opened its connections on. The sync face is therefore a thin proxy that submits
each call to that thread and waits; the async face awaits the same submission,
so it never blocks the event loop.

Server (scale) mode: an ``AsyncQdrantClient`` for request paths plus a
``QdrantClient`` for Celery/CLI callers — no executor funnel: this process owns
no folder and the server serves concurrent callers.
"""
from __future__ import annotations

import asyncio
import functools
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from app.config import settings

_LOCAL_THREAD_PREFIX = "qdrant-local"

# Request timeout for the SERVER-mode clients, in seconds: a hung server must
# fail a caller, never hold it (reads re-raise, writes stay pending in the
# outbox — but the latency stays bounded). One shared constant, not a setting.
# int: the client's signature is `int | None` and it ceils the value anyway.
SERVER_REQUEST_TIMEOUT = 5

# Per-kind payload indexes (spec §4.2): the fields a filter reads must be
# indexed or every search is a full scan. SERVER mode only — the embedded
# store ignores payload indexes and warns on each request, and lite mode
# holds one user's small store anyway.
_PAYLOAD_INDEXES: dict[str, dict[str, PayloadSchemaType]] = {
    "memory": {
        "user_id": PayloadSchemaType.KEYWORD,
        "tags": PayloadSchemaType.KEYWORD,
        "pinned": PayloadSchemaType.BOOL,
        "salience": PayloadSchemaType.FLOAT,
        "captured_at": PayloadSchemaType.DATETIME,
    },
}


def _new_local_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix=_LOCAL_THREAD_PREFIX)


_LOCAL_EXECUTOR = _new_local_executor()
_OPEN_LOCK = threading.Lock()

_sync_client: QdrantClient | _LocalSyncProxy | None = None
_async_client: AsyncQdrantClient | _SyncAsAsync | None = None
_local_client: QdrantClient | None = None  # the ONE embedded client (local mode)


def is_local_mode() -> bool:
    """True when this process owns an embedded Qdrant storage folder."""
    return settings.QDRANT_MODE == "local"


def _open_local_client() -> QdrantClient:
    """Construct the embedded client. Must run ON the owner thread."""
    return QdrantClient(path=settings.QDRANT_LOCAL_PATH)


def _owner_local_client() -> QdrantClient:
    """The ONE embedded client, lazily opened on the owner thread (§3.1).

    Every caller waits for the open: until it returns there is no client, and
    the thread that gets it is the thread that must drive it.
    """
    global _local_client
    with _OPEN_LOCK:
        if _local_client is None:
            _local_client = _LOCAL_EXECUTOR.submit(_open_local_client).result()
    return _local_client


class _LocalSyncProxy:
    """Sync face over the ONE embedded client: each call runs on the owner thread.

    Deliberately not a ``QdrantClient``: a caller on any thread can hold this
    and still be serialised onto the single owner thread.
    """

    def __init__(self, inner: QdrantClient) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def _call(*args: Any, **kwargs: Any) -> Any:
            return _LOCAL_EXECUTOR.submit(functools.partial(attr, *args, **kwargs)).result()

        return _call


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


def get_sync_client() -> QdrantClient | _LocalSyncProxy:
    """The process's sync client, opened on first use.

    Local mode: a proxy over the ONE embedded client (spec §3.1), so a caller on
    any thread still runs on the owner thread. Server mode: a real client.
    """
    global _sync_client
    if _sync_client is None:
        if is_local_mode():
            _sync_client = _LocalSyncProxy(_owner_local_client())
        else:
            _sync_client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY or None,
                timeout=SERVER_REQUEST_TIMEOUT,
            )
    return _sync_client


def get_async_client() -> AsyncQdrantClient | _SyncAsAsync:
    """Async client: a real ``AsyncQdrantClient`` (server) or the local shim."""
    global _async_client
    if _async_client is None:
        if is_local_mode():
            _async_client = _SyncAsAsync(_owner_local_client())
        else:
            _async_client = AsyncQdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY or None,
                timeout=SERVER_REQUEST_TIMEOUT,
            )
    return _async_client


async def close_clients() -> None:
    """Close the client(s) and stop the local executor. Idempotent.

    Releasing the folder lock here is what lets the next process — or the
    offline migration CLI — open the same path. The local close runs on the
    owner thread like every other local operation.
    """
    global _sync_client, _async_client, _local_client, _LOCAL_EXECUTOR

    async_client, sync_client, local_client = _async_client, _sync_client, _local_client
    _async_client, _sync_client, _local_client = None, None, None

    if async_client is not None and not isinstance(async_client, _SyncAsAsync):
        await async_client.close()
    if local_client is not None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(_LOCAL_EXECUTOR, local_client.close)
    elif sync_client is not None and not isinstance(sync_client, _LocalSyncProxy):
        sync_client.close()
    _LOCAL_EXECUTOR.shutdown(wait=True)
    _LOCAL_EXECUTOR = _new_local_executor()


def _missing_payload_indexes(kind: str, present: Any) -> dict[str, PayloadSchemaType]:
    """The kind's payload indexes this collection does not have yet."""
    have = set(present or {})
    return {
        field: schema
        for field, schema in _PAYLOAD_INDEXES.get(kind, {}).items()
        if field not in have
    }


def _ensure_payload_indexes_sync(client: Any, kind: str, generation: str) -> None:
    if is_local_mode():
        return
    created = _missing_payload_indexes(kind, client.get_collection(generation).payload_schema)
    for field_name, field_schema in created.items():
        client.create_payload_index(
            collection_name=generation, field_name=field_name, field_schema=field_schema
        )


async def _ensure_payload_indexes_async(client: Any, kind: str, generation: str) -> None:
    if is_local_mode():
        return
    info = await client.get_collection(generation)
    created = _missing_payload_indexes(kind, info.payload_schema)
    for field_name, field_schema in created.items():
        await client.create_payload_index(
            collection_name=generation, field_name=field_name, field_schema=field_schema
        )


def ensure_collection(kind: str, generation: str, dim: int) -> None:
    """Create ``kind``'s collection for ``generation`` if it is missing.

    The generation token IS the physical name (spec §4.2): a cutover is a
    pointer flip in SQLite, never a rename in the store. ``kind`` selects the
    payload index set the family filters on (server mode only — the embedded
    store ignores indexes; in server mode this costs one metadata read per
    call, which is what "ensure" means). ``dim`` is the ACTIVE embedding
    dimension supplied by the caller from its fingerprint (384 local / 1024
    Jina / 1536 OpenAI) — never a constant here. An existing collection is left
    exactly as it is: its dim/metric belong to the contract guard, not to a
    silent overwrite.
    """
    client = get_sync_client()
    if client.collection_exists(generation):
        _ensure_payload_indexes_sync(client, kind, generation)
        return
    try:
        client.create_collection(
            collection_name=generation,
            vectors_config=VectorParams(size=int(dim), distance=Distance.COSINE),
        )
    except (ValueError, UnexpectedResponse):
        # Two booting workers can lose this check-then-create race: embedded
        # raises ValueError, a server answers 409. Anything else — including a
        # real failure wearing one of these types — re-raises below.
        if not client.collection_exists(generation):
            raise
    _ensure_payload_indexes_sync(client, kind, generation)


async def ensure_collection_async(kind: str, generation: str, dim: int) -> None:
    """Async twin of :func:`ensure_collection` (request paths never block)."""
    client = get_async_client()
    if not await client.collection_exists(generation):
        try:
            await client.create_collection(
                collection_name=generation,
                vectors_config=VectorParams(size=int(dim), distance=Distance.COSINE),
            )
        except (ValueError, UnexpectedResponse):
            if not await client.collection_exists(generation):
                raise
    await _ensure_payload_indexes_async(client, kind, generation)


def _info_from(info: Any) -> dict:
    vectors = info.config.params.vectors
    return {
        "dim": int(vectors.size),
        "distance": str(getattr(vectors.distance, "value", vectors.distance)),
    }


def collection_info(kind: str, generation: str) -> dict:
    """The collection's actual contract as the store reports it: dim/metric."""
    return _info_from(get_sync_client().get_collection(generation))


async def collection_info_async(kind: str, generation: str) -> dict:
    """Async twin of :func:`collection_info`."""
    return _info_from(await get_async_client().get_collection(generation))
