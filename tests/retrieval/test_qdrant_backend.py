"""Qdrant client foundation: ONE owner in local mode, async+sync in server mode.

DB-free and service-free by construction: local mode runs against a private
``tmp_path`` folder (closed in teardown, so its exclusive lock never leaks into
another test), and no test dials a socket: the one server-mode test builds its
remote clients with the version handshake disabled. The single-owner rule is a
real property of qdrant-client 1.19 local mode — ``QdrantClient(path=...)``
locks the folder and the second opener raises ``RuntimeError: ... already
accessed`` — pinned here against the library, not mocked.
"""
from __future__ import annotations

import sys
import threading
import uuid
from typing import Any

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app import main
from app.config import Settings, settings
from app.retrieval import vector_backend

COLLECTION = "orivory_memories__testgen"


@pytest_asyncio.fixture
async def local(tmp_path, monkeypatch):
    """QDRANT_MODE=local on a private folder; the folder is always closed."""
    folder = tmp_path / "qdrant"
    folder.mkdir()
    monkeypatch.setattr(settings, "QDRANT_MODE", "local")
    monkeypatch.setattr(settings, "QDRANT_LOCAL_PATH", str(folder))
    yield folder
    await vector_backend.close_clients()


async def test_local_client_is_single_owner(local):
    sync_client = vector_backend.get_sync_client()
    async_client = vector_backend.get_async_client()

    assert isinstance(sync_client, vector_backend._LocalSyncProxy)
    assert isinstance(async_client, vector_backend._SyncAsAsync)
    # Both faces drive the ONE embedded client — a second QdrantClient on the
    # same folder is what already-accesses the lock.
    assert async_client._inner is sync_client._inner
    assert vector_backend.get_async_client() is async_client  # cached handle
    assert vector_backend.get_sync_client() is sync_client

    # Every local call is submitted to the dedicated local executor. Swapped by
    # hand (not monkeypatch): the fixture teardown closes the executor, and a
    # deferred restore would put a shut-down one back in the module.
    threads: list[str] = []
    real_executor = vector_backend._LOCAL_EXECUTOR

    class _RecordingExecutor:
        def submit(self, fn, /, *args, **kwargs):
            def _run():
                threads.append(threading.current_thread().name)
                return fn()

            return real_executor.submit(_run)

        def shutdown(self, /, *args, **kwargs):
            return real_executor.shutdown(*args, **kwargs)

    vector_backend._LOCAL_EXECUTOR = _RecordingExecutor()
    try:
        vector_backend.ensure_collection("memory", COLLECTION, dim=384)
        vector = [0.25] * 384
        point_id = str(uuid.uuid4())
        await async_client.upsert(
            collection_name=COLLECTION, points=[PointStruct(id=point_id, vector=vector)]
        )
        hits = (
            await async_client.query_points(collection_name=COLLECTION, query=vector, limit=1)
        ).points
    finally:
        vector_backend._LOCAL_EXECUTOR = real_executor

    assert [hit.id for hit in hits] == [point_id]
    # Cosine SIMILARITY (1.0 for an identical vector), never 1 - distance.
    assert hits[0].score == pytest.approx(1.0)
    assert (await async_client.count(COLLECTION)).count == 1
    assert threads, "no local call went through the executor"
    assert all(name.startswith("qdrant-local") for name in threads)


async def test_local_client_is_opened_on_and_driven_from_owner_thread(local, monkeypatch):
    """Spec §3.1, literally: the ONE local client is CONSTRUCTED on the
    ``qdrant-local`` executor thread, and a call made from this (main) thread
    runs there too — sync face and async face alike.
    """
    opened: list[str] = []
    called: list[str] = []

    class _RecordingClient(QdrantClient):  # a REAL client that leaves thread evidence
        def __init__(self, *args, **kwargs):
            opened.append(threading.current_thread().name)
            super().__init__(*args, **kwargs)

        def collection_exists(self, collection_name: str, **kwargs: Any) -> bool:
            called.append(threading.current_thread().name)
            return super().collection_exists(collection_name, **kwargs)

    monkeypatch.setattr(vector_backend, "QdrantClient", _RecordingClient)
    caller_thread = threading.current_thread().name

    sync_client = vector_backend.get_sync_client()  # triggers the lazy open
    assert vector_backend.is_local_mode()
    assert opened, "the local client was never constructed"
    assert opened[0].startswith("qdrant-local")
    assert opened[0] != caller_thread

    # Real calls from this thread, through both faces, land on the owner thread.
    assert sync_client.collection_exists(COLLECTION) is False
    vector_backend.ensure_collection("memory", COLLECTION, dim=384)
    assert sync_client.collection_exists(COLLECTION) is True
    await vector_backend.get_async_client().count(COLLECTION)

    assert called and all(name.startswith("qdrant-local") for name in called)
    assert set(called) == set(opened)  # the thread that opened it is the one running it


async def test_second_owner_is_blocked(local):
    vector_backend.get_sync_client()  # the owner

    with pytest.raises(RuntimeError, match="already accessed"):
        QdrantClient(path=str(local))


async def test_multiworker_local_refuses_boot(local, monkeypatch):
    monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
    monkeypatch.delenv("UVICORN_WORKERS", raising=False)
    monkeypatch.delenv("UVICORN_RELOAD", raising=False)
    monkeypatch.setattr(sys, "argv", ["app.main:app"])
    main._refuse_multi_owner_local_qdrant()  # exactly one process: fine

    monkeypatch.setenv("WEB_CONCURRENCY", "2")
    with pytest.raises(RuntimeError, match="QDRANT_MODE=local"):
        main._refuse_multi_owner_local_qdrant()

    monkeypatch.delenv("WEB_CONCURRENCY")
    monkeypatch.setattr(sys, "argv", ["app.main:app", "--reload"])
    with pytest.raises(RuntimeError, match="QDRANT_MODE=local"):
        main._refuse_multi_owner_local_qdrant()

    monkeypatch.setattr(sys, "argv", ["app.main:app", "--workers", "4"])
    with pytest.raises(RuntimeError, match="QDRANT_MODE=local"):
        main._refuse_multi_owner_local_qdrant()

    # uvicorn also takes the single-token spelling and the UVICORN_ envvar.
    monkeypatch.setattr(sys, "argv", ["app.main:app", "--workers=4"])
    with pytest.raises(RuntimeError, match="QDRANT_MODE=local"):
        main._refuse_multi_owner_local_qdrant()

    monkeypatch.setattr(sys, "argv", ["app.main:app"])
    monkeypatch.setenv("UVICORN_WORKERS", "4")
    with pytest.raises(RuntimeError, match="QDRANT_MODE=local"):
        main._refuse_multi_owner_local_qdrant()

    # One worker, spelled any way, is still one owner.
    monkeypatch.setenv("UVICORN_WORKERS", "1")
    main._refuse_multi_owner_local_qdrant()
    monkeypatch.delenv("UVICORN_WORKERS")
    monkeypatch.setattr(sys, "argv", ["app.main:app", "--workers=1"])
    main._refuse_multi_owner_local_qdrant()

    # The guard is the first thing the lifespan does: the app refuses to serve.
    monkeypatch.setattr(sys, "argv", ["app.main:app"])
    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    with pytest.raises(RuntimeError, match="QDRANT_MODE=local"):
        async with main.lifespan(main.app):
            pass


def test_server_mode_allows_workers(monkeypatch):
    monkeypatch.setattr(settings, "QDRANT_MODE", "server")
    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    monkeypatch.setattr(sys, "argv", ["app.main:app", "--workers", "4"])

    main._refuse_multi_owner_local_qdrant()  # server mode has no folder lock


async def test_server_mode_uses_async_and_sync_clients(monkeypatch):
    monkeypatch.setattr(settings, "QDRANT_MODE", "server")
    # Nothing listens on localhost:6333 in this test and these are REAL clients:
    # skip the version handshake both constructors would otherwise dial for.
    monkeypatch.setattr(
        vector_backend, "QdrantClient", lambda **kw: QdrantClient(**kw, check_compatibility=False)
    )
    monkeypatch.setattr(
        vector_backend,
        "AsyncQdrantClient",
        lambda **kw: AsyncQdrantClient(**kw, check_compatibility=False),
    )

    async_client = vector_backend.get_async_client()
    sync_client = vector_backend.get_sync_client()

    assert isinstance(async_client, AsyncQdrantClient)  # never the local shim
    assert isinstance(sync_client, QdrantClient)
    assert async_client is not sync_client
    await vector_backend.close_clients()  # no server needed: nothing connected


async def test_server_mode_clients_carry_a_bounded_request_timeout(monkeypatch):
    """A hung Qdrant server must fail a caller, not hold it: both server-mode
    clients are built with the module's bounded request timeout."""
    monkeypatch.setattr(settings, "QDRANT_MODE", "server")
    seen: dict[str, dict] = {}

    def _sync(**kwargs):
        seen["sync"] = kwargs
        return QdrantClient(**kwargs, check_compatibility=False)

    def _async(**kwargs):
        seen["async"] = kwargs
        return AsyncQdrantClient(**kwargs, check_compatibility=False)

    monkeypatch.setattr(vector_backend, "QdrantClient", _sync)
    monkeypatch.setattr(vector_backend, "AsyncQdrantClient", _async)

    vector_backend.get_sync_client()
    vector_backend.get_async_client()

    assert seen["sync"]["timeout"] == vector_backend.SERVER_REQUEST_TIMEOUT
    assert seen["async"]["timeout"] == vector_backend.SERVER_REQUEST_TIMEOUT
    await vector_backend.close_clients()  # no server needed: nothing connected


async def test_close_clients_releases_lock(local):
    folder = str(local)
    vector_backend.get_sync_client()

    await vector_backend.close_clients()
    await vector_backend.close_clients()  # idempotent

    reopened = QdrantClient(path=folder)  # the lock is really released
    reopened.close()


def test_mode_flip_is_unconditional(monkeypatch):
    monkeypatch.delenv("QDRANT_MODE", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    monkeypatch.delenv("QDRANT_URL", raising=False)  # a dev shell's URL must not decide this

    # no API key + a localhost URL -> embedded local Qdrant (the same flip rule
    # the retired Chroma mode had); anything that can reach a server stays
    # "server". No LITE_MODE flag: this IS the product's default shape.
    assert Settings(_env_file=None).QDRANT_MODE == "local"
    assert Settings(_env_file=None, QDRANT_API_KEY="key").QDRANT_MODE == "server"
    assert (
        Settings(_env_file=None, QDRANT_URL="http://qdrant:6333").QDRANT_MODE
        == "server"
    )


async def test_async_shim_covers_local_client_surface(local):
    client = vector_backend.get_async_client()

    assert await client.collection_exists(COLLECTION) is False
    await client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    assert await client.collection_exists(COLLECTION) is True
    assert COLLECTION in [c.name for c in (await client.get_collections()).collections]
    assert (await client.get_collection(COLLECTION)).config.params.vectors.size == 384

    vector = [0.5] * 384
    ids = [str(uuid.uuid4()) for _ in range(2)]
    await client.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(id=i, vector=vector, payload={"memory_id": i}) for i in ids],
    )
    assert (await client.count(COLLECTION)).count == 2
    assert [r.id for r in await client.retrieve(collection_name=COLLECTION, ids=[ids[0]])] == [ids[0]]
    hits = (await client.query_points(collection_name=COLLECTION, query=vector, limit=2)).points
    assert {hit.id for hit in hits} == set(ids)
    await client.delete(collection_name=COLLECTION, points_selector=[ids[0]])
    assert (await client.count(COLLECTION)).count == 1

    # ensure_collection: create-if-missing with the CALLER's dim (the
    # fingerprint's 384/1024/1536 — never a constant here), and idempotent.
    wide = "orivory_memories__wide"
    vector_backend.ensure_collection("memory", wide, dim=1024)
    vector_backend.ensure_collection("memory", wide, dim=1024)
    assert vector_backend.collection_info("memory", wide) == {"dim": 1024, "distance": "Cosine"}
    assert vector_backend.collection_info("memory", COLLECTION) == {
        "dim": 384,
        "distance": "Cosine",
    }
