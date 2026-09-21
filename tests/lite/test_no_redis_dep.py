"""The cache stack must work with the `redis` package absent.

The one-container product never talks to a Redis server, so nothing in app/
may import the `redis` package; every cache/rate-limit call lands on the
in-process ``InMemoryRedis``.
"""

from __future__ import annotations

import asyncio
import importlib
import sys

import pytest


class _BlockRedis:
    """Meta-path finder that makes `import redis` raise ImportError."""

    def find_module(self, name, path=None):
        if name == "redis" or name.startswith("redis."):
            return self
        return None

    def load_module(self, name):
        raise ImportError(f"blocked for lite check: {name}")


@pytest.fixture()
def no_redis_package(monkeypatch):
    purged = {k: v for k, v in sys.modules.items() if k == "redis" or k.startswith("redis.")}
    for k in purged:
        del sys.modules[k]
    monkeypatch.setattr(sys, "meta_path", [_BlockRedis(), *sys.meta_path])
    yield
    for k in list(sys.modules):
        if k == "redis" or k.startswith("redis."):
            del sys.modules[k]


def test_lite_cache_stack_without_redis_package(no_redis_package, monkeypatch):
    import app.redis_client as rc

    importlib.reload(rc)  # prove the module itself imports clean

    async def _roundtrip() -> None:
        client = await rc.get_redis()
        assert type(client).__name__ == "InMemoryRedis"
        await client.set("k", "v", ex=60)
        assert await client.get("k") == "v"

    asyncio.run(_roundtrip())

    from app.retrieval.parent_store import store_parents_sync
    from app.retrieval.retrieval_cache import invalidate_query_cache_sync

    assert invalidate_query_cache_sync("cid") == 0
    assert store_parents_sync("cid", []) is None


async def test_inmemory_mget_is_expiry_aware():
    from app.redis_client import InMemoryRedis

    client = InMemoryRedis()
    await client.set("a", "1")
    await client.set("b", "2", ex=60)
    await client.set("gone", "3", ex=0)  # already past its TTL

    assert await client.mget(["a", "b", "gone", "never"]) == ["1", "2", None, None]


async def test_inmemory_scan_pages_once_with_a_terminal_cursor():
    """The cache invalidators loop on `(cursor, keys)` until cursor == 0."""
    from app.redis_client import InMemoryRedis

    client = InMemoryRedis()
    await client.set("rag:query:conv:c1:a", "1")
    await client.set("rag:query:conv:c1:b", "2")
    await client.set("rag:query:conv:c2:c", "3")

    cursor, keys = await client.scan(cursor=0, match="rag:query:conv:c1:*", count=100)

    assert cursor == 0
    assert sorted(keys) == ["rag:query:conv:c1:a", "rag:query:conv:c1:b"]


async def test_inmemory_set_nx_only_wins_on_a_missing_key():
    from app.redis_client import InMemoryRedis

    client = InMemoryRedis()
    assert await client.set("window", 0, ex=60, nx=True) is True
    assert await client.set("window", 0, ex=60, nx=True) is None  # key exists
    assert await client.incr("window") == 1  # the first set survived


async def test_inmemory_sets_are_expired_and_deleted_like_redis():
    """The refresh-token index is a set: `expire` must TTL it and `delete` must
    remove it. A set left behind grows for the container's lifetime and makes
    `delete` return a lie (redis-py answers 1 here)."""
    from app.redis_client import InMemoryRedis

    client = InMemoryRedis()
    assert await client.sadd("refresh_user:u1", "a", "b") == 2
    assert await client.expire("refresh_user:u1", 0) is True  # TTL already past
    assert await client.smembers("refresh_user:u1") == set()

    assert await client.sadd("refresh_user:u1", "a") == 1
    assert await client.delete("refresh_user:u1") == 1
    assert await client.smembers("refresh_user:u1") == set()
    assert await client.srem("refresh_user:u1", "a") == 0
