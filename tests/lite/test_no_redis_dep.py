"""Lite must work with the `redis` package absent (InMemoryRedis only)."""

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
    monkeypatch.setenv("REDIS_URL", "")
    import app.config as _cfg

    monkeypatch.setattr(_cfg.settings, "REDIS_URL", "")
    import app.redis_client as rc

    importlib.reload(rc)

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
