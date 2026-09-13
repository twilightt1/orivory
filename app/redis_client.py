"""Redis access with a zero-dependency lite-mode fallback.

Full-stack deployments set ``REDIS_URL`` and get a real Redis connection
pool. Lite mode (``REDIS_URL`` empty, as ``LITE_MODE=1`` defaults) returns
an in-process ``InMemoryRedis`` that covers the exact API surface the app
uses — get/set/setex/delete/incr/expire/ping plus the sorted-set calls of
the rate limiter — so caches and rate limiting keep working without Redis.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover — imported lazily in get_pool()
    from redis.asyncio import ConnectionPool, Redis

from app.config import settings

_pool: ConnectionPool | None = None
_pool_loop: object | None = None
_memory_redis: InMemoryRedis | None = None


class InMemoryRedis:
    """Process-local Redis stand-in (single-instance lite mode only).

    Implements the subset of redis.asyncio.Redis that Orivory calls:
    string ops, counters, expiry, and the sorted-set operations of the
    rate limiter. Data is per-process and lost on restart — acceptable for
    caches and rate limits, which is all lite mode uses Redis for.
    """

    def __init__(self) -> None:
        self._data: dict[str, str] = {}
        self._expiry: dict[str, float] = {}
        self._zsets: dict[str, dict[str, float]] = {}
        self._sets: dict[str, set[str]] = {}

    def _live(self, key: str) -> bool:
        expires = self._expiry.get(key)
        if expires is not None and expires < time.monotonic():
            self._data.pop(key, None)
            self._zsets.pop(key, None)
            self._expiry.pop(key, None)
            return False
        return key in self._data or key in self._zsets

    async def get(self, key: str) -> str | None:
        return self._data.get(key) if self._live(key) else None

    async def set(self, key: str, value: Any, ex: int | None = None) -> bool:
        self._data[key] = str(value)
        if ex is not None:
            self._expiry[key] = time.monotonic() + ex
        else:
            self._expiry.pop(key, None)
        return True

    async def setex(self, key: str, seconds: int, value: Any) -> bool:
        return await self.set(key, value, ex=seconds)

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            if self._live(key):
                removed += 1
            self._data.pop(key, None)
            self._zsets.pop(key, None)
            self._expiry.pop(key, None)
        return removed

    async def incr(self, key: str) -> int:
        current = int(self._data.get(key, 0) or 0) + 1
        self._data[key] = str(current)
        return current

    async def expire(self, key: str, seconds: int) -> bool:
        if not self._live(key):
            return False
        self._expiry[key] = time.monotonic() + seconds
        return True

    async def ping(self) -> bool:
        return True

    async def ttl(self, key: str) -> int:
        expires = self._expiry.get(key)
        if expires is None or not self._live(key):
            return -1
        return int(expires - time.monotonic())

    # sets (refresh-token index)
    async def sadd(self, key: str, *members: str) -> int:
        added = 0
        self._expiry.pop(key, None)
        members_set = self._sets.setdefault(key, set())
        for member in members:
            if member not in members_set:
                members_set.add(member)
                added += 1
        return added

    async def srem(self, key: str, *members: str) -> int:
        members_set = self._sets.get(key, set())
        removed = 0
        for member in members:
            if member in members_set:
                members_set.discard(member)
                removed += 1
        return removed

    async def smembers(self, key: str) -> set[str]:
        return set(self._sets.get(key, set()))

    # sorted sets (rate limiter)
    async def zadd(self, key: str, mapping: dict[str, float]) -> int:
        added = 0
        zset = self._zsets.setdefault(key, {})
        for member, score in mapping.items():
            if member not in zset:
                added += 1
            zset[member] = float(score)
        return added

    async def zcard(self, key: str) -> int:
        self._live(key)  # prune
        return len(self._zsets.get(key, {}))

    async def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> int:
        zset = self._zsets.get(key, {})
        stale = [m for m, s in zset.items() if min_score <= s <= max_score]
        for member in stale:
            del zset[member]
        return len(stale)

    async def scan_iter(self, match: str | None = None) -> AsyncIterator[str]:
        import fnmatch

        for key in list(self._data):
            if self._live(key) and (match is None or fnmatch.fnmatch(key, match)):
                yield key

    # async-iterator-less contexts some callers may use
    def pipeline(self):  # pragma: no cover — unused by current call sites
        return _InMemoryPipeline(self)


class _InMemoryPipeline:
    """Command-buffering pipeline: queues calls, replays on execute()."""

    def __init__(self, redis: InMemoryRedis) -> None:
        self._redis = redis
        self._commands: list[tuple[str, tuple[Any, ...]]] = []

    def _queue(self, name: str, *args: Any) -> _InMemoryPipeline:
        self._commands.append((name, args))
        return self

    def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> _InMemoryPipeline:
        return self._queue("zremrangebyscore", key, min_score, max_score)

    def zadd(self, key: str, mapping: dict[str, float]) -> _InMemoryPipeline:
        return self._queue("zadd", key, mapping)

    def zcard(self, key: str) -> _InMemoryPipeline:
        return self._queue("zcard", key)

    def expire(self, key: str, seconds: int) -> _InMemoryPipeline:
        return self._queue("expire", key, seconds)

    def setex(self, key: str, seconds: int, value: Any) -> _InMemoryPipeline:
        return self._queue("setex", key, seconds, value)

    def set(self, key: str, value: Any, ex: int | None = None) -> _InMemoryPipeline:
        return self._queue("set", key, value, ex)

    def get(self, key: str) -> _InMemoryPipeline:
        return self._queue("get", key)

    def delete(self, *keys: str) -> _InMemoryPipeline:
        return self._queue("delete", *keys)

    def incr(self, key: str) -> _InMemoryPipeline:
        return self._queue("incr", key)

    def ping(self) -> _InMemoryPipeline:
        return self._queue("ping")

    def sadd(self, key: str, *members: str) -> _InMemoryPipeline:
        return self._queue("sadd", key, *members)

    def srem(self, key: str, *members: str) -> _InMemoryPipeline:
        return self._queue("srem", key, *members)

    def smembers(self, key: str) -> _InMemoryPipeline:
        return self._queue("smembers", key)

    async def execute(self) -> list[Any]:
        results = []
        for name, args in self._commands:
            results.append(await getattr(self._redis, name)(*args))
        return results


def get_pool() -> ConnectionPool:
    global _pool, _pool_loop
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    if _pool is None or (_pool_loop is not None and current_loop is not None and _pool_loop is not current_loop):
        from redis.asyncio import ConnectionPool as _ConnectionPool

        _pool = _ConnectionPool.from_url(
            settings.REDIS_URL,
            max_connections=settings.REDIS_POOL_MAX,
            decode_responses=True,
        )
        _pool_loop = current_loop
    return _pool


async def get_redis() -> Redis | InMemoryRedis:
    """Return the shared Redis client, or the in-memory stand-in in lite mode."""
    global _memory_redis
    if not settings.REDIS_URL:
        if _memory_redis is None:
            _memory_redis = InMemoryRedis()
        return _memory_redis
    return Redis(connection_pool=get_pool())


__all__ = ["InMemoryRedis", "get_pool", "get_redis"]
