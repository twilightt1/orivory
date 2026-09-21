"""Redis access: the in-process ``InMemoryRedis`` stand-in, always.

One container, zero external services: there is no Redis server to talk to,
so ``get_redis()`` hands every caller the same process-local client. It
covers the exact API surface the app uses — get/set/setex/delete/mget/incr/
expire/ping, the scan forms of the cache invalidators, and the sorted-set
calls of the rate limiter — so caches and rate limiting keep working without
Redis. Data is per-process and lost on restart: acceptable for caches and
rate limits, which is all this store is used for.
"""
from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any


class InMemoryRedis:
    """Process-local Redis stand-in (single-process deployment only)."""

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
            self._sets.pop(key, None)
            self._expiry.pop(key, None)
            return False
        return key in self._data or key in self._zsets or key in self._sets

    async def get(self, key: str) -> str | None:
        return self._data.get(key) if self._live(key) else None

    async def mget(self, keys: list[str]) -> list[str | None]:
        return [self._data.get(key) if self._live(key) else None for key in keys]

    async def set(
        self, key: str, value: Any, ex: int | None = None, nx: bool = False
    ) -> bool | None:
        """SET with the redis-py NX contract.

        ``nx=True`` sets only when the key does not exist, answering True on
        the set and None when it already exists (the atomic-window pattern
        ``_count_with_window`` relies on). Without ``nx`` it always sets and
        answers True.
        """
        if nx and self._live(key):
            return None
        self._data[key] = str(value)
        if ex is not None:
            self._expiry[key] = time.monotonic() + ex
        else:
            self._expiry.pop(key, None)
        return True

    async def setex(self, key: str, seconds: int, value: Any) -> bool:
        await self.set(key, value, ex=seconds)
        return True

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            if self._live(key):
                removed += 1
            self._data.pop(key, None)
            self._zsets.pop(key, None)
            self._sets.pop(key, None)
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
        if not self._live(key):
            return 0
        members_set = self._sets.get(key, set())
        removed = 0
        for member in members:
            if member in members_set:
                members_set.discard(member)
                removed += 1
        return removed

    async def smembers(self, key: str) -> set[str]:
        return set(self._sets.get(key, set())) if self._live(key) else set()

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

    async def scan(
        self, cursor: int = 0, match: str | None = None, count: int | None = None
    ) -> tuple[int, list[str]]:
        """One-shot SCAN: every matching live key, with a terminal cursor.

        The in-process store has no bucket to page through, so the first call
        answers ``(0, keys)`` and every caller's ``while True`` loop exits on
        the same check it already makes.
        """
        return 0, [key for key in self._scan_keys() if match is None or _match(key, match)]

    async def scan_iter(self, match: str | None = None) -> AsyncIterator[str]:
        for key in self._scan_keys():
            if match is None or _match(key, match):
                yield key

    def _scan_keys(self) -> list[str]:
        return [key for key in list(self._data) if self._live(key)]

    # async-iterator-less contexts some callers may use
    def pipeline(self):  # pragma: no cover — unused by current call sites
        return _InMemoryPipeline(self)


def _match(key: str, pattern: str) -> bool:
    import fnmatch

    return fnmatch.fnmatch(key, pattern)


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


_memory_redis: InMemoryRedis | None = None


async def get_redis() -> InMemoryRedis:
    """Return the shared in-process client (there is no other kind)."""
    global _memory_redis
    if _memory_redis is None:
        _memory_redis = InMemoryRedis()
    return _memory_redis


__all__ = ["InMemoryRedis", "get_redis"]
