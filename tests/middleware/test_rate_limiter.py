"""Rate limiter: bounded window state (finding 71) and its test double.

The limiter ZADDed every attempt — including the rejected ones — and never
removed one inside the window, so one key's sorted set (the Redis stand-in's
dict entry, a real Redis zset) grew with the caller's request rate: limit=2
and five calls left ``zcard == 5``. Retention is now bounded to the newest
``limit`` attempts, and the members trimmed are exactly the ones that leave
the window first, so no decision changes.

The last test drives ``POST /api/v1/memories/recall`` through the REAL
``enforce_llm_quota``: the suite's redis double must reply the same FIVE
values the real pipeline does, or the limiter's unpack raises and the request
500s.
"""
from __future__ import annotations

import time

import pytest
from fastapi import HTTPException

from app.middleware import rate_limiter
from app.redis_client import InMemoryRedis


@pytest.fixture()
def redis(monkeypatch) -> InMemoryRedis:
    """The real in-process client: the suite's root conftest mocks it away."""
    client = InMemoryRedis()

    async def _get_redis() -> InMemoryRedis:
        return client

    monkeypatch.setattr(rate_limiter, "get_redis", _get_redis)
    return client


async def _call(user: str, *, window: int = 60, limit: int = 2) -> str:
    try:
        await rate_limiter.check_rate_limit(user, window_seconds=window, limit=limit)
        return "allowed"
    except HTTPException as exc:
        assert exc.status_code == 429
        return "rejected"


async def test_rejected_attempts_do_not_grow_the_window(redis):
    """limit=2, five calls: two allowed, three rejected — and two members kept."""
    outcomes = [await _call("user") for _ in range(5)]

    assert outcomes == ["allowed", "allowed", "rejected", "rejected", "rejected"]
    assert await redis.zcard("ratelimit:user:60") <= 2, (
        "rejected attempts must not be retained: the window is what limits, not the log")


async def test_entries_that_left_the_window_are_still_pruned(redis):
    """The prune is unchanged: only what is inside the window counts."""
    await redis.zadd("ratelimit:user:60", {"stale": time.time() - 120})

    assert await _call("user") == "allowed"
    assert await redis.zcard("ratelimit:user:60") == 1  # the stale member is gone


async def test_recall_endpoint_runs_the_real_quota_dependency(tmp_path, monkeypatch):
    """POST /api/v1/memories/recall through the REAL ``enforce_llm_quota``.

    No dependency override: auth, the limiter and the quota guard all run. The
    limiter unpacks FIVE pipeline replies, so the suite's redis double has to
    answer with the real shape — with a 4-value double this request raised
    ``ValueError: not enough values to unpack (expected 5, got 4)`` and the
    endpoint answered 500. Self-contained (own temp-SQLite DB, no fixtures) so
    the guaranteed ``--confcutdir=tests/middleware`` CI run exercises it too.
    """
    from httpx import ASGITransport, AsyncClient
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from app import database
    from app.database import Base, get_db
    from app.main import app
    from app.retrieval.memory import freshness
    from app.retrieval.memory import retriever as retriever_module

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'quota.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    # The barrier and the drain open their OWN sessions: point them at this DB.
    monkeypatch.setattr(freshness, "AsyncSessionLocal", sessions)
    monkeypatch.setattr(database, "AsyncSessionLocal", sessions)

    async def _db_override():
        async with sessions() as session:
            yield session

    # Only the retrieval seams (no LLM / no model / no vector store in tests):
    # everything the guard itself touches stays real.
    async def _empty_context(*_args, **_kwargs):
        return []

    async def _identity_rewrite(*_args, **_kwargs):
        return {"rewritten_query": "hello", "entities": [], "reasoning": None,
                "_fallback_used": False}

    async def _zero_embedding(*_args, **_kwargs):
        return [0.0] * 8

    async def _no_vectors(*_args, **_kwargs):
        return []

    monkeypatch.setattr(retriever_module, "fetch_personal_context", _empty_context)
    monkeypatch.setattr(retriever_module, "rewrite_query", _identity_rewrite)
    monkeypatch.setattr(retriever_module, "embed_query", _zero_embedding)
    monkeypatch.setattr(retriever_module, "search_memories", _no_vectors)

    app.dependency_overrides[get_db] = _db_override
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v1/memories/recall", json={"query": "hello"}
            )
    finally:
        app.dependency_overrides.clear()
        await engine.dispose()

    assert response.status_code == 200, response.text
    assert response.json()["results"] == []
