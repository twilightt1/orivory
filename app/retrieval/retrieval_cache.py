from __future__ import annotations

import json
import logging

from app.config import settings
from app.redis_client import get_redis

log = logging.getLogger(__name__)
CACHE_TTL = 300


def query_cache_prefix(conversation_id: str) -> str:
    return f"rag:query:conv:{conversation_id}:"


def query_cache_key(conversation_id: str, query_hash: str) -> str:
    return f"{query_cache_prefix(conversation_id)}{query_hash}"


async def get_cached_chunks(conversation_id: str, query_hash: str) -> list[dict] | None:
    redis = await get_redis()
    cached = await redis.get(query_cache_key(conversation_id, query_hash))
    if not cached:
        return None
    payload = json.loads(cached)
    return payload.get("chunks", [])


async def set_cached_chunks(
    conversation_id: str,
    query_hash: str,
    chunks: list[dict],
    ttl: int = CACHE_TTL,
) -> None:
    redis = await get_redis()
    await redis.setex(
        query_cache_key(conversation_id, query_hash),
        ttl,
        json.dumps({"chunks": chunks}),
    )


async def invalidate_query_cache(conversation_id: str) -> int:
    redis = await get_redis()
    pattern = f"{query_cache_prefix(conversation_id)}*"
    cursor = 0
    deleted = 0
    while True:
        cursor, keys = await redis.scan(cursor=cursor, match=pattern, count=100)
        if keys:
            deleted += await redis.delete(*keys)
        if cursor == 0:
            break
    return deleted


def invalidate_query_cache_sync(conversation_id: str) -> int:
    if not settings.REDIS_URL:  # lite mode: async path uses InMemoryRedis
        return 0
    import redis as redis_lib

    redis = redis_lib.from_url(settings.REDIS_URL, decode_responses=True)
    pattern = f"{query_cache_prefix(conversation_id)}*"
    deleted = 0
    for key in redis.scan_iter(match=pattern, count=100):
        deleted += redis.delete(key)
    return deleted
