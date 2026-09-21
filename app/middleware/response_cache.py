"""
Response Caching Middleware

Caches GET responses in Redis with configurable TTL.
Supports cache invalidation on mutations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time

from fastapi import Response

from app.redis_client import get_redis

log = logging.getLogger(__name__)

# Cache configuration per route pattern. The routes this used to cover
# (discovery / insights / workspaces) went with the full-stack surface, so no
# route opts in today — the middleware stays mounted and simply passes through.
CACHE_CONFIG: dict[str, dict] = {}


def _get_cache_key(path: str, query_params: dict, user_id: str) -> str:
    """Generate cache key from request details."""
    # Sort query params for consistent hashing
    sorted_params = json.dumps(query_params, sort_keys=True)
    params_hash = hashlib.md5(sorted_params.encode()).hexdigest()[:12]
    return f"response:{path}:{user_id}:{params_hash}"


def _match_cache_config(path: str) -> dict | None:
    """Find matching cache config for path."""
    for pattern, config in CACHE_CONFIG.items():
        if path.startswith(pattern):
            return config
    return None


async def get_cached_response(cache_key: str) -> Response | None:
    """Get cached response from Redis."""
    try:
        redis = await get_redis()
        cached = await redis.get(cache_key)
        if cached:
            data = json.loads(cached)
            return Response(
                content=data["body"],
                status_code=data["status"],
                headers=data["headers"],
                media_type=data.get("media_type", "application/json"),
            )
    except Exception as e:
        log.warning(f"Cache read error: {e}")
    return None


async def set_cached_response(
    cache_key: str,
    response: Response,
    ttl: int,
) -> None:
    """Cache response in Redis."""
    try:
        redis = await get_redis()
        # Read response body
        body = b""
        async for chunk in response.body_iterator:
            body += chunk

        # Store in cache
        cache_data = {
            "body": body.decode("utf-8"),
            "status": response.status_code,
            "headers": dict(response.headers),
            "media_type": response.media_type,
            "cached_at": time.time(),
        }

        await redis.setex(cache_key, ttl, json.dumps(cache_data))
        log.debug(f"Cached response: {cache_key} (TTL: {ttl}s)")

        # Return new response with cached body
        return Response(
            content=body,
            status_code=response.status_code,
            headers=response.headers,
            media_type=response.media_type,
        )
    except Exception as e:
        log.warning(f"Cache write error: {e}")
    return None


class CacheInvalidation:
    """Utility to invalidate caches on mutations."""

    @staticmethod
    async def invalidate_user_cache(user_id: str) -> int:
        """Invalidate all cached responses for a user."""
        redis = await get_redis()
        pattern = f"response:*:{user_id}:*"
        cursor = 0
        deleted = 0

        while True:
            cursor, keys = await redis.scan(cursor=cursor, match=pattern, count=100)
            if keys:
                deleted += await redis.delete(*keys)
            if cursor == 0:
                break

        log.info(f"Invalidated {deleted} cached responses for user {user_id}")
        return deleted

    @staticmethod
    async def invalidate_pattern(pattern: str) -> int:
        """Invalidate all cached responses matching a pattern."""
        redis = await get_redis()
        cursor = 0
        deleted = 0

        while True:
            cursor, keys = await redis.scan(cursor=cursor, match=pattern, count=100)
            if keys:
                deleted += await redis.delete(*keys)
            if cursor == 0:
                break

        log.info(f"Invalidated {deleted} cached responses matching {pattern}")
        return deleted
