import logging
import time
import uuid

from fastapi import HTTPException

from app.redis_client import get_redis

log = logging.getLogger(__name__)


async def check_rate_limit(
    user_id: str,
    window_seconds: int = 60,
    limit: int = 60,
) -> None:
    redis = await get_redis()
    key   = f"ratelimit:{user_id}:{window_seconds}"
    now   = time.time()
    # Unique member per request: two requests within the same microsecond
    # previously shared the same zset member and one overwrote the other,
    # undercounting the window.
    member = f"{now}:{uuid.uuid4().hex}"

    pipe = redis.pipeline()
    pipe.zremrangebyscore(key, 0, now - window_seconds)
    # Add FIRST, then count — zcard-before-zadd saw N-1 for the Nth request
    # and let limit+1 requests through each window.
    pipe.zadd(key, {member: now})
    pipe.zcard(key)
    # …then keep only the newest `limit` attempts. The members trimmed are the
    # ones that leave the window FIRST, so every decision (and every release
    # moment) is unchanged — while a hammering caller can no longer grow the
    # key's sorted set with request volume: it is bounded by the limit.
    pipe.zremrangebyrank(key, 0, -limit - 1)
    pipe.expire(key, window_seconds)
    _, _, count, _, _ = await pipe.execute()

    if count > limit:
        log.warning("Rate limit exceeded", extra={"user_id": user_id})
        raise HTTPException(
            429,
            detail={
                "error":       "rate_limit_exceeded",
                "retry_after": window_seconds,
                "limit":       limit,
            },
        )
