from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from sqlalchemy import text

from app.config import settings

CheckPayload = dict[str, Any]
CheckFn = Callable[[], Awaitable[None]]


async def _check_postgres() -> None:
    from app.database import engine

    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))


async def _check_redis() -> None:
    from app.redis_client import get_redis

    redis = await get_redis()
    try:
        await redis.ping()
    finally:
        if hasattr(redis, "aclose"):
            await redis.aclose()


async def _check_minio() -> None:
    from app.storage import bucket_exists

    exists = await bucket_exists(settings.MINIO_BUCKET)
    if not exists:
        raise RuntimeError(f"MinIO bucket '{settings.MINIO_BUCKET}' does not exist")


async def _check_chroma() -> None:
    if settings.CHROMA_MODE == "local":
        # In-process Chroma: prove the client opens (no HTTP heartbeat).
        from app.retrieval.memory.vector_store import _get_sync_client

        if _get_sync_client() is None:
            raise RuntimeError("local Chroma client unavailable")
        return
    url = f"http://{settings.CHROMA_HOST}:{settings.CHROMA_PORT}/api/v2/heartbeat"
    async with httpx.AsyncClient(timeout=2.0) as client:
        response = await client.get(url)
        response.raise_for_status()


async def _check_sqlite() -> None:
    from app.database import engine

    async with engine.connect() as conn:
        await conn.exec_driver_sql("SELECT 1")


async def _check_storage() -> None:
    """Backend-agnostic storage check (MinIO bucket or fs root)."""
    from app.storage import bucket_exists

    if not await bucket_exists(settings.MINIO_BUCKET):
        raise RuntimeError(f"Storage backend not ready ({settings.STORAGE_BACKEND})")


async def _check_mcp_hub() -> None:
    """Prove the MCP hub is wired: flag on + SDK app constructible.

    Deliberately offline — a deep handshake lives in the integration suite;
    readiness only asserts the mount exists so `docker compose` operators
    see a failed check the moment the hub is misconfigured.
    """
    if not settings.MCP_HUB_ENABLED:
        raise RuntimeError("MCP hub disabled (MCP_HUB_ENABLED=false)")
    from app.mcp_hub.server import get_mcp_app

    app = get_mcp_app()
    if not callable(app):
        raise RuntimeError("MCP app is not callable")


def _sanitize_error(error: Exception) -> str:
    message = str(error).replace("\n", " ").strip()
    if not message:
        message = error.__class__.__name__
    return message[:300]


async def _measure(name: str, checker: CheckFn) -> tuple[str, CheckPayload]:
    started = time.perf_counter()
    try:
        await checker()
        return name, {
            "status": "ok",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    except Exception as exc:
        return name, {
            "status": "failed",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "error": _sanitize_error(exc),
        }


def _default_readiness_checkers() -> dict[str, CheckFn]:
    """Deployment-aware readiness checks.

    Postgres (full stack): postgres + redis + minio + chroma-http + mcp_hub.
    SQLite (lite mode):   sqlite + redis(memory) + storage(fs) + chroma-local
    + mcp_hub.
    """
    from app.database import IS_SQLITE

    if IS_SQLITE:
        checkers = {
            "sqlite": _check_sqlite,
            "redis": _check_redis,
            "storage": _check_storage,
            "chroma": _check_chroma,
        }
    else:
        checkers = {
            "postgres": _check_postgres,
            "redis": _check_redis,
            "minio": _check_minio,
            "chroma": _check_chroma,
        }
    if settings.MCP_HUB_ENABLED:
        # Flag-off is a deliberate configuration, not a degraded service —
        # the check (and its failure signal) only exists while enabled.
        checkers["mcp_hub"] = _check_mcp_hub
    return checkers


async def run_readiness_checks(
    extra_checkers: dict[str, CheckFn] | None = None,
) -> dict[str, CheckPayload]:
    checkers = _default_readiness_checkers()
    if extra_checkers:
        checkers.update(extra_checkers)
    results = await asyncio.gather(
        *[_measure(name, checker) for name, checker in checkers.items()]
    )
    return dict(results)


async def check_readiness() -> CheckPayload:
    checks = await run_readiness_checks()
    status = "ok" if all(check["status"] == "ok" for check in checks.values()) else "degraded"
    return {
        "status": status,
        "version": "1.1.0",
        "checks": checks,
    }
