from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from app.config import settings

CheckPayload = dict[str, Any]
CheckFn = Callable[[], Awaitable[None]]


async def _check_redis() -> None:
    from app.redis_client import get_redis

    redis = await get_redis()
    try:
        await redis.ping()
    finally:
        if hasattr(redis, "aclose"):
            await redis.aclose()


async def _check_qdrant() -> None:
    """Vector readiness.

    Lite mode runs the vector store in-process, so readiness proves the
    process's ONE embedded Qdrant owner client opens (no server to poll) —
    the same client the store itself uses, never a second one on the folder.
    Server mode asks Qdrant's own ``/readyz`` (2s bound): one HTTP call that
    answers 200 only once the shards are ready.
    """
    from app.retrieval import vector_backend

    if vector_backend.is_local_mode():
        # The Qdrant owner client IS the vector store in lite mode: opening it
        # is the real check — the open raises (through the owner executor) if
        # the folder is unreadable, and it is the SAME client the store uses,
        # so a second one on the folder is impossible by construction and lite
        # starts no server for an HTTP probe to dial. There is no "is it None"
        # branch to take: a failed open raises rather than returning None.
        vector_backend.get_sync_client()
        return
    url = f"{settings.QDRANT_URL.rstrip('/')}/readyz"
    async with httpx.AsyncClient(timeout=2.0) as client:
        response = await client.get(url)
        response.raise_for_status()


async def _check_sqlite() -> None:
    from app.database import engine

    async with engine.connect() as conn:
        await conn.exec_driver_sql("SELECT 1")


async def _check_storage() -> None:
    """Storage check: the fs root must exist (it is created on demand)."""
    from app.storage import bucket_exists

    if not await bucket_exists():
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
    """The one-container readiness checks: sqlite + redis(memory) + storage(fs)
    + qdrant (+ mcp_hub while enabled)."""
    checkers = {
        "sqlite": _check_sqlite,
        "redis": _check_redis,
        "storage": _check_storage,
        "qdrant": _check_qdrant,
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
