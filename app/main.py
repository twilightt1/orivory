import os
import sys
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette import status

from app.api.v1.router import api_router
from app.config import settings
from app.middleware.logging_middleware import LoggingMiddleware
from app.retrieval.embedder import EmbeddingDimensionMismatch
from app.retrieval.vector_retriever import VectorUnavailableError

log = structlog.get_logger()

# A boot replays at most this many drain batches of 50 intents. A crash-time
# backlog heals over a few starts without delaying readiness; anything left
# stays pending in SQLite for the next boot (a background loop is P3's, and is
# deliberately absent here).
_STARTUP_DRAIN_BATCHES = 2
_STARTUP_DRAIN_BATCH_SIZE = 50


async def _drain_index_outbox_on_startup() -> None:
    """Replay pending index intents at boot — SQLite deployments only.

    Mirrors the ``bootstrap_sqlite`` guard. Bounded and failure-tolerant: a
    vector outage (or any drain-level error) must never keep the app from
    booting, so it is logged and the intents stay pending with backoff for the
    next start. The report of every batch is logged for observability.
    """
    if not settings.DATABASE_URL.startswith("sqlite"):
        return
    from app.retrieval.memory.outbox import drain_pending

    try:
        for _ in range(_STARTUP_DRAIN_BATCHES):
            report = await drain_pending(batch_size=_STARTUP_DRAIN_BATCH_SIZE)
            log.info("Index outbox startup drain", **report)
            if report.get("claimed", 0) < _STARTUP_DRAIN_BATCH_SIZE:
                break  # nothing left: a second batch would claim zero
    except Exception as e:
        log.warning("Index outbox startup drain failed", error=str(e))


def _requested_processes() -> int:
    """How many app processes the launcher asked for (1 = a single owner).

    ``uvicorn --reload``/``--workers N`` re-launch the app in each process with
    the same argv; ``WEB_CONCURRENCY`` is uvicorn's/Gunicorn's worker count and
    ``UVICORN_RELOAD`` is uvicorn's env fallback for the ``--reload`` flag.
    """
    if os.environ.get("UVICORN_RELOAD", "").strip() or "--reload" in sys.argv:
        return 2  # a reloader: a supervisor plus the app it restarts
    workers = os.environ.get("WEB_CONCURRENCY", "").strip()
    if workers.isdigit() and int(workers) > 1:
        return int(workers)
    if "--workers" in sys.argv:
        index = sys.argv.index("--workers")
        value = sys.argv[index + 1] if index + 1 < len(sys.argv) else ""
        if value.isdigit() and int(value) > 1:
            return int(value)
    return 1


def _refuse_multi_owner_local_qdrant() -> None:
    """Local Qdrant is owned by exactly ONE process (spec §3.1).

    An embedded storage folder is exclusive: the second process to touch it dies
    with qdrant-client's raw "already accessed" RuntimeError at its first write.
    Refuse the boot instead — before anything is served — when the launcher
    asked for more than one process while ``QDRANT_MODE=local``.
    """
    from app.retrieval.vector_backend import is_local_mode

    if not is_local_mode():
        return
    processes = _requested_processes()
    if processes > 1:
        raise RuntimeError(
            f"QDRANT_MODE=local owns the storage folder {settings.QDRANT_LOCAL_PATH} "
            f"exclusively, but the launcher asked for {processes} processes: run a "
            "single process, or set QDRANT_MODE=server with a Qdrant server"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _refuse_multi_owner_local_qdrant()
    log.info("Starting RAG backend", environment=settings.ENVIRONMENT, lite=settings.LITE_MODE)
    if settings.DATABASE_URL.startswith("sqlite"):
        # Lite mode: versioned fresh-install bootstrap; an existing unversioned
        # SQLite schema fails closed until a reviewed migration is applied.
        from app.database import bootstrap_sqlite
        await bootstrap_sqlite()
        log.info("SQLite schema bootstrapped")
    await _drain_index_outbox_on_startup()
    try:
        from app.storage import ensure_bucket
        await ensure_bucket()
        log.info("Storage ready (backend=%s)", settings.STORAGE_BACKEND)
    except Exception as e:
        log.warning("Storage init failed", error=str(e))
    try:
        if settings.MCP_HUB_ENABLED:
            # Starlette does not run a mounted app's lifespan, so the host
            # lifespan must run the MCP session manager itself (see
            # app/mcp_hub/server.py).
            from app.mcp_hub.server import build_mcp_server

            async with build_mcp_server().session_manager.run():
                log.info("MCP hub ready", path="/mcp")
                yield
        else:
            yield
    finally:
        # Always close the vector client(s): in local mode this is what releases
        # the storage-folder lock for the next process — or for the offline
        # migration CLI that owns the folder during a cutover.
        from app.retrieval.vector_backend import close_clients

        await close_clients()
        log.info("Shutting down")


app = FastAPI(
    title="Orivory API",
    version="1.1.0",
    description="Personal AI Second Brain — multi-source RAG, time-aware retrieval, knowledge graph, agentic actions.",
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url=None,
    lifespan=lifespan,
)

app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.ALLOWED_ORIGINS.split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


# Typed readiness errors: an embedding contract mismatch or an unreachable
# vector store must answer 503 with a machine-readable body — never an
# unhandled 500 and never a silent empty 200 (see MemoryRetriever.recall).
@app.exception_handler(EmbeddingDimensionMismatch)
async def _embedding_contract_mismatch_handler(
    _request: Request, exc: EmbeddingDimensionMismatch
) -> JSONResponse:
    log.error("Embedding contract mismatch", error=str(exc))
    return JSONResponse(
        {"error": "embedding_contract_mismatch"},
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    )


@app.exception_handler(VectorUnavailableError)
async def _vector_unavailable_handler(
    _request: Request, exc: VectorUnavailableError
) -> JSONResponse:
    log.warning("Vector store unavailable", error=str(exc))
    return JSONResponse(
        {"error": "vector_unavailable"},
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    )

if settings.MCP_HUB_ENABLED:
    from starlette.routing import Route

    from app.mcp_hub.server import get_mcp_app

    app.mount("/mcp", get_mcp_app())
    # Starlette 307-redirects an exact POST /mcp (mount root) to /mcp/, a hop
    # that can drop the Authorization header — register the same app on an
    # exact route so both /mcp and /mcp/ answer directly.
    app.router.routes.append(Route("/mcp", get_mcp_app(), name="mcp"))


@app.get("/health", tags=["health"])
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "version": "1.1.0"})


@app.get("/ready", tags=["health"])
async def ready() -> JSONResponse:
    from app.services.health_service import check_readiness

    payload = await check_readiness()
    status_code = status.HTTP_200_OK if payload["status"] == "ok" else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(payload, status_code=status_code)
