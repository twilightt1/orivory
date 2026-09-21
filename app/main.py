import asyncio
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
from app.retrieval.memory.outbox import IndexFreshnessTimeout
from app.retrieval.vector_retriever import VectorUnavailableError

log = structlog.get_logger()

#: click's BOOL true-values, which is what parses uvicorn's ``--reload`` flag
#: (and therefore its ``UVICORN_RELOAD`` env fallback). Everything else — the
#: false spellings AND the empty string — is OFF.
_RELOAD_TRUE_VALUES = frozenset({"1", "true", "t", "yes", "y", "on"})

# The boot replays ONE bounded batch of intents before serving: the background
# loop's first tick can be a whole interval away, and a booting app should not
# make a user wait for it. Everything after that batch is
# app/retrieval/memory/drain_loop.py's — on both dialects (P3).


async def _drain_index_outbox_at_boot() -> None:
    """Replay one batch of pending intents at boot — SQLite and Postgres alike.

    Bounded and failure-tolerant: a vector outage (or any drain-level error)
    must never keep the app from booting, so it is logged and the intents stay
    pending with backoff for the background loop. The report is logged for
    observability.
    """
    from app.retrieval.memory.drain_loop import _should_drain, drain_once

    if not _should_drain():
        return
    try:
        report = await drain_once(batch_size=settings.OUTBOX_DRAIN_BATCH_SIZE)
        log.info("Index outbox boot drain", **report)
    except Exception as e:
        log.warning("Index outbox boot drain failed", error=str(e))


async def _warm_embedder_at_boot() -> None:
    """Build the local embedding session before the drain and the API (P2/T1).

    Both the boot drain and the first request embed, and a cold session is
    610-685 ms (plus C-level parse lag even inside a thread): the lifespan pays
    it while nothing is served yet. Best-effort — see ``warmup_embedder``.
    """
    if not settings.EMBED_WARMUP_ON_BOOT:
        return
    from app.retrieval.embedder import warmup_embedder

    await warmup_embedder()


def _requested_processes() -> int:
    """How many app processes the launcher asked for (1 = a single owner).

    ``uvicorn --reload``/``--workers N`` re-launch the app in each process with
    the same argv; ``WEB_CONCURRENCY`` and ``UVICORN_WORKERS`` are uvicorn's /
    Gunicorn's worker counts — uvicorn reads ``--workers`` from
    ``UVICORN_WORKERS`` through its ``UVICORN_`` envvar prefix, so the env var
    alone is a multi-process launcher. ``UVICORN_RELOAD`` is uvicorn's env
    fallback for the ``--reload`` flag: 1/true/t/yes/y/on mean ON, and 'false',
    '0', 'no', 'off', 'f', 'n' and the empty string mean OFF — exactly the
    spellings click's BOOL accepts. Garbage is NOT parity: click exits 2 on it
    ('maybe', '2'), here it reads as OFF, because a launcher typo must not
    become a boot refusal the operator cannot explain.
    """
    if (
        os.environ.get("UVICORN_RELOAD", "").strip().casefold() in _RELOAD_TRUE_VALUES
        or "--reload" in sys.argv
    ):
        return 2  # a reloader: a supervisor plus the app it restarts
    for source in ("WEB_CONCURRENCY", "UVICORN_WORKERS"):
        workers = os.environ.get(source, "").strip()
        if workers.isdigit() and int(workers) > 1:
            return int(workers)
    for index, token in enumerate(sys.argv):
        if token.startswith("--workers="):
            value = token.split("=", 1)[1]  # the single-token spelling
        elif token == "--workers":
            value = sys.argv[index + 1] if index + 1 < len(sys.argv) else ""
        else:
            continue
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
    log.info("Starting RAG backend", environment=settings.ENVIRONMENT)
    # Versioned fresh-install bootstrap; an existing unversioned SQLite schema
    # fails closed until a reviewed migration is applied.
    from app.database import bootstrap_sqlite
    await bootstrap_sqlite()
    log.info("SQLite schema bootstrapped")

    # The install's ONE identity, created (or reused) at boot so the boot
    # drain and the first request already have the owner row every per-user
    # path keys on. Not part of the schema ladder: a pinned older binary must
    # still boot against an older database (app/database.py:bootstrap_sqlite).
    from app.database import AsyncSessionLocal
    from app.services.local_owner import ensure_local_owner

    async with AsyncSessionLocal() as session:
        owner = await ensure_local_owner(session)
    log.info("Local owner ready", email=owner.email)
    from app.retrieval.memory.drain_loop import start_drain_loop, stop_drain_loop

    # The session is built BEFORE the boot drain: the drain embeds too, and a
    # cold InferenceSession's 610-685 ms belongs to the boot, not to a request.
    await _warm_embedder_at_boot()
    await _drain_index_outbox_at_boot()
    # Created INSIDE the try whose ``finally`` stops it: storage init can raise
    # a BaseException (a shutdown cancel), and one raised outside the try would
    # leak a running drain task into teardown — and skip ``close_clients()``.
    drain_task: asyncio.Task | None = None
    try:
        drain_task = await start_drain_loop()
        try:
            from app.storage import ensure_bucket
            await ensure_bucket()
            log.info("Storage ready (backend=%s)", settings.STORAGE_BACKEND)
        except Exception as e:
            log.warning("Storage init failed", error=str(e))
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
        # Stop the drain loop first: in local mode its vector writes hold the
        # storage folder, and closing the client under a live batch would fail
        # that batch for nothing (it stays pending with backoff).
        await stop_drain_loop(drain_task)
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


# Typed readiness errors: an embedding contract mismatch, an unreachable
# vector store, or a recall that waited out its freshness budget for a write
# still in flight must answer 503 with a machine-readable body — never an
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


@app.exception_handler(IndexFreshnessTimeout)
async def _index_freshness_timeout_handler(
    _request: Request, exc: IndexFreshnessTimeout
) -> JSONResponse:
    """A recall waited out its budget for this tenant's pending index intents.

    ``results: []`` would be a false no-match for a memory that was just
    written, so this is a readiness answer instead.
    """
    log.warning("Recall freshness budget exhausted", error=str(exc))
    return JSONResponse(
        {"error": "index_freshness_timeout"},
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
