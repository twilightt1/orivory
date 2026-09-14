from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette import status

from app.api.v1.router import api_router
from app.config import settings
from app.middleware.logging_middleware import LoggingMiddleware

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting RAG backend", environment=settings.ENVIRONMENT, lite=settings.LITE_MODE)
    if settings.DATABASE_URL.startswith("sqlite"):
        # Lite mode: versioned fresh-install bootstrap; an existing unversioned
        # SQLite schema fails closed until a reviewed migration is applied.
        from app.database import bootstrap_sqlite
        await bootstrap_sqlite()
        log.info("SQLite schema bootstrapped")
    try:
        from app.storage import ensure_bucket
        await ensure_bucket()
        log.info("Storage ready (backend=%s)", settings.STORAGE_BACKEND)
    except Exception as e:
        log.warning("Storage init failed", error=str(e))
    if settings.MCP_HUB_ENABLED:
        # Starlette does not run a mounted app's lifespan, so the host lifespan
        # must run the MCP session manager itself (see app/mcp_hub/server.py).
        from app.mcp_hub.server import build_mcp_server

        async with build_mcp_server().session_manager.run():
            log.info("MCP hub ready", path="/mcp")
            yield
    else:
        yield
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
