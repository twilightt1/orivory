# The Orivory image — the whole memory hub in one container: API + MCP server,
# SQLite storage, in-process Qdrant, in-memory caches, eager tasks, local
# file uploads. Zero external services.
#
#   docker build -t orivory:lite .
#   docker run -d -p 8000:8000 -v orivory-data:/data \
#     orivory:lite
#
# The Next.js frontend is NOT included (Orivory targets agents via /mcp + API).
# This is the ONLY Dockerfile: the legacy full-stack one (libpq/Qdrant server/
# Redis/MinIO era) was folded into this file on the slim branch.

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATABASE_URL=sqlite+aiosqlite:////data/orivory.db \
    QDRANT_MODE=local \
    QDRANT_LOCAL_PATH=/data/qdrant \
    LOCAL_E5_DIR=/data/models/e5 \
    STORAGE_BACKEND=fs \
    FS_STORAGE_PATH=/data/uploads

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt \
    # ponytail: gcc only builds wheels; purge it (~200MB) — runtime needs none.
    # R36: the old `pip uninstall -y kubernetes` went WITH chromadb — kubernetes
    # was chromadb's dependency (verified: `packages requiring kubernetes:
    # [chromadb]`) and nothing imports it. qdrant-client brings grpcio/httpx/
    # numpy/portalocker/protobuf/pydantic/urllib3 — no fat unused extra to strip.
    && apt-get purge -y gcc && apt-get autoremove -y

RUN groupadd --system app && useradd --system --gid app --home-dir /app app \
    && mkdir -p /data && chown -R app:app /data /app

COPY --chown=app:app . .

USER app

VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=5 \
    CMD curl -fs http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
