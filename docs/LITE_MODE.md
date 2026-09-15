# Lite mode — run Orivory in one container

> **Why:** our own research ranked setup friction as a top-3 abandonment
> cause for self-hosted tools. Competitors boot in one command; the full
> Orivory stack is 10 services. Lite mode is the answer for personal use:
> one container, one volume, zero external services.

```bash
docker run -d --name orivory -p 8000:8000 -v orivory-data:/data \
  -e OPENAI_API_KEY=sk-... ghcr.io/twilightt1/orivory:lite
```

| What runs inside | What's replaced (vs full stack) |
|---|---|
| FastAPI API + MCP server (`/mcp`) | Postgres → **SQLite** (WAL, FK enforced) |
| In-process Qdrant (persistent, `/data/qdrant`) | Redis → **in-memory** fallback (caches, rate limits) |
| Ingestion + the P3 index drain run in-process | worker/beat/broker → **not needed** (the slim branch has none) |
| Filesystem uploads (`/data/uploads`) | MinIO → **local FS** |

The Next.js frontend is not in the lite image — lite targets AI agents via
`/mcp` and the REST API (that's the primary use: your agent gets a persistent
brain). Point Claude Desktop / Cursor / OpenClaw at
`http://localhost:8000/mcp` with a token from
`POST /api/v1/agents` (see the main [README](../README.md)).

## What lite mode trades away

> **Not a production tier.** Lite is the demo/personal tier: one container
> means one blast radius (OOM anywhere loses everything in flight), eager
> tasks have no retry queue, the cost ledger is single-process SQLite, and
> in-memory rate limits reset on restart. Anything multi-user,
> multi-instance, or load-bearing belongs on the full stack.

- **Single-user, single-instance** — SQLite + in-memory caches don't do
  horizontal scale. For teams or heavy agents, use the full compose stack.
- **JWT secret is ephemeral** — auto-generated per container; users re-login
  after an upgrade unless they set `JWT_SECRET_KEY` explicitly.
- **No task queue** — work runs inline in the API process; a crash mid-task
  loses that task (fine: SQL is truth, and anything a write enqueued into
  `index_outbox` is replayed by the P3 drain loop on the next run).
- **The first boot may download the embedding model.** Lite warms the local
  ONNX session during the lifespan — a REAL inference, before the boot drain and
  before anything is served — and on a fresh volume that means downloading
  `snowflake-arctic-embed-xs` (~90 MB) first. That download has **no timeout**,
  so a slow or blocked network extends boot; pre-seed the model cache (persist
  `/data/models` or `~/.cache/orivory/e5` on the volume) when that matters.
  `EMBED_WARMUP_ON_BOOT=false` skips the boot warm-up and pays a cold session
  (~610-685 ms) on the first request instead.

## Full stack still exists

`docker compose up -d` (Postgres + Qdrant behind the app)
is unchanged and remains the path for teams and production. Lite and full
share the same code paths — `LITE_MODE=1` only swaps the drivers.

## Implementation map

| Concern | Full stack | Lite mode |
|---|---|---|
| Database | Postgres (`postgresql+asyncpg://`) + Alembic | SQLite (`sqlite+aiosqlite://`) + `bootstrap_sqlite()` |
| UUID columns | `GUID` type (coerces str→UUID, asyncpg-safe) | same |
| Vectors | Qdrant container (`QDRANT_URL`) | embedded Qdrant (`QDRANT_MODE=local`, `QDRANT_LOCAL_PATH`) |
| Cache/rate-limit | Redis | `InMemoryRedis` (process-local) |
| Tasks | inline in the API process | same (in-process, no broker) |
| Uploads | MinIO | filesystem under `FS_STORAGE_PATH` |

Model column types are cross-dialect now (`sa.Uuid`-derived `GUID`, `JSON`
instead of `ARRAY`/`JSONB`); the Postgres-only GIN index and cast
server-defaults live exclusively in the Alembic migrations, which stay
Postgres-only. SQLite deployments bootstrap from model metadata
(`bootstrap_sqlite()`).

## Verified

E2E in the built container: register → verify → login → agent client
registration → MCP `initialize` (200) → `add_memory` → `search_memory` —
all green, `/ready` reports `sqlite/redis/storage/qdrant/mcp_hub` all ok.
