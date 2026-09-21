# One container — how Orivory runs

> **Why:** our own research ranked setup friction as a top-3 abandonment
> cause for self-hosted tools. Orivory is therefore ONE container, one
> volume, zero external services — that is not a "mode", it is the product.

```bash
docker run -d --name orivory -p 127.0.0.1:8000:8000 -v orivory-data:/data \
  -e OPENAI_API_KEY=sk-... ghcr.io/twilightt1/orivory:lite
```

The API binds **127.0.0.1 on the host** (loopback only) — put a reverse proxy
in front of it to expose it. Inside the container it listens on 0.0.0.0.

| What runs inside | What it is |
|---|---|
| FastAPI API + MCP server (`/mcp`) | the single process |
| SQLite store (WAL, FK enforced, `/data/orivory.db`) | canonical state, versioned schema ladder |
| In-process Qdrant (persistent, `/data/qdrant`) | the vector store (embedded, `QDRANT_MODE=local`) |
| Ingestion + the P3 index drain run in-process | no worker/beat/broker |
| In-memory caches + rate limits | `InMemoryRedis`, process-local (no Redis server) |
| Filesystem uploads (`/data/uploads`) | no object store (no MinIO) |

The Next.js frontend is not in the image — Orivory targets AI agents via
`/mcp` and the REST API (that's the primary use: your agent gets a persistent
brain). Point Claude Desktop / Cursor / OpenClaw at
`http://localhost:8000/mcp` with a token from
`POST /api/v1/agents` (see the main [README](../README.md)).

There is **no account auth** to run: no register, no login, no email
verification, no OAuth, no password reset, no JWT session. The install has
exactly one identity — the **local owner** (`LOCAL_OWNER_EMAIL`, default
`owner@orivory.local`), created on first boot and reused untouched afterwards.
A request with no Authorization header IS that owner; an agent token
(`oa_…`) scopes a call to one registered client and lands in the ledger.

## What the one container trades away

> **Not a horizontal-scale tier.** One container means one blast radius (OOM
> anywhere loses everything in flight), eager tasks have no retry queue, the
> cost ledger is single-process SQLite, and in-memory rate limits reset on
> restart.

- **Single-user, single-instance** — SQLite + in-memory caches don't do
  horizontal scale, and embedded Qdrant owns its folder exclusively (keep
  workers at 1).
- **One identity, no accounts** — the local owner is created on first boot.
  There is nothing to log into and no session to expire; agent tokens are the
  only credentials, and they are for scoping/auditing agents, not for
  authenticating a human.
- **No task queue** — work runs inline in the API process; a crash mid-task
  loses that task (fine: SQL is truth, and anything a write enqueued into
  `index_outbox` is replayed by the P3 drain loop on the next run).
- **The first boot may download the embedding model.** Boot warms the local
  ONNX session during the lifespan — a REAL inference, before the boot drain and
  before anything is served — and on a fresh volume that means downloading
  `snowflake-arctic-embed-xs` (~90 MB) first. That download has **no timeout**,
  so a slow or blocked network extends boot; pre-seed the model cache (persist
  `/data/models` or `~/.cache/orivory/e5` on the volume) when that matters.
  `EMBED_WARMUP_ON_BOOT=false` skips the boot warm-up and pays a cold session
  (~610-685 ms) on the first request instead.
- **Provider keys are optional** — with no embedding key the bundled local ONNX
  model is the default; OpenAI / OpenRouter / Jina keys are opt-ins.

## Implementation map

| Concern | This stack |
|---|---|
| Database | SQLite (`sqlite+aiosqlite://`) + `bootstrap_sqlite()`; app/database.py refuses any other URL |
| UUID columns | `GUID` type (coerces str→UUID, SQLite-safe) |
| Vectors | embedded Qdrant (`QDRANT_MODE=local`, `QDRANT_LOCAL_PATH`) |
| Cache/rate-limit | `InMemoryRedis` (process-local, `app/redis_client.py`) |
| Tasks | in-process, no broker |
| Uploads | filesystem under `FS_STORAGE_PATH` (`app/storage.py` is fs-only) |

Model column types are cross-dialect (`sa.Uuid`-derived `GUID`, `JSON` instead
of `ARRAY`/`JSONB`), and the schema comes from model metadata plus the versioned
SQLite ladder (`bootstrap_sqlite()`); there is no Alembic step any more.

## Verified

E2E in the built container: boot (SQLite ladder + local owner) → agent client
registration → MCP `initialize` (200) → `add_memory` → `search_memory` —
all green, `/ready` reports `sqlite/redis/storage/qdrant/mcp_hub` all ok.

Re-verified with the renamed image (2026-09-22):
`docker build -t orivory:wave-d2 .` → exit 0;
`docker run --rm orivory:wave-d2 python -c "import app.main"` →
`import app.main OK`; `docker compose config -q` → exit 0.
