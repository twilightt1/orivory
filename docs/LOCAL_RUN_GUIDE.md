# 💻 Local Development Guide

This guide explains how to run **Orivory** locally — the one container, which
needs no Docker-backed infrastructure — and how to diagnose the most common
failures.

## 🛠️ Prerequisites

- Python 3.13 (see `.python-version`)
- Docker and Docker Compose
- Git

---

## 🚀 Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/twilightt1/orivory.git
cd orivory
```

### 2. Create a virtual environment

The repo convention is `.venv`:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

---

## ⚙️ Configuration

### 1. Environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in the provider keys you use (OpenRouter, OpenAI, Jina —
all optional; embeddings and storage run locally out of the box).

> [!WARNING]
> Do not commit `.env`. Keep API keys local.

### 2. Start the container (or run the app directly)

There is no infrastructure to start: `docker-compose.yml` is ONE service —
`app` — which builds the repo's `Dockerfile`, binds `./data` to `/data`,
pins the store settings (`sqlite+aiosqlite:////data/orivory.db`,
`QDRANT_MODE=local`, `STORAGE_BACKEND=fs`), publishes the API on
`127.0.0.1:8000` and polls `/health`.

```bash
docker compose up -d
```

For production-like validation, use
[docker-compose.prod.yml](../docker-compose.prod.yml) with the deployment docs:

- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- [OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md)
- [BACKUP_RESTORE.md](BACKUP_RESTORE.md)

Check container health:

```bash
docker compose ps
```

### 3. No migration step

There is none to run: the app executes the versioned SQLite ladder
(`bootstrap_sqlite()`) during its lifespan, before it serves traffic. An older
database is upgraded on boot; a NEWER schema than the image knows is refused
loudly rather than repaired.

---

## 🏃 Running the System

### 1. Start the API server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Or:

```bash
make dev
```

Open Swagger UI at <http://localhost:8000/docs>.

### 2. Check liveness and readiness

`/health` is a lightweight liveness endpoint:

```bash
curl http://localhost:8000/health
```

`/ready` checks the in-container dependencies (SQLite, the in-memory store,
filesystem storage, the embedded Qdrant, and the MCP hub when enabled):

```bash
curl http://localhost:8000/ready
```

Expected healthy response:

```json
{
  "status": "ok",
  "version": "1.1.0",
  "checks": {
    "sqlite": {"status": "ok", "latency_ms": 1.2},
    "redis": {"status": "ok", "latency_ms": 2.1},
    "storage": {"status": "ok", "latency_ms": 3.4},
    "qdrant": {"status": "ok", "latency_ms": 8.7},
    "mcp_hub": {"status": "ok", "latency_ms": 0.4}
  }
}
```

`version` is the app build (`1.1.0`); the `checks` map is the one-container set
(`redis` is the in-process `InMemoryRedis`, `qdrant` the embedded store, and
`mcp_hub` only while `MCP_HUB_ENABLED=true`). If any check fails, `/ready`
returns `503` with `status: degraded` and an `error` on the failing check.

### 3. Nothing else to start

Ingestion and the background index drain run INSIDE the API process (P3):
there is no worker, beat, broker or queue to launch and none to watch. Uploads
are ingested inline, and the writes that did not land are replayed by the
outbox drain loop (`OUTBOX_DRAIN_INTERVAL_SECONDS`, default 5s) — memory recall
waits for its own tenant's pending writes, bounded by
`RECALL_FRESHNESS_BUDGET_SECONDS`, and answers a typed 503 rather than an empty
result when they cannot land. See
[OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md#background-indexing-p3).

---

## 🧪 Testing & Quality

### CI-safe tests that do not need infrastructure

```bash
.venv/bin/python -m pytest --confcutdir=tests/api tests/api/test_health_api.py tests/api/test_admin_diagnostics.py -q
.venv/bin/python -m pytest --confcutdir=tests/services tests/services/test_health_service.py tests/services/test_diagnostics_service.py -q
.venv/bin/python -m pytest --confcutdir=tests/rag tests/rag -q
.venv/bin/python -m pytest --confcutdir=tests/eval tests/eval/test_eval_metrics.py -q
.venv/bin/python -m pytest --confcutdir=tests/config tests/config/test_settings_validation.py -q
```

Run the deterministic RAG evaluation report:

```bash
.venv/bin/python eval/run_eval.py --mode offline --output-dir eval/results --top-k 5
```

Reports are written to [latest_report.md](../eval/results/latest_report.md) and [latest_report.json](../eval/results/latest_report.json).

These tests use mocks/monkeypatching and do not need any service (SQLite and
the embedded store included) or external LLM/API credentials.

### Live integration tests

Live tests exercise a really running API (its own SQLite + embedded Qdrant).
They are marked `requires_infra` and skipped unless `RUN_LIVE_INTEGRATION=1`
is set.

```bash
cp .env.test.example .env.test
docker compose up -d
export RUN_LIVE_INTEGRATION=1
.venv/bin/python -m pytest --confcutdir=tests/integration tests/integration -q
```

To clean up the local service volumes afterwards:

```bash
docker compose down -v
```

### Live API RAG evaluation — removed

The `--mode live-api` lane drove the chat API (`/api/v1/chat/*`), which was
deleted with the full-stack surface; `eval/live_api_eval.py` and its tests
went with it. The deterministic offline lane above is the whole evaluation
harness.

### Full test suite

The full suite needs no external service: [tests/conftest.py](../tests/conftest.py) points `DATABASE_URL` at a private
SQLite file before the app is imported, and the live modules skip themselves
unless `RUN_LIVE_INTEGRATION=1` is set.

```bash
.venv/bin/python -m pytest tests -q
```

### Linting and formatting

Full-repo lint (what CI runs — a module is linted because it is in the tree,
not because it was added to a list):

```bash
.venv/bin/python -m ruff check app tests eval scripts
```

Auto-fix safe issues:

```bash
.venv/bin/python -m ruff check app tests eval scripts --fix
```

Run the security readiness gate (CI-executable):

```bash
.venv/bin/python scripts/security_check.py
```

Format:

```bash
.venv/bin/python -m ruff format app/
```

Or:

```bash
make lint
make format
make security-check
```

---

## 🛠️ Troubleshooting

### `/ready` says `degraded`

Read the failing check by name — each one names its own cause:

- `sqlite` — the `DATABASE_URL` path is not writable. The container's `app`
  user needs a writable `/data`; on a Linux host bind-mounting `./data`, run
  `mkdir -p data && chmod 777 data` once before the first boot.
- `qdrant` — the embedded store owns its folder exclusively, so a second
  process on the same `QDRANT_LOCAL_PATH` fails to open it. Run ONE app process
  (keep `--workers 1`), and stop any other process holding the folder (an
  offline `scripts/migrate_qdrant.py` run owns it too).
- `storage` — `FS_STORAGE_PATH` is missing or not writable; it is created on
  demand under `/data/uploads` in the image.
- `redis` — the check pings the process-local `InMemoryRedis`. A failure here
  means the process is broken, not that a server is down (there is no Redis
  server to start).
- `mcp_hub` — `MCP_HUB_ENABLED=true` but the hub app could not be built. Set it
  to `false` to run the REST surface alone, or check the logs for the import
  error.

```bash
docker compose ps
docker compose logs app --tail=100
curl http://localhost:8000/ready
```

### A second process dies on the vector folder

The embedded Qdrant store is exclusive: the second process to touch
`QDRANT_LOCAL_PATH` fails on its first write. That is why the app refuses to
boot when the launcher asks for more than one process, and why
`docker-compose.prod.yml` pins `--workers 1`. Stop the other process — or the
offline migration CLI — before starting a second one.

### The first boot looks stuck on the model

Boot warms the local ONNX embedding session before it serves anything, which on
a fresh volume downloads `snowflake-arctic-embed-xs` (~90 MB) with no timeout.
Pre-seed the cache (persist `/data/models`) or set `EMBED_WARMUP_ON_BOOT=false`
to pay a cold session (~610-685 ms) on the first request instead. See
[LITE_MODE.md](LITE_MODE.md).

### Recall answers `503`

`POST /api/v1/memories/recall` answers a typed `503` instead of a silent empty
result (`{"error": "..."}`): `index_freshness_timeout` means a write for this
tenant is still in flight and the recall waited out
`RECALL_FRESHNESS_BUDGET_SECONDS`; `vector_unavailable` means the store could
not be reached; `embedding_contract_mismatch` means the stored generation and
the configured embedding contract disagree. On SQLite the recall answers from
the FTS5 lexical leg instead of 503-ing on a vector outage, with
`trace.counts["lexical"]` set. See
[OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md).

### Ingestion failed after an outage

1. Check readiness (`/ready` above) — the drain loop retries what it can.
2. Re-upload the document, or re-run the import that failed; already-imported
   items are skipped per `(user, source_type, source_ref)`, so a retry does not
   duplicate them.

The ingestion path records the failing stage in `Document.error_msg`
(`storage_read`, the lexical-index build, or the vector upsert), and anything a
write enqueued lands in `index_outbox` with backoff rather than being dropped —
the drain loop replays it. See
[OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md).
