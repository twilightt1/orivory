# 💻 Local Development Guide

This guide explains how to run **Orivory** locally with Docker-backed
infrastructure and how to diagnose the most common dependency failures.

## 🛠️ Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose
- Git

---

## 🚀 Setup & Installation

### 1. Clone the repository

```powershell
git clone https://github.com/twilightt1/orivory.git
cd orivory
```

### 2. Create a virtual environment

The repo convention is `.venv`:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

---

## ⚙️ Configuration

### 1. Environment variables

```powershell
copy .env.example .env
```

Open `.env` and fill in your OpenRouter, OpenAI, Jina, JWT, SendGrid, and
OAuth values as needed.

> [!WARNING]
> Do not commit `.env`. Keep API keys and JWT secrets local.

### 2. Start infrastructure

```powershell
docker compose up -d postgres qdrant migrate
```

The one-shot `migrate` service runs `alembic upgrade head` before the app
starts — with it up (exited 0), tables exist and the API is safe to boot.

`docker-compose.yml` is optimized for local development and uses source bind
mounts plus API `--reload`. For production-like validation, use
[docker-compose.prod.yml](../docker-compose.prod.yml) with the deployment docs:

- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- [OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md)
- [BACKUP_RESTORE.md](BACKUP_RESTORE.md)

Check container health:

```powershell
docker compose ps
```

### 3. Run migrations

```powershell
alembic upgrade head
```

Or:

```powershell
make migrate
```

---

## 🏃 Running the System

### 1. Start the API server

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Or:

```powershell
make dev
```

Open Swagger UI at <http://localhost:8000/docs>.

### 2. Check liveness and readiness

`/health` is a lightweight liveness endpoint:

```powershell
curl http://localhost:8000/health
```

`/ready` checks Postgres, Redis, MinIO, and Qdrant:

```powershell
curl http://localhost:8000/ready
```

Expected healthy response:

```json
{
  "status": "ok",
  "version": "1.0.0",
  "checks": {
    "postgres": {"status": "ok", "latency_ms": 10.2},
    "redis": {"status": "ok", "latency_ms": 2.1},
    "minio": {"status": "ok", "latency_ms": 15.4},
    "qdrant": {"status": "ok", "latency_ms": 8.7}
  }
}
```

If any dependency fails, `/ready` returns `503` with `status: degraded`.

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

```powershell
.\.venv\Scripts\python.exe -m pytest --confcutdir=tests/api tests/api/test_health_api.py tests/api/test_sse.py tests/api/test_admin_diagnostics.py -q
.\.venv\Scripts\python.exe -m pytest --confcutdir=tests/services tests/services/test_health_service.py tests/services/test_diagnostics_service.py -q
.\.venv\Scripts\python.exe -m pytest --confcutdir=tests/rag tests/rag -q
.\.venv\Scripts\python.exe -m pytest --confcutdir=tests/eval tests/eval/test_eval_metrics.py tests/eval/test_live_api_eval.py -q
.\.venv\Scripts\python.exe -m pytest --confcutdir=tests/config tests/config/test_settings_validation.py -q
```

Run the deterministic RAG evaluation report:

```powershell
.\.venv\Scripts\python.exe eval/run_eval.py --mode offline --output-dir eval/results --top-k 5
```

Reports are written to [latest_report.md](../eval/results/latest_report.md) and [latest_report.json](../eval/results/latest_report.json).

These tests use mocks/monkeypatching and do not need Postgres, Redis, MinIO,
Qdrant, or external LLM/API credentials.

### Live integration tests

Live tests exercise real Postgres, Redis, Qdrant, and MinIO services. They are
marked `requires_infra` and skipped unless `RUN_LIVE_INTEGRATION=1` is set.

```powershell
copy .env.test.example .env.test
docker compose up -d postgres qdrant
$env:RUN_LIVE_INTEGRATION="1"
.\.venv\Scripts\python.exe -m pytest --confcutdir=tests/integration tests/integration -q
```

To clean up the local service volumes afterwards:

```powershell
docker compose down -v
```

### Live API RAG evaluation

Live API evaluation exercises the real API, document upload/ingestion, SSE chat
streaming, returned sources, and trace metadata. Start the API first —
ingestion and the index drain live inside that process (no worker to start).
Then run one of these commands:

```powershell
.\.venv\Scripts\python.exe eval/run_eval.py --mode live-api `
  --api-base-url http://localhost:8000 `
  --email eval-user@example.com `
  --password EvalPassword123! `
  --sample-docs sample_docs `
  --output-dir eval/results
```

Or use an existing bearer token:

```powershell
.\.venv\Scripts\python.exe eval/run_eval.py --mode live-api `
  --api-base-url http://localhost:8000 `
  --access-token $env:ACCESS_TOKEN `
  --sample-docs sample_docs `
  --output-dir eval/results
```

Reports are written to `eval/results/live_api_report.md` and
`eval/results/live_api_report.json`.

### Full test suite

The full suite expects a local test database at the URL configured in
[tests/conftest.py](../tests/conftest.py).

```powershell
pytest tests/ -v
```

### Linting and formatting

Targeted lint (matches CI):

```powershell
.\.venv\Scripts\python.exe -m ruff check app/main.py app/config.py app/agents app/services/health_service.py app/services/diagnostics_service.py app/storage.py app/retrieval app/api/v1/chat.py app/api/v1/sse.py app/api/v1/admin.py eval/run_eval.py eval/live_api_eval.py eval/metrics.py eval/reporting.py tests/api/test_health_api.py tests/api/test_sse.py tests/api/test_admin_diagnostics.py tests/api/conftest.py tests/rag/test_ai_hardening.py tests/rag/conftest.py tests/services/test_health_service.py tests/services/test_diagnostics_service.py tests/services/conftest.py tests/eval/test_eval_metrics.py tests/eval/test_live_api_eval.py tests/config/test_settings_validation.py tests/integration
```

Full-repo lint (stricter, mirrors the Phase 1-3 remediation pass):

```powershell
.\.venv\Scripts\python.exe -m ruff check app tests eval scripts
```

Auto-fix safe issues:

```powershell
.\.venv\Scripts\python.exe -m ruff check app tests eval scripts --fix
```

Run the security readiness gate (CI-executable):

```powershell
.\.venv\Scripts\python.exe scripts\security_check.py
```

Format:

```powershell
ruff format app tests
```

Or:

```powershell
make lint
make format
make security-check
```

---

## 🛠️ Troubleshooting

### Qdrant connection refused

Symptom:

```text
qdrant_client.http.exceptions.ResponseHandlingException: Connection refused
```

Fix:

```powershell
docker compose up -d qdrant
docker compose logs qdrant --tail=100
curl http://localhost:6333/readyz
curl http://localhost:8000/ready
```

> [!NOTE]
> Docker Compose maps Postgres to host port `55432` by default to avoid
> collisions with a developer's local Postgres on `5432`. Container-to-container
> traffic still uses `postgres:5432`.

If you run Qdrant manually instead of Docker (lite mode runs it embedded —
no server at all):

```powershell
docker run -p 6333:6333 -v ${PWD}/qdrantdata:/qdrant/storage qdrant/qdrant
```

### Redis is unavailable

Fix:

```powershell
docker compose up -d redis
docker compose logs redis --tail=100
curl http://localhost:8000/ready
```

### MinIO bucket or credential errors

The app creates the configured bucket on startup via [ensure_bucket](../app/storage.py).
If readiness reports MinIO failed:

```powershell
docker compose up -d minio
docker compose logs minio --tail=100
```

Open the console at <http://localhost:9001> and verify credentials from `.env`.

### Postgres migration or connection errors

Fix:

```powershell
docker compose up -d postgres
docker compose logs postgres --tail=100
alembic upgrade head
curl http://localhost:8000/ready
```

### Docker service health is stuck

Check the exact failing service:

```powershell
docker compose ps
docker inspect --format='{{json .State.Health}}' orivory-postgres-1
```

If container names differ, get the name from `docker compose ps` first.

### Ingestion failed after dependency outage

1. Bring dependencies back up.
2. Confirm readiness:

```powershell
curl http://localhost:8000/ready
```

3. Retry ingestion through the admin retry endpoint or re-upload the document.

The ingestion task now records a clearer `Document.error_msg`, including the
failing stage such as `minio_read`, `redis_parent_cache`, or the vector upsert.