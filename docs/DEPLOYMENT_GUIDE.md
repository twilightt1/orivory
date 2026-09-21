# Deployment Guide

This guide describes a production-like Docker Compose deployment for Orivory.
Local development should continue using [docker-compose.yml](../docker-compose.yml). Production-like deployments should combine it with [docker-compose.prod.yml](../docker-compose.prod.yml).

## Required Services

One container, zero external services:

- the FastAPI app + MCP server process
- SQLite (`/data/orivory.db`) — the canonical store
- embedded Qdrant (`/data/qdrant`, in-process)
- in-memory caches / rate limits (no Redis server)
- filesystem uploads (`/data/uploads`)

## Environment Setup

Copy the example environment and replace every placeholder before deployment:

```bash
cp .env.example .env
```

> **Note:** there is no migration service and no Alembic any more. The app runs
> the versioned SQLite ladder (`bootstrap_sqlite()`) itself during the lifespan,
> before anything is served; an older database is upgraded on boot and a NEWER
> schema is refused loudly.

Production must use:

- `ENVIRONMENT=production`
- a random `JWT_SECRET_KEY` with at least 32 characters
- explicit `ALLOWED_ORIGINS`, never `*`
- real provider keys for `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, and `JINA_API_KEY`

The published port binds loopback only (`127.0.0.1:8000`) — put a reverse proxy
in front of it to expose the API. Also set when they differ from the defaults:
`FS_STORAGE_PATH`, `QDRANT_LOCAL_PATH` and `APP_PORT` (the API's listen port —
the P1b migration CLI probes it, plus `migrate.lock`, to refuse to run while
the app is alive).

The app validates these guardrails at startup in production mode.

## Refresh Token Rotation

Refresh tokens are stored as SHA-256 hashes under `refresh:{hash}` in the
process-local `InMemoryRedis`. A per-user index set (`refresh_user:{user_id}`)
lets the application revoke every active session for a user in
O(N_user_tokens) without scanning the full `refresh:*` keyspace. Consequences:

- a container restart drops the store, so every user re-authenticates — the
  correct behaviour for a secret-bearing store, and nothing to back up.
- `REFRESH_TOKEN_EXPIRE_DAYS` still bounds a session in the database; the
  in-memory index only carries the revocation state.

## Validate Compose Config

```bash
docker compose config --quiet
docker compose -f docker-compose.yml -f docker-compose.prod.yml config --quiet
```

## Build Image

```bash
docker build -t Orivory-api:latest .
```

The [Dockerfile](../Dockerfile) runs the app as a non-root user and defaults to a production `uvicorn` command. Development Compose can still override this with `--reload`.

## Start Production-like Stack

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

View service status:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps
```

## Database Migrations

There is no migration step to run: the app executes the versioned SQLite
ladder (`bootstrap_sqlite()`) during startup, before it serves traffic. Watch
the container logs for the ladder version it lands on, and keep the
pre-upgrade `/data` archive from [BACKUP_RESTORE.md](BACKUP_RESTORE.md) until
the new image reports `/ready` ok.

## Health Checks

Check the API liveness endpoint:

```bash
curl -fsS http://localhost:8000/health
```

Check dependency readiness:

```bash
curl -fsS http://localhost:8000/ready
```

`/ready` returns HTTP 503 when any dependency is degraded.

For authenticated operational checks, use the admin diagnostics endpoint:

```bash
curl -fsS -H "Authorization: Bearer $ADMIN_ACCESS_TOKEN" \
  http://localhost:8000/api/v1/admin/diagnostics
```

Diagnostics includes dependency readiness, ingestion status and the
`index_outbox` summary (pending / done / blocked intents — see
[OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md#background-indexing-p3)), but it
must remain admin-only.

## Reverse Proxy Notes

Put a reverse proxy in front of the API for HTTPS and request buffering control.

### Nginx sketch

```nginx
server {
  listen 443 ssl http2;
  server_name api.example.com;

  location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto https;
  }

  location /api/v1/chat/ {
    proxy_pass http://127.0.0.1:8000;
    proxy_buffering off;
    proxy_cache off;
  }
}
```

### Caddy sketch

```caddyfile
api.example.com {
  reverse_proxy 127.0.0.1:8000
}
```

## Post-deploy Smoke Checks

Run:

```bash
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/ready
curl -fsS -H "Authorization: Bearer $ADMIN_ACCESS_TOKEN" \
  http://localhost:8000/api/v1/admin/diagnostics
python eval/run_eval.py --mode offline --output-dir eval/results --top-k 5
```

If a test user and provider keys are available, run live API eval:

```bash
python eval/run_eval.py --mode live-api \
  --api-base-url http://localhost:8000 \
  --access-token "$ACCESS_TOKEN" \
  --sample-docs sample_docs \
  --output-dir eval/results
```

## Stop Stack

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml down
```

Use `down -v` only when intentionally deleting persistent data.
