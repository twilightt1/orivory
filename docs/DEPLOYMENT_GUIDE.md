# Deployment Guide

This guide describes a production-like Docker Compose deployment for Orivory.
Local development should continue using [docker-compose.yml](../docker-compose.yml). Production-like deployments should combine it with [docker-compose.prod.yml](../docker-compose.prod.yml).

## Required Services

Orivory expects these services to be available:

- FastAPI app
- Postgres
- Redis
- Qdrant
- MinIO

## Environment Setup

Copy the example environment and replace every placeholder before deployment:

```bash
cp .env.example .env
```

> **Note:** the dev `docker-compose.yml` ships a one-shot `migrate` service —
> `app` starts only after `alembic upgrade head` completes
> (the `service_completed_successfully` gate). In production, run migrations
> as a deploy step with the same guarantee.

Production must use:

- `ENVIRONMENT=production`
- a random `JWT_SECRET_KEY` with at least 32 characters
- explicit `ALLOWED_ORIGINS`, never `*`
- non-default `MINIO_ACCESS_KEY` and `MINIO_SECRET_KEY`
- real provider keys for `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, and `JINA_API_KEY`
- strong `POSTGRES_PASSWORD`

Also set when they differ from the defaults: `QDRANT_URL` (the Qdrant
endpoint the app dials; `http://qdrant:6333` inside compose) and `APP_PORT`
(the API's listen port — the P1b migration CLI probes it, plus `migrate.lock`,
to refuse to run while the app is alive).

The app validates these guardrails at startup in production mode.

## Refresh Token Rotation

Refresh tokens are stored in Redis as SHA-256 hashes under
`refresh:{hash}`. A per-user index set (`refresh_user:{user_id}`)
lets the application revoke every active session for a user in
O(N_user_tokens) without scanning the full `refresh:*` keyspace.
Confirm the Redis instance you point at:

- has `REFRESH_TOKEN_EXPIRE_DAYS` consistent with your product
  expectations (default 30 days in `Settings`).
- is reachable from the API process (it is the only process, and therefore
  the only reader or revoker of sessions).
- is backed up with the rest of the persistent data — a Redis
  wipe forces every user to re-authenticate, which is the correct
  behaviour for a secret-bearing store.

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

Run migrations after services are healthy:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec app alembic upgrade head
```

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
