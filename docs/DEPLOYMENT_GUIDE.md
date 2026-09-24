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
- an explicit `CONFIG_ENCRYPTION_KEY` (Fernet; no default is derived in production)
- explicit `ALLOWED_ORIGINS`, never `*`
- real provider keys for `OPENROUTER_API_KEY` and `OPENAI_API_KEY`

The published port binds loopback only (`127.0.0.1:8000`) — put a reverse proxy
in front of it to expose the API. Also set when they differ from the defaults:
`FS_STORAGE_PATH`, `QDRANT_LOCAL_PATH` and `APP_PORT` (the API's listen port —
the P1b migration CLI probes it, plus `migrate.lock`, to refuse to run while
the app is alive).

The app validates these guardrails at startup in production mode.

The default embedding lane is local ONNX (384 dimensions). Deployments with an
existing hosted-embedding index must re-embed into a fresh collection before
cutover; stored vectors cannot be converted, and the fingerprint guard refuses
to mix contracts. Preserve the old index until the new one is verified.

## Identity

There are no accounts and no sessions to rotate. The install has ONE identity —
the local owner (`LOCAL_OWNER_EMAIL`, default `owner@orivory.local`), created
on first boot and reused untouched afterwards; a request with no Authorization
header acts as it. The only credentials are **agent tokens** (`oa_…`), minted
by `POST /api/v1/agents`, hashed at rest (SHA-256) and revocable immediately via
`DELETE /api/v1/agents/{client_id}`: revoking one stops it on the next request
and the attempt is not silently downgraded to the owner.

## Validate Compose Config

```bash
docker compose config --quiet
docker compose -f docker-compose.yml -f docker-compose.prod.yml config --quiet
```

## Build Image

```bash
docker build -t ghcr.io/twilightt1/orivory:lite .
```

The [Dockerfile](../Dockerfile) is the only Dockerfile: it carries the lite
defaults (`DATABASE_URL` on `/data/orivory.db`, `QDRANT_MODE=local`,
`STORAGE_BACKEND=fs`), runs the app as the non-root `app` user, declares
`VOLUME /data`, and ships a `HEALTHCHECK` against `/health`. Development
Compose can override the command with `--reload`.

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

There is no admin API: the install has one identity, and the REST surface it
serves is the whole surface. Operational state comes from `/ready`, the
container logs and the SQLite file itself — see
[OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md).

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

  # The MCP endpoint streams: disable buffering, and preserve Host
  # (the hub answers a non-localhost Host with 421 unless
  # MCP_HUB_ALLOWED_HOSTS names yours).
  location /mcp {
    proxy_pass http://127.0.0.1:8000;
    proxy_buffering off;
    proxy_cache off;
    proxy_set_header Host $host;
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
python eval/run_eval.py --mode offline --output-dir eval/results --top-k 5
```

The offline eval lane is the only one (`--mode offline`); it needs no server,
no keys and no database.

## Stop Stack

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml down
```

Use `down -v` only when intentionally deleting persistent data.
