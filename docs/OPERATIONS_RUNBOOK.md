# Operations Runbook

This runbook covers common Orivory production-like operations.

## Quick Status

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/ready
```

`/health` checks API liveness. `/ready` checks Postgres, Redis, MinIO, and Qdrant.

## Admin Diagnostics

Use the admin-only diagnostics endpoint when `/ready` is degraded or ingestion appears stuck:

```bash
curl -fsS -H "Authorization: Bearer $ADMIN_ACCESS_TOKEN" \
  http://localhost:8000/api/v1/admin/diagnostics
```

The response includes:

- dependency checks for Postgres, Redis, MinIO, Qdrant, and Celery
- secret-safe config summary such as model names, rate limits, and MinIO bucket
- ingestion counts by status
- recent failed documents
- documents stuck in `pending` or `processing` longer than the configured threshold

`status: degraded` means at least one dependency check failed. The endpoint intentionally excludes secrets such as JWT keys, provider API keys, DB URLs, Redis URLs, and MinIO secret keys.

## Logs

API logs:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f app
```

Celery worker logs:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f celery_worker
```

Infrastructure logs:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f postgres qdrant
```

## Restart Services

Restart API only:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml restart app
```

Restart ingestion worker:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml restart celery_worker
```

Restart all app services without deleting data:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d app celery_worker celery_beat
```

## Investigate `/ready` Degraded

1. Call `/ready` and inspect the failing dependency name.
2. Check logs for that dependency.
3. Verify service health:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec postgres pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec redis redis-cli ping
curl -fsS http://localhost:6333/readyz
curl -fsS http://localhost:9000/minio/health/live
```

In the production overlay, Qdrant and MinIO ports are internal by default. Temporarily expose them only when direct host checks are needed.

## Source Sync Failures

Symptoms:

- `POST /api/v1/sources/{id}/sync` returns errors > 0
- `Source.status` flips to `error` instead of returning to `connected`
- Newly ingested memories do not show up in chat retrieval

Checklist:

1. Inspect `Source.sync_error` from the admin diagnostics endpoint or a
   direct DB read — the message includes the failing connector or stage
   (config validation, fetch, persist, etc.).
2. Confirm the source config still matches the registered connector
   requirements (e.g. OAuth token is not expired, RSS URL is reachable).
3. Re-run a sync through the API or admin endpoint after the fix:

```bash
curl -fsS -X POST -H "Authorization: Bearer $ACCESS_TOKEN" \
  http://localhost:8000/api/v1/sources/$SOURCE_ID/sync
```

4. The dispatcher (`SourceSyncService`) is idempotent on
   `(source_id, source_ref)`: re-running the same sync after a fix
   updates rather than duplicates items.

## Stuck Document Ingestion

Symptoms:

- uploaded document stays `pending` or `processing`
- chat does not retrieve newly uploaded content

Checklist:

1. Check Celery worker logs.
2. Confirm Redis is healthy.
3. Confirm MinIO object exists.
4. Confirm Qdrant health.
5. Confirm provider keys are configured.
6. Restart `celery_worker` if the worker is wedged.

Useful commands:

```bash
curl -fsS -H "Authorization: Bearer $ADMIN_ACCESS_TOKEN" \
  http://localhost:8000/api/v1/admin/diagnostics

docker compose -f docker-compose.yml -f docker-compose.prod.yml logs --tail=200 celery_worker
docker compose -f docker-compose.yml -f docker-compose.prod.yml restart celery_worker
```

## P1b cutover (SQLite / lite deployments)

P1b moves the vector store from the retired Chroma install to Qdrant and swaps
the embedding contract from masked mean to CLS pooling. It is an **offline**
cutover: stop the app, run the migration CLI, start the app.

### Migration before serving

An install that starts the app **before** the migration fails loud instead of
quietly returning zero hits: until `cutover` flips the generation pointers the
install keeps serving its old generation, and once the new code is live the
store's generation manifest and the vectors' embedding contract cannot be
reconciled — the read path raises `EmbeddingDimensionMismatch`
("populated generation has no manifest row — quarantine/rebuild") and refuses
to serve. That is the intended contract (Tasks 4/5 rulings): migrate first,
serve second.

### Sequence (run inside the app stack)

```bash
# 1. stop the app — every command except `inventory` refuses to run while the
#    app answers on APP_PORT (default 8000) or another migration holds
#    migrate.lock
docker compose stop app

# 2. inventory — read-only counts + quarantine lists (the only command that
#    may run while the app is up)
python scripts/migrate_qdrant.py inventory --out /backups/p1b-inventory.json

# 3. backup — VACUUM INTO snapshot + checksum manifest (+ uploads and the
#    retired-store copies, when they exist)
python scripts/migrate_qdrant.py backup --dir /backups/p1b

# 4. backfill — build the CLS generation for BOTH kinds (keyset scan; resumable)
python scripts/migrate_qdrant.py backfill --kind memory --batch 200
python scripts/migrate_qdrant.py backfill --kind chunk  --batch 200
#    interrupted? re-run the same command with --resume

# 5. verify — full read-side audit of the live collection, per kind
python scripts/migrate_qdrant.py verify --kind memory
python scripts/migrate_qdrant.py verify --kind chunk

# 6. cutover — flip BOTH generation pointers in one transaction, block the
#    intents that still target a retired generation, write the rollback marker
python scripts/migrate_qdrant.py cutover --yes

# 7. start the app and confirm readiness (the `qdrant` check is the key)
docker compose start app
curl -fsS http://localhost:8000/ready
```

Exit codes: `0` did what it says, `1` ran but a gate failed (verify findings, a
cutover the findings blocked, a backfill that stopped before every batch was
acked), `2` refused before doing anything (app still up, lock held, missing
`--yes`). `cutover` refuses while verify has open findings; re-run `verify` to
see them.

`APP_PORT` (config setting, default `8000`) is the port the CLI probes to decide
"the app is alive" before every mutating command. Set it when the API listens
elsewhere. Note the probe answers on the FIRST thing behind that port: if a
reverse proxy or the compose gateway answers there, the CLI sees "app alive" and
refuses — intended (the migration needs a quiesced store).

### Maintenance window (measured)

Measured on the calibration dry run: ≈1.8k short rows/s single-process, which
extrapolates at ~1000 chars per row to:

| eligible rows | projected backfill | fits a 60-min window? |
|---|---|---|
| 10,000 | ≈9 min | yes |
| 65,000 | ≈55 min | yes |
| 100,000 | ≈92–99 min | **no** — needs a longer window or a chunked plan |

Sizing rule: budget **≈65k eligible rows per 60-minute window** at ~1000 chars
per row. A larger store needs a longer window, or a chunked cutover (backfill +
verify one kind while the app still serves, then a short `cutover`). Backfill is
resumable, so a window that overruns can be extended and restarted with
`--resume` from the checkpoint.

### Artifacts left beside the database

All of these are written next to the SQLite file (`<db>` = the database path):

| artifact | written by | what it is |
|---|---|---|
| `<db>.pre-p1b.bak` (+ `.manifest.json`) | `backup` | VACUUM INTO snapshot of the pre-cutover DB + sha256 manifest; never overwritten |
| `migrate.lock` | every mutating command | single-migration lock (pid inside; stale locks are detected by pid liveness) |
| `<db>.backfill-checkpoint.json` | `backfill` | last acked keyset per kind — what `--resume` continues from |
| `<db>.p1b-expand-record.json` | `expand` (inside backfill/cutover) | the pointer the install served before the expand; the rollback report's `rollback_from` fallback |
| `<db>.p1b-rollback-marker.json` | `cutover` | `active` + `cutover_at`, written AFTER the transaction commits (it can be missing if the process died in that window; the expand record is then the fallback) |

### Postgres deployments: no boot drain in P1b

The boot drain is SQLite-only. A Postgres deployment has **no** startup drain of
`index_outbox`: pending index intents accumulate until P3 ships the
worker-side drain, so after a vector-store outage the intents stay queued
(durably) instead of being replayed at boot. Write-through is unaffected —
every write still embeds inline on the write path; only the retry path is
deferred. `/ready` shows the store red until it is back.

### Suppression / GC semantics of projection points

A document-projection point (a chunk point) whose SQL row is no longer eligible
is GC'd: the backfill's GC pass deletes points whose row is deleted, superseded,
dirty, suppressed or unowned, and the memory path refuses to serve a point whose
memory is gone. Concretely: after a source is forgotten, its
`memory_suppressions` entry blocks re-import/re-projection **and** the
projection's existing point is deleted by the next GC pass, so the content stops
being vector-searchable. That is the intended forgetting semantics — a
read-path behaviour change worth knowing about when a "forgotten" memory is
expected to be findable by vector search.

### After the rollback window closes

The pre-P1b store is retired: nothing serves from it once `cutover` has flipped
the pointers. Keep it (and the `.pre-p1b.bak` snapshot) until the one-release
rollback window closes, then drop the retired collection/store — the
`LEGACY_CHROMA_PATH` directory (its `Orivory_memories` and `rag_conv_*`
collections) — and its backups to reclaim the disk. See
[ROLLBACK_P1B.md](ROLLBACK_P1B.md) for the swap-back procedure and its removal
condition.

## Run Operational Smoke Evaluation

Offline smoke:

```bash
python eval/run_eval.py --mode offline --output-dir eval/results --top-k 5
```

Live API smoke after login/token setup:

```bash
python eval/run_eval.py --mode live-api \
  --api-base-url http://localhost:8000 \
  --access-token "$ACCESS_TOKEN" \
  --sample-docs sample_docs \
  --output-dir eval/results
```

## Flower

Flower is behind the `ops` profile in production compose.

Start it only when needed:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml --profile ops up -d flower
```

Stop it after use:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml stop flower
```
