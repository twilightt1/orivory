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

- dependency checks (Postgres/SQLite, Redis, MinIO/storage, Qdrant, MCP hub —
  plus a dormant `celery` key kept for payload compatibility: the slim branch
  has no broker)
- secret-safe config summary such as model names, rate limits, and MinIO bucket
- ingestion counts by status
- recent failed documents
- documents stuck in `pending` or `processing` longer than the configured threshold
- the index outbox (`index_outbox`): counts by status and by kind,
  `stuck_pending` (pending longer than 15 minutes) and `oldest_pending_at` —
  see [Background indexing (P3)](#background-indexing-p3). `blocked` counts are
  terminal intents that will never land; they are part of the summary on
  purpose.

`status: degraded` means at least one dependency check failed. The endpoint intentionally excludes secrets such as JWT keys, provider API keys, DB URLs, Redis URLs, and MinIO secret keys.

## Logs

API logs:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f app
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

Restart all app services without deleting data:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d app
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

Ingestion runs INLINE in the API process (there is no worker, queue or broker
to inspect): an upload returns once `process_document_sync` has parsed and
indexed the file, so a run that is wedged is wedged inside the API.

Checklist:

1. Check the API logs — the ingestion run logs its stages there.
2. Confirm Redis is healthy.
3. Confirm MinIO object exists.
4. Confirm Qdrant health.
5. Confirm provider keys are configured.
6. Restart the API if the run is wedged. Anything the ingestion enqueued into
   `index_outbox` survives the restart and is replayed by the background drain
   (see [Background indexing (P3)](#background-indexing-p3)).

Useful commands:

```bash
curl -fsS -H "Authorization: Bearer $ADMIN_ACCESS_TOKEN" \
  http://localhost:8000/api/v1/admin/diagnostics

docker compose -f docker-compose.yml -f docker-compose.prod.yml logs --tail=200 app
docker compose -f docker-compose.yml -f docker-compose.prod.yml restart app
```

## Background indexing (P3)

Every canonical write stamps a durable `index_outbox` intent in the SAME SQL
commit as the row (document chunks and memories). A write whose vector write
failed stays owed there, and the background drain loop
(`app/retrieval/memory/drain_loop.py`, started and stopped by the app lifespan)
replays it against the latest SQL state until the vector store confirms it.

- **One loop, both dialects.** P1b's SQLite-only gate and boot-only role are
  gone: a Postgres deployment replays its backlog through the same loop, and
  the boot's one bounded batch replays there too — a restart is a warm start.
  Write-through is unaffected — every write still embeds inline; the loop owns
  the RETRY path.
- **Settings.** `OUTBOX_DRAIN_ENABLED` (default `true`),
  `OUTBOX_DRAIN_INTERVAL_SECONDS` (default `5`) between idle rounds, and
  `OUTBOX_DRAIN_BATCH_SIZE` (default `50`). A round that applied anything runs
  the next one immediately, so a backlog drains at full speed. Set
  `OUTBOX_DRAIN_ENABLED=false` only to quiesce a store for a cutover: memory
  recall then stops waiting for its own writes.
- **Memory recall is guarded, and fails closed.** `MemoryRetriever.recall`
  waits for the calling tenant's pending intents, bounded by
  `RECALL_FRESHNESS_BUDGET_SECONDS` (default `2.0`), and answers
  `503 {"error": "index_freshness_timeout"}` rather than an empty result for a
  write that has not landed. The guard covers memory recall only (MCP reads are
  SQL and already fresh). An outbox it cannot READ ends in the same typed 503,
  so an unreadable queue looks like a write still in flight — check the summary
  below before treating a 503 as transient.
- **Several app processes (Qdrant server mode) are a known limitation.** Each
  process runs its own loop, so the same batch can be drained twice. The claims
  are idempotent (upserts by id, deletes with a readback, acks with a
  predicate), so a duplicate is wasted work, never wrong data. One draining
  process is the supported shape.
- **Reading it.** The diagnostics payload carries `index_outbox` (served by
  `/api/v1/admin/diagnostics` where the admin router is mounted):
  `by_status` — `pending` (still owed), `done`, and `blocked` (TERMINAL: an
  embedding-contract mismatch or a generation the cutover superseded; it will
  never land, so no recall will ever wait for it — a contract mismatch needs the
  contract fixed plus a reindex, a superseded generation was covered by the
  migration's backfill); `by_kind` — `memory` / `chunk`; `stuck_pending` —
  pending longer than 15 minutes; `oldest_pending_at` — the ISO timestamp of
  the oldest pending write's `created_at` (null when nothing is pending), not a
  duration. Every failed drain round also counts the
  `index.outbox_drain_failed` fallback (`app/observability/fallbacks.py`,
  log-grep `Fallback activated`): a rising rate means the retry path itself is
  failing — look at Qdrant, not at the loop.
- **No retention policy in P3.** The outbox grows monotonically: nothing prunes
  `done` rows, and the summary's `by_status` / `by_kind` counts scan the whole
  table (no status predicate), so the table — and the cost of reading it — grow
  with every write. Fine at current scale; the upgrade path is a prune/retention
  policy for `done` rows (plus a status-predicated index if that scan ever
  matters).

```bash
curl -fsS -H "Authorization: Bearer $ADMIN_ACCESS_TOKEN" \
  http://localhost:8000/api/v1/admin/diagnostics | python -m json.tool
```

## P1b cutover (SQLite / lite deployments)

P1b moves the vector store from the retired Chroma install to Qdrant and swaps
the embedding contract from masked mean to CLS pooling. It is an **offline**
cutover: stop the app, run the migration CLI, start the app.

### Migration before serving

The migration is not optional, and what skipping it costs depends on what the
generation pointer says:

- **Loud (a contract change).** Until `cutover` flips the generation pointers
  the install keeps serving its OLD generation. Where that pointer names a
  contract the new code no longer matches — the lite/arctic upgrade, whose
  P1a transitional row is the masked-mean generation — the read path raises
  `EmbeddingDimensionMismatch` ("same dim but different embedding contract:
  ... — fresh reindex required") and refuses to serve. That is the intended
  contract (Tasks 4/5 rulings): migrate first, serve second.
- **Empty (full-stack/Postgres, or an unchanged contract).** With NO active
  manifest row, `outbox.active_generation()` falls back to the transitional
  generation name with no fingerprint, an EMPTY generation is deliberately
  allowed, and the read path creates/reads that generation and answers `[]`.
  P1a never seeded `index_generations` on Postgres, so an un-migrated
  full-stack deployment serves empty vector results — and so does an install
  whose contract token did not change: the guard passes on token equality, but
  the generation it names is still the pre-P1b transitional one, whose Qdrant
  collection is empty (the old vectors live in the retired Chroma store, which
  the runtime cannot read). SQL stays canonical in both cases; the migration
  rebuilds the vectors, so no recall is lost permanently.

> On a full-stack/Postgres or unchanged-contract install an un-migrated
> deployment serves EMPTY vector results — nothing is pending, so nothing is
> waiting on anything. P3's freshness barrier closes the OTHER hole, the
> in-flight write, and it does so on the memory-recall path only (ruling R11;
> the MCP reads are SQL and already fresh): if this tenant has pending intents
> that do not land inside `RECALL_FRESHNESS_BUDGET_SECONDS`, recall answers
> `503 {"error": "index_freshness_timeout"}` instead of a false no-match —
> fail-closed (R14), never a `200` with `[]`. An outbox database the barrier
> cannot read produces the same typed 503, so it reads like a write that is
> still in flight: check `index_outbox` in the diagnostics payload (see
> [Background indexing (P3)](#background-indexing-p3)) before calling a 503
> transient.

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

Measured on the calibration dry run: ≈1.8k short rows/s ≈ 17.9k chars/s
single-process. Char throughput is the honest basis once rows get longer: at
~1000 chars per row that is ≈18 rows/s, which extrapolates to:

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

### Postgres deployments: the drain is not boot-only any more

P1b's boot drain was SQLite-only, so a Postgres deployment replayed nothing at
startup and pending `index_outbox` intents accumulated between restarts. P3
removed the SQLite-only gate and the boot-only role — not the boot batch: the
background drain loop runs on BOTH dialects, and the boot's one bounded batch
now replays on both too — see
[Background indexing (P3)](#background-indexing-p3). Write-through
is unaffected (every write still embeds inline on the write path); only the
retry path was ever deferred, and it is no longer deferred by dialect.

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
