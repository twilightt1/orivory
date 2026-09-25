# Operations Runbook

This runbook covers common Orivory operations. The product is ONE container
(the `app` service in `docker-compose.yml`) with an embedded vector store and a
SQLite database — there is no second service to inspect, and no admin router:
every check here is `/ready`, the logs, or a read of the SQLite file.

## Quick Status

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/ready
```

`/health` checks API liveness. `/ready` checks the one-container dependencies:
`sqlite`, `redis` (the in-process store), `storage`, `qdrant` (embedded) and
`mcp_hub` (only while `MCP_HUB_ENABLED=true`); a failing check carries an
`error` and the endpoint answers `503 status: degraded`.

## Diagnostics on the box

There is no admin endpoint — the full-stack admin router went with the retired
surfaces. Read the state directly:

```bash
# the failing check, by name
curl -fsS http://localhost:8000/ready | python -m json.tool

# what the app is doing (the drain loop and the boot report are logged here)
docker compose logs --tail=200 app

# the index outbox: what is still owed to the vector store
sqlite3 data/orivory.db \
  "SELECT status, kind, COUNT(*) FROM index_outbox GROUP BY status, kind"

# the oldest unlanded intent, and anything the retry gave up on
sqlite3 data/orivory.db \
  "SELECT created_at, kind, last_error FROM index_outbox \
   WHERE status != 'done' ORDER BY created_at LIMIT 20"

# ingestion state and the recent failures
sqlite3 data/orivory.db \
  "SELECT status, COUNT(*) FROM documents GROUP BY status"
sqlite3 data/orivory.db \
  "SELECT id, filename, error_msg FROM documents \
   WHERE error_msg IS NOT NULL ORDER BY created_at DESC LIMIT 10"
```

`blocked` outbox rows are TERMINAL: an embedding-contract mismatch, or a
generation a migration superseded. They will never land, so no recall waits for
them — a contract mismatch needs the contract fixed plus a reindex. `pending`
rows are still owed (see [Background indexing (P3)](#background-indexing-p3)).

## Logs

All logs — there is one service and no infrastructure container:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f app
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

1. Call `/ready` and read the failing check's `error` (the check names its own
   cause — see [LOCAL_RUN_GUIDE.md](LOCAL_RUN_GUIDE.md)).
2. `sqlite` / `storage`: confirm the data directory is writable by the container's
   `app` user.
3. `qdrant`: the embedded store is owned by ONE process — stop anything else
   holding `QDRANT_LOCAL_PATH` (a second app process, or an offline
   `scripts/migrate_qdrant.py` run) and restart.
4. `mcp_hub`: check the logs for the build error, or run with
   `MCP_HUB_ENABLED=false` to serve the REST surface alone.

```bash
docker compose ps
docker compose logs --tail=200 app
curl -fsS http://localhost:8000/ready | python -m json.tool
```

## An import landed but recall does not find it

Symptoms:

- `POST /api/v1/imports` reports a non-zero `index_failures` (the rows were
  committed; their vectors were not)
- a memory written a moment ago is missing from `POST /api/v1/memories/recall`

The import runs INLINE in the API process (there is no worker, queue or broker
to inspect) and its rows commit first, then index best-effort — a vector write
that failed stays owed in `index_outbox`, so this is normally a wait, not a
loss.

Checklist:

1. Check the API logs — the import logs its stages there.
2. Read the outbox: `pending` rows are still owed (see the diagnostics block
   above); `index.outbox_drain_failed` in the log means the retry path itself
   is failing — look at the `qdrant` check, not at the loop.
3. Confirm the data directory is writable (`sqlite` and `storage` checks) and
   that the embedding model session is not the blocker — a cold session
   downloads ~90 MB on a fresh volume.
4. Restart the API if a run is wedged. Anything it enqueued into `index_outbox`
   survives the restart and is replayed by the background drain (see
   [Background indexing (P3)](#background-indexing-p3)).

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs --tail=200 app
sqlite3 data/orivory.db "SELECT status, COUNT(*) FROM index_outbox GROUP BY status"
```

## P4a — namespace ACL (personal-only) and the v5 ladder

P4a makes the namespace a REAL column and a single authorization predicate, with
sharing deliberately OFF: the only namespace in existence is `personal`, every
pre-P4 row was backfilled into it, so a single-namespace deployment answers
exactly as it did before the column existed.

- **Schema v5.** `memories.namespace` (`VARCHAR(32) NOT NULL DEFAULT 'personal'`)
  plus the `ix_memories_namespace_user(namespace, user_id)` index, installed by
  the ladder's `v4 -> v5` step — ONCE, on the version transition, and on a fresh
  install too (which has nothing to back up). The ADD COLUMN *is* the backfill:
  the constant default stamps every pre-existing row, and nothing per-row is ever
  computed. A later boot never re-asserts or repairs the column (a restart must
  not re-assert a namespace an operator moved). The step's own milestone
  snapshot is `<db>.pre-p4.bak` — never overwritten, reused when an interrupted
  upgrade resumes.
- **One predicate.** `app/retrieval/memory/visibility.py::namespace_predicate`
  is the only spelling of the boundary, and its value comes from
  `namespaces.PERSONAL` / `personal_namespace(user_id)` — never from client
  input (ruling R33: the reindex request body and the migration CLI take no
  namespace from a caller). Every reader composes it into the SAME statement as
  the row it protects (before any LIMIT/aggregate). The PK surfaces the mounted
  REST router owns (`db.get` + the one `_owned` check) compare the loaded row
  against the same value; three PK reads do NOT, and none of them takes an id
  from a request: `app/graph/builder.py:60,145` (the graph write-back re-reads a
  row the app itself just wrote — carried to P4b), `app/retrieval/memory/outbox.py:597`
  (deliberate — the applier must read the row to write THAT row's namespace onto
  its point, rulings R33/R34) and `scripts/migrate_qdrant.py:388-391`
  (`correction_chains`, the migration CLI's whole-database inventory read).
  `namespace_predicate` itself refuses `None`/empty, so the predicate side cannot
  compile the CLI's "whole DB" value into a reader; the fence above does not scan
  `db.get` at all.
- **Qdrant payload, and the `is_empty` branch (R32).** The payload carries
  `namespace`, and the memory filter is
  `should[match(namespace), is_empty(namespace)]` for the `personal` query. The
  `is_empty` branch exists because a point written BEFORE the key existed is
  personal: a bare `must` match would make a pre-P4 install recall nothing until
  a full reindex. It is a spelling of the ONE namespace (`personal`) — a query
  for any other namespace must not inherit pre-P4 points — and it must be
  dropped when a second namespace is served (sharing on): from then on a
  key-less point cannot be assumed personal.
- **What is NOT here.** Sharing/team namespaces (NOT P4b either — P4b ships
  the lifecycle, `personal` is still the only namespace in existence; a later
  phase owns sharing); a namespace component in the cache keys (§4.3 — no cache
  read/write path is live: the retrieval query cache has no producer or
  consumer and the response cache has no caller, both are invalidation-only
  today, so the component is owed by whoever wires a live one — P4a AND P4b
  both left them unwired); and a namespace column on the outbox record (the
  applier re-reads the row and writes THAT row's namespace — R33).
- **The fence, and its ceiling.** The repo-wide `select(...)` fence watched
  every statement that NAMED a memory model under `app/`, allowlisted by file
  and statement count, alias-aware — it was removed together with the dormant
  full-stack routers whose surfaces it covered. Green there was never "every
  read is guarded":
  `sqlalchemy.select` behind a module alias, `text()` queries, a model passed
  through a variable, `update`/`delete` writes and Python-side row checks
  (`db.get` + `_owned`) are outside such a scan. Those surfaces stay pinned
  behaviourally (`tests/retrieval/test_visibility.py`,
  `tests/retrieval/test_p4a_gate.py`, `tests/retrieval/test_namespace_acl.py`).

### Rollback: a pre-P4 binary, and coming back

A pre-P4 binary's ladder tops out at v4 and refuses a v5 file with
`unsupported SQLite schema version 5; expected 4`. The documented way back is
[ROLLBACK_P1B.md](ROLLBACK_P1B.md) §7 — re-stamp `user_version = 4` on a COPY
(drop the `namespace` column and its index too if the target binary must not see
them), and roll forward by booting a P4a binary: the step re-runs, inspects the
column instead of re-adding it, and reuses the operator's `.pre-p4.bak`.

**A milestone snapshot is not a clean "before" state, and its name is not its
version stamp.** The snapshots are taken mid-ladder, and SQLite commits DDL
immediately while `user_version` is stamped only at the very END of the whole
ladder run. Measured, and pinned by
`tests/retrieval/test_p4a_gate.py::test_a_milestone_snapshot_is_not_promised_clean`:
a v3 install upgraded in one boot gets a `<db>.pre-p4.bak` that reads
`user_version = 3` while already containing the v4 FTS objects, and an adopted
(unversioned v1-shape) install gets snapshots that all read `user_version = 0`,
with `.pre-p2.bak`/`.pre-p4.bak` already holding the v2 tables and columns. Check
the stamp AND the objects inside a snapshot
(`sqlite3 <snapshot> "PRAGMA user_version"`) before pointing an older binary at
it.

### Erasure receipts: the out-of-namespace residuals

`db_residual` gained two namespace counters, and either one keeps a receipt from
ever reaching `completed` (`completed_with_residual` at best):

- `cascaded_out_of_namespace` (R36/F1): rows the `parent_id` FK cascade removes
  even though the walk did not collect them — a same-account row in another
  namespace (or another user's) whose vector the erasure never purges;
- `derived_out_of_namespace` (I3): rows of the erasing user's other namespaces
  that derive from an erased id; the walk inside the namespace still reports
  `derived_closure='complete'` for what it DID complete, and this key records
  what it deliberately left outside.

Three known ceilings, all diagnostic — none of them changes a verdict:

- the pre-delete cascade count under-counts below the first level (a cascade
  that removes three rows two levels down is counted once);
- the hard-erase walk collects ONE `cm_derived_from` hop (`collect_derived_ids`),
  while the pre-delete cascade count spans the whole `parent_id` cascade below
  the walked set (the S5 fix, T1) — so a cascade that removes a row BELOW a
  derived node IS reported, but a derived view of a derived view is not walked
  at all. Measured: such a row survives the erase while the receipt reads
  `completed` (no residual key covers it). Rule v1 — the P4b consolidation
  producer — is non-recursive and never builds that chain itself; the soft path
  (MCP `forget_memory`) DOES walk the closure transitively, so this residual is
  hard-erase-only;
- `residual_rows` is an upper bound, not a partition: a cross-user child is
  counted in BOTH `cross_user_children` and `cascaded_out_of_namespace`, so the
  number can exceed the rows that exist.

### Migration CLI / export: `--namespace`

`backfill` (and `rollback_to_chroma.py`) read ONE namespace: their `--namespace`
flag defaults to `personal` — the only namespace P4a can hold — so an operator's
export never sweeps in rows the ACL would refuse to serve. `verify` takes no
`--namespace` flag at all: it audits the same default (`sql_rows`'s
`namespace=PERSONAL`, `scripts/migrate_qdrant.py:410`), so a verify run cannot be
scoped by a flag someone forgot to pass. The internal `namespace=None` escape
hatch is a whole-database AUDIT read, not a serving path; when a second namespace
gets its own points, that audit will see them as "orphans" (points with no row in
the exported namespace) and block `verify`/`cutover` — P4b must revisit this
before enabling sharing.

### Timeline neighbours hide dirty rows (behavior change)

MCP `timeline` neighbours are filtered by the caller's namespace AND
`not_dirty_predicate()`: a stale derived row is wrong data, not history, so it
no longer appears beside the anchor. Superseded neighbours still do (labelled).

## P4b — lifecycle: soft forget, consolidation, retention

P4b is the lifecycle phase on top of P4a's namespace: invalidation and the
dependency closure, the same-slot correction CAS, soft forget with universal
suppression, a budgeted consolidation producer, and opt-in retention. Sharing
did NOT land here (see the P4a "What is NOT here" note).

### The lifecycle states

One rule, `correction.state_of` (mirrored in SQL by
`visibility.state_expression`), precedence
`invalidated > superseded > dirty > needs-check > current`:

| state | meaning | served? |
|---|---|---|
| `current` | the row's fact stands | yes |
| `needs-check` | ambiguous/late-import/refused — a human or agent must decide | yes, labelled |
| `superseded` | a newer version points back at it | history: REST list and direct reads yes, recall no (unless a surface asks for history) |
| `dirty` | a stale derived view — wrong data, not history | never, on any surface (not even `timeline`) |
| `invalidated` | forgotten, or expired by retention — provenance kept | history: direct reads and `timeline` yes, serving surfaces no |

`pin` protects a row from AUTO-retention only. Explicit forget and hard erase
both win over a pin.

### Soft forget (MCP `forget_memory`)

- Invalidates the target AND its transitive closure (`parent_id` +
  `cm_derived_from`, `collect_dependency_closure`) — unlike the hard-erase
  walk, which collects one derived hop (ceiling above). A truncated closure is
  REFUSED before any write.
- Writes a `memory_suppressions` row for EVERY affected `source_ref` (not only
  the root), plus the projection's upload-time `content_hash` when it carries
  one — the re-upload guard (R38).
- The row, its content and its provenance stay; the vector point stays too, and
  its payload `visibility_state` is refreshed to `invalidated` by the drain
  (R37). SQL `not_dirty_predicate()` is what closes serving immediately.
- The receipt (`detail.mode: "soft"`) reports `completed` only after a
  serving-off readback; leftovers are `completed_with_residual`, an unreadable
  check is `completed_unverified`.
- The tool's response keys are `receipt_id`, `status`, `invalidated`,
  `suppressed`, `skipped`, `invalid` (no `erased`): `invalidated` counts the
  REQUESTED targets, `suppressed` every affected source.

### Consolidation producer (drain hook, R39)

`run_consolidation` runs after every drain round that landed work AND on idle
ticks, at most `CONSOLIDATION_BUDGET_PER_RUN` (default 10) groups per user per
pass for at most `CONSOLIDATION_USERS_PER_PASS` (10) users per pass.

- Rule v1 (`tag-summary.v1`): servable memories grouped by tag, ≥2 sources per
  group, summarized through the shared LLM seam. NON-RECURSIVE: a summary is
  never a source for another summary.
- Provenance on the row: `cm_derived_from`, `cm_source_revisions`,
  `cm_rule_version`, `cm_derived_key` = `sha256(sorted source ids |
  revisions | rule)` — the dedupe key. A re-run whose key is already published
  in a servable row skips the group BEFORE any LLM call.
- Publish-time guard: the sources are re-read right before the publish
  (`populate_existing`); anything that moved, vanished, left the namespace or
  stopped being servable stands the publish down and marks the closure's stale
  views `dirty`. The window it protects is one generation pass; a source that
  changes and NO pass runs leaves the old view serving (still labelled
  `derived`) until a later pass re-walks it.
- `derived` is the serving label, and it lives in the MCP provenance only
  (`get_memory`/`add_memory`/`correct_memory`: `derived`, `derived_from`,
  `source_revisions`, `rule_version`). REST/compact payloads carry the raw
  metadata markers (`metadata.cm_assertion == "derived"` plus the lineage keys)
  and never the computed flag.
- Idempotence is PER PROCESS: the in-memory key set is the run's own view, so
  two app processes passing over the same store can each publish the same key
  (duplicate summaries, redundant LLM spend — the same single-flight ceiling
  the drain loop documents). One app process is the supported shape.

### Retention (opt-in, spec §8.1)

- Default OFF, and nothing is swept for a user who has not opted in. Opt in (or
  change the window) with `PATCH /api/v1/users/me/settings` — a FULL REPLACE
  body; `{}` turns it off. `enabled` without a window is a 422. Read back via
  `GET /api/v1/users/me`.
- The sweep rides IDLE drain ticks only (`_retain_after_drain`), and with no
  user enabled the whole pass is one `SELECT` on `users` — no memory scan.
- Per expired row: `invalidated` + one append-only `memory_access_logs` row
  (`action: retention_expired`, detail `{reason, retention_days, indexed_at}`)
  + the same payload-refresh intent soft forget uses. The clock is
  `indexed_at` — how long this install has HELD the row. `pinned` rows are
  exempt; a second sweep is a no-op (already-invalidated rows are never
  selected).
- Expiry RELABELS what it touches: precedence is `invalidated` > `superseded`,
  so a superseded row that expires reports `invalidated` from then on — the
  timeline label changes with it, and the history surfaces that widen to
  superseded (`include_superseded` recall, MCP `include_history`) stop
  returning it.
- **Retention is a per-row expiry, NOT a closure walk, and NOT a privacy
  guarantee.** It writes NO suppression row (auto expiry is not the user
  forgetting a source), so a re-import after a row expired is caught only by
  the ordinary `source_ref` dedup (`skipped_duplicates` — silently, like any
  duplicate), and a derived view whose sources expired keeps serving until its
  OWN `indexed_at` passes the window. Treat "retention expired this" as "this
  row left serving", never as "the fact is gone from every surface".
- **Several app processes:** the sweep is per-process like the drain loop, and
  it selects and writes per row — two processes racing the same row can both
  write an audit row for it (the second invalidation is a no-op on the row
  itself, but the audit log can show the expiry twice). Redundant audit rows,
  never a second state change; one app process is the supported shape.

### The correction CAS (MCP `correct_memory`)

An explicit `memory_id` carries the revision the tool just read, so the
supersede applies only if the slot is still where the read found it; a
concurrent writer that moved it answers `status: "conflict"` (new row flagged
`needs-check`, nothing superseded, read the slot again). A slot holding more
than one exact candidate is refused the same way — the tool never supersedes a
candidate the caller did not name. Ledger rows carry the status, so
`mcp_correct` volumes with a rising `conflict` share mean agents are racing on
the same slot (or reading a slot that is genuinely ambiguous).

## Background indexing (P3)

Every canonical write stamps a durable `index_outbox` intent in the SAME SQL
commit as the row (document chunks and memories). A write whose vector write
failed stays owed there, and the background drain loop
(`app/retrieval/memory/drain_loop.py`, started and stopped by the app lifespan)
replays it against the latest SQL state until the vector store confirms it.

- **One loop.** The boot replays one bounded batch (a restart is a warm start)
  and the loop owns everything after it. Write-through is unaffected — every
  write still embeds inline; the loop owns the RETRY path.
- **Settings.** `OUTBOX_DRAIN_ENABLED` (default `true`),
  `OUTBOX_DRAIN_INTERVAL_SECONDS` (default `5`) between idle rounds, and
  `OUTBOX_DRAIN_BATCH_SIZE` (default `50`). A round that applied anything runs
  the next one immediately, so a backlog drains at full speed. Set
  `OUTBOX_DRAIN_ENABLED=false` only to quiesce a store for a cutover: memory
  recall then stops waiting for its own writes. The hooks that ride the drain
  stop with it — with the drain disabled, neither the consolidation producer
  (R39) nor the retention sweep (`_consolidate_after_drain` /
  `_retain_after_drain`) ever runs.
- **Memory recall is guarded, and fails closed.** `MemoryRetriever.recall`
  waits for the calling tenant's pending intents, bounded by
  `RECALL_FRESHNESS_BUDGET_SECONDS` (default `2.0`), and answers
  `503 {"error": "index_freshness_timeout"}` rather than an empty result for a
  write that has not landed. The guard covers memory recall only (MCP reads are
  SQL and already fresh). An outbox it cannot READ ends in the same typed 503,
  so an unreadable queue looks like a write still in flight — check the summary
  below before treating a 503 as transient.
- **Several app processes are a known limitation.** One process is the supported
  shape (`QDRANT_MODE=local` refuses to boot with more than one; see
  `app/main.py`): it drains by design (the loop, the boot's one
  bounded batch and the 5s cadence above), and a second process drains the same
  rows again. That is redundant WORK, never corruption: every claim is
  idempotent by entity id — an upsert writes the row's point id and re-reads the
  row after the write (rewriting the point from the refreshed row when the
  revision moved on), a delete is read back before it is acked, and an ack is
  predicated on the row's revision and generation. A duplicate therefore costs
  embedding time and a redundant store call — the two-writer shape converges on
  one point per entity (the point id is the entity's, not the write's) and the
  post-write re-check rewrites it from the refreshed row when the revision has
  moved on. That fence is not global: a THIRD write landing between an
  applier's post-write re-read and its rewrite stays unfenced — the accepted
  residual documented on `_settle_written_snapshot`
  (`app/retrieval/memory/outbox.py`); closing it needs a revision-fenced write
  Qdrant does not offer. Treat extra drainers as wasted budget, not as a risk
  inside that documented bound. The sibling residual has the same upgrade
  path: R22's repair is best-effort — when the repair write itself raises, the
  retry re-reads, sees the revision has already moved on and acks the intent
  `skipped`, so the older revision's payload stays ownerless, exactly the
  pre-R22 outcome (no regression, no repair either, until a revision-fenced
  write exists).
- **Reading it.** The outbox is read with SQL (the diagnostics block above), and
  the drain reports its own counts in the app log:
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
  failing — look at Qdrant, not at the loop. An ack commit that keeps failing
  is the same class at a higher cost: the row stays `pending`, so the next
  round re-runs the write WHOLE — embedding included — every 5s until the
  commit lands (no backoff by design, ruling R31; the interval bounds the
  churn, the counter makes it visible, and nothing is lost — the retry
  converges on the same point id).
- **No retention policy in P3.** The outbox grows monotonically: nothing prunes
  `done` rows, and the summary's `by_status` / `by_kind` counts scan the whole
  table (no status predicate), so the table — and the cost of reading it — grow
  with every write. Fine at current scale; the upgrade path is a prune/retention
  policy for `done` rows (plus a status-predicated index if that scan ever
  matters).

```bash
sqlite3 data/orivory.db "SELECT status, kind, COUNT(*) FROM index_outbox GROUP BY status, kind"
```

## P2 — rerank, hybrid recall and the FTS5 lexical index

### What ships, and what the dials mean

- **Rerank is opt-in** (`RETRIEVAL_SEMANTIC_RERANK=false` by default). Where it
  is on, the cross-encoder reorders the fetched pool and MERGES into dense
  order: the served count never shrinks because rerank ran — it is
  `min(top_k, eligible)`, always.
  - `RERANK_TOP_N` (default **20**) is the per-call CAP on the reranker's own
    answer (`min(top_k, cap)`), **not** the rerank window. The window is the
    retrieval pool: `top_k x RETRIEVAL_RERANK_POOL_MULTIPLIER` (default 2.0, so
    20 for the default `top_k=10`). A cap below the window leaves the tail of
    every recall on the dense x boost x decay regime inside the same sort
    (ruling R13) — keep them in step.
  - The lane is the bundled ONNX cross-encoder
    (`gte-multilingual-reranker-base` int8, Apache-2.0, ~341 MB downloaded once
    into `LOCAL_E5_DIR`, the same directory as the embedding models): no API
    key, no per-call cost, no outbound memory text. Nothing warms it at boot —
    the first call pays the ONNX session build (~2 s, ~1.1 GB resident), so
    enable it before the traffic, not under it. It scores the pool in CPU time
    (~0.3 s per 1024-token pair): expect a recall in the seconds, not
    milliseconds, while it is on.
  - A scoring failure (missing or corrupt model files, a broken session, an
    OOM) is typed: the answer continues in dense order and the
    `retrieval.rerank_failed` counter increments. A rising rate means the model
    cache or the box is broken — alert on the rate, not on the occurrence.
  - The diagnostics payload's `config.reranker_top_n` is that CAP: it is not a
    result count and not the rerank window.
- **Hybrid recall ships OFF** (`RETRIEVAL_HYBRID_ENABLED=false`). With it on,
  recall runs the SQLite FTS5 lexical leg beside the dense leg and fuses them by
  RRF (`RETRIEVAL_RRF_K`, default 60). It may only be enabled by the T7
  ablation artifact passing the signed gate (`eval/ablation_retrieval_p2.json`
  records the verdict); that flip is a separate, documented decision — never a
  config default. The OFF path is dense-only: no lexical query, no
  `lexical`/`fused` counters.
- **The lexical leg is SQLite-only** (`memory_fts`, created by the schema
  ladder's v4 step). Where there is no lexical leg the recall keeps the typed
  `503` on a vector outage instead.
- **A vector outage answers from the lexical leg.** The recall returns
  FTS5 results, `trace.counts.lexical` is set, `dense` is absent, and
  `retrieval.vector_unavailable` increments (ruling R19). This fallback is NOT
  gated by the hybrid flag — it only ever replaces the typed `503`.
- **MCP search keeps the freshness barrier (ruling R24).** `search_memory`
  ranks through the same recall seam as the API, so the P3 barrier applies: on
  a deployment that cannot land its intents, an MCP search pays the
  `RECALL_FRESHNESS_BUDGET_SECONDS` wait and then answers from the SQL
  ordering. That is the design (the `add_memory` → `search_memory` chain is
  what P3 protects), and the escape hatch is `OUTBOX_DRAIN_ENABLED=false`,
  which quiesces the store AND stops recall waiting for its own writes.
- **Budgets (signed §12.2).** Read-your-writes stays **2.0 s** for a recall
  ALONE (the P3 barrier); under concurrent bulk ingest a `503
  index_freshness_timeout` is the accepted shape (ruling R9) — clients retry.
  Recall **p95 <= 150 ms** at fixture scale, asserted by the P3 gate
  (`tests/retrieval/test_p3_gate.py`, `LATENCY_P95_MS`) — the P2 gate asserts
  the §9 retrieval row, not a latency budget. RSS **<= 1 GB** is asserted as
  the P2 gate process' peak RSS, stdlib only
  (`tests/retrieval/test_p2_gate.py`, `resource.getrusage`), where the real
  embedding session + the embedded Qdrant dominate the number. The embed
  executor (`EMBED_EXECUTOR_WORKERS`, default 2) is the loop-lag lever; ORT's
  `EMBED_ORT_INTRA_OP_THREADS` is the oversubscription dial (0 = ORT default,
  set 1 only to cap a small machine — and re-measure, it costs ~4x).
  **The p95 and the 2.0 s RYW tests are asserted only where the arctic ONNX
  cache is warm and are SKIPPED cold** (the suite refuses to download ~90 MB in
  CI), so the committed evidence for those numbers is a LOCAL run — the
  artifact and the task reports, not a CI green.
- **The shared LLM gate parks its waiters in the loop's default executor.**
  `llm_client._SharedSemaphore` (the `LLM_MAX_CONCURRENCY` budget) waits with
  `run_in_executor(None, ...)` — the same default pool `asyncio.to_thread`
  uses for uploads, reindex and the graph builds. The pool holds
  `min(32, cpu_count + 4)` threads: on a 16-core box that is ~17+ contended
  waits (pool size minus the permits in flight) before unrelated `to_thread`
  work queues behind them. A dedicated wait-executor is the upgrade if that
  ceiling is ever reached; today the budget (`LLM_MAX_CONCURRENCY`, default 3)
  stays far below it. Known latent issue, deliberately NOT fixed here:
  `llm_client.complete()` re-acquires the same gate for its
  structured-outputs retry while still holding it (line 286 -> 304), so a
  saturated budget can deadlock the callers that all need the second permit.

### The lexical index, erasure and disk

`memory_fts` is an FTS5 table plus its shadow tables, maintained by triggers in
the SAME transaction as the `memories` write (update = delete-by-`memory_id`
then insert; identity never rides on the FTS rowid). Deleting or erasing a
memory removes its row and its index entry, and the serving paths confirm
absence — but **deleted text can persist in the file's bytes**: inside the FTS
shadow tables' free pages, in the SQLite freelist, and in the WAL, until a
checkpoint or `VACUUM` reclaims them. Erasure receipts verify the SERVING path
(no row, no point), never the physical bytes. If an erasure must be reflected
on disk immediately, checkpoint and compact:

```bash
sqlite3 /data/orivory.db "PRAGMA wal_checkpoint(TRUNCATE); VACUUM;"
```

### Ladder v4 and going back to a pre-P2 binary

A P2 install runs SQLite `user_version = 4` (the v3 schema + `memory_fts` + its
three triggers); the ladder steps v3 -> v4 once, backfills the index from
coverage (by id, not by count) and takes its own `<db>.pre-p2.bak` milestone
snapshot beside the database. A later boot at v4 never re-asserts or rebuilds
the index — repair is explicit (`lexical_index.rebuild`).

A pre-P2 binary cannot open a v4 file: its ladder tops out at 3 and refuses with
`unsupported SQLite schema version 4; expected 3`. The documented way back is
[ROLLBACK_P1B.md](ROLLBACK_P1B.md) §6 (drop the triggers + the virtual table,
re-stamp `user_version = 3`) — run it on a COPY, and roll forward by booting a
P2 binary, which re-runs the step.

### Fallback counter labels

`app/observability/fallbacks.py` (inspect with `fallback_counts()`, or grep the
`Fallback activated` debug logs; aggregate centrally in a multi-process
deployment). Alert on the RATE, not on any single activation. These are the
labels the current code emits — this table IS the alert surface:

| label | means |
|---|---|
| `retrieval.vector_unavailable` | vector store down, the lexical leg answered (SQLite) or the typed 503 was served |
| `retrieval.rerank_failed` | local reranker model failure, dense order kept |
| `mcp.search_sql_fallback` | MCP search answered from the SQL ordering — barrier timeout, vector outage, or a degraded leg served empty (R23/R25) |
| `index.outbox_drain_failed` | a drain round raised; intents stay pending for the retry |

Two labels from older releases — `retrieval.bm25_rebuild_failed` and
`crag.grading_failed` — have NO emitting call site in the current code (they
survive only in the module's docstring). Their counters stay at zero: do not
build alerts on them.

```bash
# the counters are live only in the process that emitted them
docker compose logs app | grep "Fallback activated"
```


## Offline vector-store maintenance

`scripts/migrate_qdrant.py` is the offline CLI for the embedded store: it reads
and audits the store beside a QUIESCED app, and it is not part of normal
operation. Every command except `inventory` refuses to run while the app answers
on `APP_PORT` (default `8000`) or another migration holds `migrate.lock`.

```bash
# inventory — read-only counts + quarantine lists (the only command that may
# run while the app is up)
python scripts/migrate_qdrant.py inventory --out /backups/inventory.json

# backup — VACUUM INTO snapshot + checksum manifest beside the database
python scripts/migrate_qdrant.py backup --dir /backups

# verify — full read-side audit of the live collection, per kind
python scripts/migrate_qdrant.py verify --kind memory
python scripts/migrate_qdrant.py verify --kind chunk

# verify --restore-drill — open a backup and prove it is a usable store
python scripts/migrate_qdrant.py verify --restore-drill --dir /backups
```

`backfill` and `cutover` (build a new embedding generation, then flip the
pointer to it in one transaction) are still there for a contract migration; both
refuse while `verify` has open findings. Exit codes: `0` did what it says, `1`
ran but a gate failed, `2` refused before doing anything (app still up, lock
held, missing `--yes`).

Artifacts written beside the SQLite file (`<db>` = the database path):

| artifact | written by | what it is |
|---|---|---|
| `<db>.pre-p1b.bak` (+ `.manifest.json`) | `backup` | VACUUM INTO snapshot + sha256 manifest; never overwritten |
| `migrate.lock` | every mutating command | single-migration lock (pid inside; stale locks are detected by pid liveness) |
| `<db>.backfill-checkpoint.json` | `backfill` | last acked keyset per kind — what `--resume` continues from |
| `<db>.p1b-expand-record.json` | `expand` (inside backfill/cutover) | the pointer the install served before the expand; the rollback report's `rollback_from` fallback |
| `<db>.p1b-rollback-marker.json` | `cutover` | `active` + `cutover_at`, written AFTER the transaction commits (it can be missing if the process died in that window; the expand record is then the fallback) |

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

> **P4b clarification.** That GC pass is the migration CLI's OFFLINE pass — it
> runs when an operator runs it. The LIVE path does not wait for it: a soft
> forget (`forget_memory`, P4b/T3) closes serving immediately through the SQL
> visibility rule (`not_dirty_predicate`) and refreshes the memory point's
> payload state to `invalidated` (R37) — the point is kept, not purged, and
> physical removal is the reconciliation repair's job (R34).

### After the rollback window closes

The pre-P1b store is retired: nothing serves from it once `cutover` has flipped
the pointers. Keep it (and the `.pre-p1b.bak` snapshot) until the one-release
rollback window closes, then drop the retired collection/store — the
`LEGACY_CHROMA_PATH` directory (its `Orivory_memories` and `rag_conv_*`
collections) — and its backups to reclaim the disk. See
[ROLLBACK_P1B.md](ROLLBACK_P1B.md) for the swap-back procedure and its removal
condition.

## Run Operational Smoke Evaluation

```bash
python eval/run_eval.py --mode offline --output-dir eval/results --top-k 5
```

The offline lane is the whole harness — the live-API lane was deleted with the
chat surface. See [../eval/README.md](../eval/README.md).
