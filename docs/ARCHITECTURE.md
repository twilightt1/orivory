# Orivory Architecture

> Single source of truth for how Orivory works. Supersedes the pre-pivot
> `architecture.md` / `TECHNICAL_ARCHITECTURE_v2.md` /
> `SOTA_TECHNICAL_SPECIFICATION.md` / `AI_ML_OVERVIEW.md` (removed — history
> lives in git). Last verified against the code: 2026-09-22.

Orivory is a **memory hub for AI agents**. One mental model:

```
                    ┌─────────────────────────────────────┐
   AI agents        │            Orivory (self-hosted)    │
   (Claude, Cursor, │                                     │
   OpenClaw, …) ────┼─▶ /mcp    ──▶ mcp_hub (scoped tools)│
                    │                                     │
                    │  ─▶ /api/v1 ──▶ REST (memories,     │
                    │                imports, erasure, …) │
                    │                                     │
                    │   memory store (SQLite + embedded   │
                    │   Qdrant) · knowledge graph         │
                    │   access ledger · erasure receipts  │
                    └─────────────────────────────────────┘
```

Everything — REST and MCP — funnels into the same memory store, so every
connected agent reads and writes one brain. The whole thing is ONE process in
one container: no server to talk to beyond the API itself, no second service to
keep alive.

## 1. The memory spine

The core claim: **one brain, many ways to ask.** Two worlds used to live
side by side (per-conversation documents vs per-user memories); the hub
unifies them.

- **`memories`** (SQLite) — the source of truth. Every memory carries
  `user_id`, `source_type` (manual_note, chatgpt_import, claude_import,
  generic_import, mcp_agent, conversation_excerpt, …), `source_ref`
  (dedup key), `content`, `tags`, and the salience fields
  (`salience`, `recall_count`, `last_used_at`).
- **Namespace (P4a) — an authorization boundary, not a tag.** Every memory row
  carries `namespace` (`VARCHAR(32) NOT NULL DEFAULT 'personal'`, index on
  `(namespace, user_id)`; SQLite ladder v5). Two
  rows with the same text in two namespaces are two facts with different
  owners' permissions: every reader/writer/admin/export composes
  `visibility.namespace_predicate(...)` — the ONE spelling, built from
  `namespaces.PERSONAL` / `personal_namespace(user_id)` — into the SAME SQL
  statement as the row it protects (before any LIMIT or aggregate). The REST
  primary-key surfaces (`db.get` + the shared `_owned` check) compare the loaded
  row against the same value; three PK reads do not, and none takes an id from a
  request — the graph write-back (`app/graph/builder.py`), the outbox applier
  (deliberate: it writes THAT row's namespace onto its point, R33/R34) and the
  migration CLI's whole-database inventory read (`correction_chains`). The AST
  fence scans `select(...)` statements and does not see `db.get` at all.
  Namespace is never derived from client input. **P4a ships personal-only:
  sharing is OFF**, `personal` is the only value that exists, and every pre-P4
  row was backfilled into it — a single-namespace deployment answers exactly as
  it did before the column existed. Qdrant payloads carry `namespace` and the
  memory filter also accepts a key-less point as personal (R32), so pre-P4
  vectors keep answering. Operationally:
  [OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md) §P4a.
- **Qdrant** — the vector index, one generation per kind and embedding
  contract (`orivory_memories__<contract>`, `orivory_chunks__<contract>`),
  written best-effort after every DB write (SQLite is truth; the drain loop
  rebuilds vectors from rows). Which generation is active per kind is a row in
  `index_generations`; a store that cannot name its contract is quarantined
  rather than served.
- **Lexical leg + reranking (P2)** — memory recall is DENSE-only by default.
  A SQLite FTS5 lexical leg (`memory_fts`, schema ladder v4) can be fused with
  the dense page by RRF when `RETRIEVAL_HYBRID_ENABLED=true` — it ships OFF,
  and only the T7 ablation artifact (`eval/ablation_retrieval_p2.json`) may
  enable it. Where a vector outage hits, the SQLite lexical leg answers alone
  (a typed 503 where no lexical leg exists). Cross-encoder rerank
  (`RETRIEVAL_SEMANTIC_RERANK`) is opt-in per deployment; a reranked head is
  MERGED into dense order, so the served count never shrinks because rerank ran.
- **Salience loop** — memories used in answers get bumped; untouched ones
  decay. Ranking is salience × recency × relevance (the Generative-Agents
  scoring, reinforced on access); the decay is computed when a memory is
  scored, not by a periodic job (there is no beat/scheduler).
- **Knowledge graph** — `entities` / `relations` extracted per memory and
  written back best-effort after the row commits (`app/graph/builder.py`,
  scheduled off the write path); nothing serves the graph — no router is mounted
  for it, and recall builds its context from the recalled memories.
- **Lifecycle states (P4b) — one state machine, one closure.** `state_of`
  (`app/retrieval/memory/correction.py`) labels every row
  `invalidated > superseded > dirty > needs-check > current`, mirrored in SQL by
  `visibility.state_expression` / `not_dirty_predicate` /
  `current_memory_predicate`, so a reader filters BEFORE its LIMIT instead of
  dropping rows after it. `dirty` (a stale derived view) is never served on any
  surface — not even `timeline`; `invalidated` (forgotten or retention-expired)
  is history: direct reads and `timeline` answer, labelled, and serving surfaces
  do not. The dependency closure (`collect_dependency_closure`: BFS over
  `parent_id` + `cm_derived_from`, cycle-defended, refuses a truncated walk) is
  what makes invalidation and dirty-propagation transitive. A same-slot
  correction carries the revision its caller read and applies the supersede as
  a CAS (`UPDATE ... WHERE revision = :expected AND cm_superseded_by IS NULL`,
  savepoint stand-down) — two writers on one slot never leave two current facts
  and a stale snapshot answers `conflict`, never a silent supersede (MCP
  `correct_memory` supplies that snapshot; spec §8.2).
- **Forgetting (soft + hard).** MCP `forget_memory` is SOFT: it invalidates the
  closure and writes a `memory_suppressions` row per affected `source_ref`
  (plus the projection's upload-time `content_hash`), keeping every row, its
  content and its provenance; the vector point survives with its payload state
  refreshed to `invalidated` (R37/R38). A forgotten source blocks re-import on
  every write path (import/re-upload/reindex/outbox applier). Hard erase
  (`delete_memory`, REST erasure) still deletes rows + vectors with the
  verification receipt; explicit erasure wins over a pin and over an
  `invalidated` row. Known ceilings (measured) are in
  [OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md) §P4b.
- **Consolidation + retention (P4b).** The drain loop's post-round hook runs a
  budgeted consolidation producer (`app/retrieval/memory/consolidation.py`,
  rule `tag-summary.v1`): ≥2 servable memories per tag summarized into a row
  labeled `derived` with `cm_derived_from` / `cm_source_revisions` /
  `cm_rule_version` and a dedupe key that makes a re-run idempotent. The
  publish-time guard re-reads its sources and stands down (marking stale views
  `dirty`) when one moved — non-recursive, per-process idempotent, and never a
  competitor for `derived` rows. Retention is OFF by default and opt-in per user
  (`PATCH /api/v1/users/me/settings`, full-replace): rows the install has held
  past the user's window are invalidated with a `retention_expired` audit row,
  `pinned` rows are exempt, and the sweep is per ROW — it writes no suppression
  row and does not walk the closure, so it is an expiry, never a privacy
  guarantee over derived views.

## 2. MCP hub (`app/mcp_hub/`)

The agent-facing surface. MCP does not pass caller identity, so the hub is
where identity, permissions and audit live.

| Module | Role |
|---|---|
| `identity.py` | `AgentPrincipal` (user + agent client + scopes), token extraction (`Authorization: Bearer` or `X-Orivory-Agent-Token`), `resolve_principal` (sha256 lookup, active-only, touches `last_used_at`) |
| `server.py` | FastMCP (official `mcp` SDK, stateless HTTP, `json_response=True`) mounted at `/mcp` when `MCP_HUB_ENABLED`; ASGI scope-normalization adapter (Starlette 1.6 exact-path 307 would strip `Authorization`); `MCP_HUB_ALLOWED_HOSTS` → explicit `TransportSecuritySettings` for reverse proxies |
| `tools.py` | Eight tools: `search_memory`, `timeline`, `get_memory`, `list_recent`, `add_memory`, `correct_memory`, `delete_memory`, `forget_memory`. Each resolves its own principal, enforces scopes, and reads/writes only the caller's own namespace; every authorized call appends a `memory_access_logs` row (the ledger) |

**Identity model.** `agent_clients` registers an external agent: name,
`sha256` token hash (plaintext `oa_<32 hex>` shown exactly once), scopes
(`memory:read`, `memory:write`), status. MCP never transmits caller identity
— the per-client token IS the identity, resolved at the hub.

**Ledger.** `memory_access_logs` is append-only: one row per authorized tool
call (`mcp_search`, `mcp_get`, `mcp_list`, `mcp_add`, `mcp_correct`,
`mcp_delete`, `mcp_forget`) with principal attribution, plus `import` for an
agent-token upload and `retention_expired` from the retention sweep.
Ledger rows survive memory deletion (`memory_id` is SET NULL) — an audit
trail records that access happened before deletion.

## 3. Erasure receipts (`app/services/erasure_service.py`)

Right-to-be-forgotten with verification. `erase_memories(db, user_id,
memory_ids, *, requested_by)`:

1. **Ownership check** — foreign/missing ids are recorded
   (`not_found_or_foreign`), never deleted.
2. **Collect the closure BEFORE deleting** — descendants via BFS over
   `parent_id` (user-filtered, visited set, no silent depth cap; a closure
   past the `_MAX_CLOSURE_IDS` safety bound refuses the erase with
   `truncated=True`), the derived-memory set (`DerivedClosureError` is
   recorded as `derived_closure="unknown"`), entity/source link counts.
3. **One closure transaction** — the row delete for every affected id, one
   durable delete intent per id (`index_outbox`, same commit), a suppression
   row when a forgotten projection's source document still exists, and the
   user's now-orphaned entities + relations.
4. **Purge + verify** — `safe_delete_from_index` for every affected id,
   then re-query the vector index and re-count residual DB rows. Each target
   carries
   `vector_state`: `verified` / `pending` / `unknown` / `residual` (a failed
   purge stays `pending` — the intent is what retries it). Two namespace
   counters ride `db_residual` (P4a): `cascaded_out_of_namespace` (rows the FK
   cascade removes that the walk never collected — another namespace, or
   another user's) and `derived_out_of_namespace` (the erasing user's other
   namespaces); either one keeps the receipt from reaching `completed`. Their
   known ceilings are documented in
   [OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md) §P4a.
5. **Receipt** — one `erasure_receipts` row per call, with additive
   `detail.verification` and `detail.index_pending` (omitted when nothing was
   erased — no erase, no verification claim):
   `completed_with_errors` > `completed_with_residual` >
   `completed_unverified` (no positive presence readback; spec §5.4/P1 gate) >
   `completed` (positively verified) — rollup precedence, highest wins
   (per-target try/except + session rollback so remaining targets still
   process). Receipt-commit failure is the one documented unrecorded mode.

REST `DELETE /memories/{id}` and MCP `delete_memory` both call
`erase_memories`; document/session/conversation deletes enqueue memory-kind
delete intents in their own commit and purge vectors after it.

**Soft forget (P4b/T3) shares the receipt table, not the erasure.**
`soft_forget(db, user_id, memory_ids, *, requested_by)` invalidates the
transitive closure (`collect_dependency_closure` — multi-hop, refused when
truncated), suppresses every affected `source_ref`, enqueues a payload-refresh
upsert per invalidated row, and verifies by a serving-off readback
(`not_dirty_predicate()`): `completed` / `completed_with_residual` /
`completed_unverified` never claim more than that readback saw. Its detail
carries `mode: "soft"`; the hard receipt keeps its earlier shape. The walk it
does NOT share is the hard path's `parent_id` BFS + ONE `cm_derived_from` hop —
a derived view of a derived view survives a hard erase (measured; rule v1's
output is one hop deep, and the ceilings are in the runbook).

Honest v0 limits: verification is absence-checking (KG-correlation
re-inference probing is a follow-up).

## 4. Import paths (`app/ingestion/import_formats.py` + `import_service.py`)

`POST /api/v1/imports` accepts a provider export and turns it into
memories:

- **Adapters** — ChatGPT (`conversations.json` mapping-DAG → transcript),
  Claude (`chat_messages`/`sender`/`text` per PAM mappings), generic JSON
  array, PAM `memory-store.json`. Format-level shape errors raise
  `ImportFormatError` (→ 422); malformed-but-JSON **entries are skipped,
  never fatal** (`_safe_item` wraps every per-item conversion).
- **Cap** — final assembled content clipped to 10,000 chars with a
  truncation marker (one chokepoint, `_cap_content`, across all formats).
- **Service** — `run_import` dedups by `(user_id, source_type,
  source_ref)` (batched pre-insert SELECT + in-file `seen_refs`), batch
  creates, single commit, then best-effort `index_new_memory` per row
  (failures counted in `index_failures`, never rolled back).
- **Honest notes** — Rewind/Limitless have no adapter (SQLCipher-encrypted
  local SQLite, no official export); ChatGPT "Memory" feature contents are
  not in the data export; OpenRecall converts via one sqlite3 query
  (recipe in docs/API.md §9).

## 5. LLM seams (`app/agents/`)

There is no agent graph: the LangGraph chat workflow went with the chat surface.
What remains is the shared LLM plumbing that every server-side seam calls.

| Module | Role |
|---|---|
| `llm_client.py` | ONE client factory + `complete()` wrapper: api key / base URL / timeout, the `LLM_MAX_CONCURRENCY` shared gate, and a cost hook that records usage |
| `llm_parsing.py` | Structured-output parsing (`parse_llm_json_object`) |
| `state.py` | The `AgentState` TypedDict the retrieval seams pass around |

Callers: recall's query rewrite + entity extraction
(`app/retrieval/memory/query_rewriter.py` — a failed or malformed rewrite falls
back to the original query with an empty entity list), the graph write-back
(`app/graph/extraction.py`), the consolidation rule
(`app/retrieval/memory/consolidation.py`, `tag-summary.v1`) and HyDE
(`app/retrieval/hyde_agent.py`). `routing.py` holds the old graph's routing
helpers and has no caller left in the tree.

## 6. Evaluation (`eval/`)

- **RAG eval** — golden dataset + deterministic offline metrics
  (source-hit, keyword coverage, citation rate, fallback accuracy). One lane:
  `python eval/run_eval.py --mode offline`. See `docs/EVALUATION_GUIDE.md`.
- **Benchmarks** (`eval/benchmarks/`) — LongMemEval-S (primary; ICLR 2025)
  and MemoryAgentBench (secondary; the only benchmark scoring selective
  forgetting) adapters with a phased runner. The no-fabrication guarantee is
  structural: results files are written only from real runs, dataset sha256
  travels with every result, hygiene fields (judge version, full-context
  baseline, deviations) are reserved. Protocol rationale and the LoCoMo
  never-lead rule: [PAPERS_AGENT_MEMORY.md §3](https://github.com/twilightt1/orivory-private/blob/main/docs/research/PAPERS_AGENT_MEMORY.md) (private).

## 7. Data, storage and the schema ladder

- **One file, one store.** `DATABASE_URL` must be SQLite
  (`app/database.py` refuses any other URL); WAL mode, foreign keys enforced.
  Uploads go to the filesystem (`STORAGE_BACKEND=fs`) and vectors into the
  embedded Qdrant folder — all of it under `/data` in the container.
- **The ladder.** `bootstrap_sqlite()` runs in the app lifespan, before any
  traffic: a versioned ladder (v1 .. v7) upgrades an older file step by step and
  takes a `<db>.pre-pN.bak` milestone snapshot as it goes (never overwritten). A
  file NEWER than the code is refused rather than repaired
  (`unsupported SQLite schema version N; expected M`). What the steps added: v2
  the durable `index_outbox` / `index_generations` / `memory_suppressions`
  tables; v4 the FTS5 lexical index + its triggers; v5 `memories.namespace` +
  its index (P4a); v6 `memory_suppressions.namespace` and `content_hash`; v7
  `users.retention_enabled` / `retention_days`.
- **One process owns the vectors.** `QDRANT_MODE=local` (the default) refuses to
  boot when the launcher asked for more than one process — an embedded storage
  folder is exclusive.
- **Writes are durable before they are indexed.** Every canonical write stamps an
  `index_outbox` intent in the same commit; the drain loop replays whatever did
  not land against the current SQL state, and a recall that outlives its
  freshness budget answers a typed 503 instead of a false no-match. Operationally:
  [OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md).
- **Health.** `/health` is liveness; `/ready` is the per-check readiness payload
  (`sqlite`, `redis`, `storage`, `qdrant`, `mcp_hub`).
- **Backup.** `scripts/migrate_qdrant.py backup` (VACUUM INTO + sha256 manifest)
  and `verify --restore-drill`; the unit of backup is the whole `/data`
  directory. See [BACKUP_RESTORE.md](BACKUP_RESTORE.md).

## 8. REST surface map

| Route prefix | Purpose |
|---|---|
| `/api/v1/users` | The local owner: profile + retention settings |
| `/api/v1/memories` | Memory CRUD + recall + stats + digest |
| `/api/v1/agents` | Agent-token mint/list/revoke + the access ledger |
| `/api/v1/erasure-receipts` | Create/list/fetch erasure receipts |
| `/api/v1/imports` | One-shot provider-export upload |
| `/mcp` | MCP server (agents), mounted while `MCP_HUB_ENABLED` |
| `/health`, `/ready` | Liveness + readiness |

`app/api/v1/router.py` mounts exactly those five routers. The rest of the
second-brain REST surface — account auth, chat, admin, analytics, discovery,
entities, insights, referral, sources, workspaces, … — was removed with the
full-stack product; the deleted modules are in git history.

Full request/response reference: [API.md](API.md).
