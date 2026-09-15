# Orivory Architecture

> Single source of truth for how Orivory works. Supersedes the pre-pivot
> `architecture.md` / `TECHNICAL_ARCHITECTURE_v2.md` /
> `SOTA_TECHNICAL_SPECIFICATION.md` / `AI_ML_OVERVIEW.md` (removed — history
> lives in git). Last verified against the code: 2026-09-04.

Orivory is a **memory hub for AI agents**. One mental model:

```
                    ┌─────────────────────────────────────┐
   AI agents        │            Orivory (self-hosted)    │
   (Claude, Cursor, │                                     │
   OpenClaw, …) ────┼─▶ /mcp  ──▶ mcp_hub (scoped tools)  │
   second-brain  ───┼─▶ /api/v1  ──▶ REST (chat, memories,│
   web app       ───┤             imports, erasure, …)    │
                    │                                     │
                    │   memory store (Postgres + Qdrant)  │
                    │   knowledge graph · access ledger   │
                    └─────────────────────────────────────┘
```

Everything — REST and MCP — funnels into the same memory store, so the
second-brain web app and any connected agent share one brain.

## 1. The memory spine

The core claim: **one brain, many ways to ask.** Two worlds used to live
side by side (per-conversation documents vs per-user memories); the hub
unifies them.

- **`memories`** (Postgres) — the source of truth. Every memory carries
  `user_id`, `source_type` (manual_note, chatgpt_import, claude_import,
  generic_import, mcp_agent, conversation_excerpt, …), `source_ref`
  (dedup key), `content`, `tags`, and the salience fields
  (`salience`, `recall_count`, `last_used_at`).
- **Qdrant** — the vector index, one generation per kind and embedding
  contract (`orivory_memories__<contract>`, `orivory_chunks__<contract>`),
  written best-effort after every DB write (Postgres is truth; the reindex
  task rebuilds vectors from rows). Which generation is active per kind is a
  row in `index_generations`; a store that cannot name its contract is
  quarantined rather than served (P1b cutover, see §7).
- **BM25 + reranking** — hybrid recall: vector candidates + keyword
  candidates fused, then reranked.
- **Salience loop** — memories used in answers get bumped; untouched ones
  decay (periodic Celery beat). Ranking is salience × recency × relevance
  (the Generative-Agents scoring, reinforced on access).
- **Knowledge graph** — `entities` / `relations` extracted per memory;
  graph snapshot/related endpoints power the UI; graph context feeds RAG.
- **Forgetting (suppression/GC)** — a forgotten source gets a
  `memory_suppressions` row; it blocks re-import AND its projection points
  (chunk points, or a memory point whose source is suppressed) are deleted by
  the GC pass, so the content stops being vector-searchable. Deliberate
  read-path behaviour: absence is the feature, not a bug.

## 2. MCP hub (`app/mcp_hub/`)

The agent-facing surface. MCP does not pass caller identity, so the hub is
where identity, permissions and audit live.

| Module | Role |
|---|---|
| `identity.py` | `AgentPrincipal` (user + agent client + scopes), token extraction (`Authorization: Bearer` or `X-Orivory-Agent-Token`), `resolve_principal` (sha256 lookup, active-only, touches `last_used_at`) |
| `server.py` | FastMCP (official `mcp` SDK, stateless HTTP, `json_response=True`) mounted at `/mcp` when `MCP_HUB_ENABLED`; ASGI scope-normalization adapter (Starlette 1.6 exact-path 307 would strip `Authorization`); `MCP_HUB_ALLOWED_HOSTS` → explicit `TransportSecuritySettings` for reverse proxies |
| `tools.py` | Six tools: `search_memory`, `get_memory`, `list_recent`, `add_memory`, `delete_memory`, `forget_memory`. Each resolves its own principal, enforces scopes, appends a `memory_access_logs` row (the ledger) |

**Identity model.** `agent_clients` registers an external agent: name,
`sha256` token hash (plaintext `oa_<32 hex>` shown exactly once), scopes
(`memory:read`, `memory:write`), status. MCP never transmits caller identity
— the per-client token IS the identity, resolved at the hub.

**Ledger.** `memory_access_logs` is append-only: one row per authorized tool
call (`mcp_search/get/list/add/delete/forget`) with principal attribution.
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
   purge stays `pending` — the intent is what retries it).
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
  (recipe in docs/API.md §15).

## 5. Multi-agent RAG (`app/agents/`)

LangGraph pipeline with specialized agents (router → context → retrieval →
grounding → answer, plus evaluator / hallucination / feedback /
graph-context / discovery / insight agents) and **corrective RAG**:
self-evaluated retrieval quality, web-search fallback, hallucination
detection before delivery, per-answer grounding confidence surfaced in SSE
and persisted in `agent_trace` (admin quality-trend endpoint aggregates it).

Answer temperature is pinned to 0.0 for factual recall; contexts are budgeted
by characters before the LLM call; the fallback answer is an explicit
"I don't recall that in your memories" (never silent invention).

## 6. Evaluation (`eval/`)

- **RAG eval** — golden dataset + deterministic offline metrics
  (source-hit, keyword coverage, citation rate, fallback accuracy) and an
  opt-in live-API mode with SSE trace collection. See
  `docs/EVALUATION_GUIDE.md`.
- **Benchmarks** (`eval/benchmarks/`) — LongMemEval-S (primary; ICLR 2025)
  and MemoryAgentBench (secondary; the only benchmark scoring selective
  forgetting) adapters with a phased runner. The no-fabrication guarantee is
  structural: results files are written only from real runs, dataset sha256
  travels with every result, hygiene fields (judge version, full-context
  baseline, deviations) are reserved. Protocol rationale and the LoCoMo
  never-lead rule: [PAPERS_AGENT_MEMORY.md §3](https://github.com/twilightt1/orivory-private/blob/main/docs/research/PAPERS_AGENT_MEMORY.md) (private).

## 7. Data & migrations

- Postgres 16 (SQLAlchemy 2.0 async + Alembic), Redis 7 (cache/queue),
  Qdrant (vectors), MinIO (attachments).
- Migrations are part of `docker compose up`: the one-shot `migrate`
  service runs `alembic upgrade head`, and `app`/`celery_worker` gate on
  `service_completed_successfully` — a server can never start against a
  table-less database. Migrations were dry-run-verified on disposable
  Postgres 16 (upgrade / downgrade / re-upgrade / INSERT probes).
- Health: `/health` liveness; `/ready` per-dependency checks (postgres,
  redis, minio, qdrant, mcp_hub) with latencies and sanitized errors.
- **P1b migration before serving.** The vector store moved from Chroma to
  Qdrant and the embedding contract from masked mean to CLS. Until `cutover`
  flips the generation pointers the install keeps serving its OLD generation.
  Where that pointer names a contract the new code no longer matches (the
  lite/P1a transitional row: masked mean) the read path deliberately **fails
  loud** — it raises `EmbeddingDimensionMismatch` ("same dim but different
  embedding contract") instead of serving vectors it cannot verify. With NO
  active manifest row it does not: `outbox.active_generation()` falls back to
  the transitional generation name with no fingerprint, an EMPTY generation is
  allowed, and reads answer `[]`. That is the state on Postgres (P1a never
  seeded `index_generations` there) and on any install whose contract token did
  not change. SQL stays canonical; the migration rebuilds the vectors. The
  offline sequence — `inventory → backup → backfill → verify →
  cutover`, app stopped throughout — is in
  [OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md), and the one-release swap-back
  in [ROLLBACK_P1B.md](ROLLBACK_P1B.md).
- **Boot drain.** SQLite (lite) replay of pending `index_outbox` intents runs at
  startup; **Postgres deployments have no boot drain in P1b** — intents
  accumulate until P3 ships the worker-side drain. Write-through is unaffected
  (every write still embeds inline); only retries are deferred.

## 8. REST surface map

| Route prefix | Purpose |
|---|---|
| `/api/v1/auth`, `/users` | JWT + OAuth auth, registration, quotas |
| `/api/v1/chat` | Streaming RAG chat (SSE traces) |
| `/api/v1/memories` | Memory CRUD + recall + digest |
| `/api/v1/agents` | Agent client registration/revoke + access ledger |
| `/api/v1/erasure-receipts` | Create/list/fetch erasure receipts |
| `/api/v1/imports` | One-shot export upload |
| `/api/v1/entities`, `/sources`, `/insights`, `/discovery`, `/workspaces`, `/analytics`, `/referral` | Second-brain surfaces |
| `/mcp` | MCP server (agents) |
| `/health`, `/ready` | Liveness + readiness |

Full request/response reference: [API.md](API.md).
