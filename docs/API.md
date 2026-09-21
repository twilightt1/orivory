# Orivory API

An Orivory install is **one container**: the REST API, the MCP hub, SQLite, an
embedded Qdrant and the uploads directory, with no external services. It has
**one identity** — the local owner — and it is built to be reached by agents on
the host it runs on.

Base URL on a default install:

```
http://localhost:8000
```

Two surfaces:

| Surface | Routes | Credential |
|---|---|---|
| REST | `/api/v1/*`, `/health`, `/ready` | none — a request **without** an `Authorization` header IS the local owner |
| MCP | `/mcp` (streamable HTTP) | an agent token: `Authorization: Bearer oa_…` or `X-Orivory-Agent-Token: oa_…` |

`POST /api/v1/imports` is the one REST route that *also* accepts an agent token.

**The endpoint catalogue below is taken from the app's own schema**, which the
running install serves:

```bash
curl -s http://localhost:8000/openapi.json | python -m json.tool
```

Swagger UI (`GET /docs`) is served only when `ENVIRONMENT` is not
`production`. If this file and `/openapi.json` ever disagree, the schema wins.

## Table of contents

1. [Identity](#1-identity)
2. [Errors, rate limits and pagination](#2-errors-rate-limits-and-pagination)
3. [Health and readiness](#3-health-and-readiness)
4. [Memories](#4-memories)
5. [Users](#5-users)
6. [Agent clients and the access ledger](#6-agent-clients-and-the-access-ledger)
7. [MCP hub](#7-mcp-hub)
8. [Erasure receipts](#8-erasure-receipts)
9. [Import paths](#9-import-paths)
10. [Appendix: worked examples](#10-appendix-worked-examples)

---

## 1. Identity

There is **no account auth**: an Orivory install is one container with one
operator, so it has exactly one identity — the **local owner** (the `users`
row named by `LOCAL_OWNER_EMAIL`, default `owner@orivory.local`).

- A request with **no** Authorization header IS the local owner. Nothing to
  log in to, nothing to expire, no 401 for a missing token.
- A request with an **agent token** (`Authorization: Bearer oa_…`, minted by
  `POST /api/v1/agents`) acts as that client on the **MCP endpoint** and on
  `POST /api/v1/imports`, which is where the token's scopes
  (`memory:read` / `memory:write`) are enforced and where every write is
  recorded in the access ledger under it. The REST API itself takes no token:
  a Bearer value on a REST route is `401` (unknown, revoked, or simply the
  wrong surface — serving its owner there would bypass scopes).

Register / login / email verification / OAuth / password reset / JWT were
removed with account auth; those endpoints no longer exist. The same wave
removed chat + SSE, admin, analytics, experiments, discovery, sources,
workspaces, system_settings, entities and the knowledge-graph routes: the
routes in this file are the whole surface. `GET /openapi.json` lists it.

---

## 2. Errors, rate limits and pagination

### Errors

Errors use FastAPI's standard `{"detail": ...}` shape, except the readiness
answers of `POST /api/v1/memories/recall`, which are typed:

| Status | Body | Raised by |
|---|---|---|
| 400 | `{"detail": "unknown scope: … (allowed: ('memory:read', 'memory:write'))"}` | `POST /api/v1/agents` |
| 401 | `{"detail": "The REST API serves the local owner; agent tokens are used with the MCP endpoint and POST /api/v1/imports."}` | any REST route called with an `Authorization` header |
| 403 | `{"detail": "Agent token lacks the memory:write scope."}` | `POST /api/v1/imports` called with a read-only token |
| 404 | `{"detail": "Memory not found."}` · `"Erasure receipt not found."` · `"Agent client not found."` · `"Memory not found or not shared"` | owned-resource lookups — a foreign or unknown id reads `404`, never an existence leak |
| 413 | `{"detail": "Import file exceeds the 20 MiB synchronous cap."}` | `POST /api/v1/imports` |
| 422 | FastAPI validation body, or a parse message (`unknown source_format: …`, `Uploaded file is empty.`) | bad query/body/form, undecodable or unparseable import |
| 429 | `{"detail": {"error": "rate_limit_exceeded", "retry_after": 60, "limit": 60}}` | the recall guard's per-60-second window (below) |
| 429 | `{"detail": "Daily quota exceeded."}` · `"Monthly quota exceeded."` | the same guard's quota half (`app/services/quota_service.py`) |
| 503 | `{"error": "embedding_contract_mismatch"}` · `{"error": "vector_unavailable"}` · `{"error": "index_freshness_timeout"}` | `POST /api/v1/memories/recall` — a readiness answer instead of a silent empty recall |

Every response carries `X-Request-ID` (the logging middleware sets one when the
request did not bring it).

### Rate limits and quota

The guard (`app/utils/dependencies.py:enforce_llm_quota`) runs on
`POST /api/v1/memories/recall`, the route that spends embedding/LLM calls. It
is keyed on the owner, not on a token, because the owner is the only principal:

- per-60-second window: `RATE_LIMIT_PER_MINUTE` (default **60**), enforced by
  `app/middleware/rate_limiter.py:check_rate_limit` → the `rate_limit_exceeded`
  body above;
- the quota half of the same guard (`quota_service.check_and_increment`) checks a
  `user_quotas` row when the database has one: `daily_limit` (server default
  **100**) → `429 {"detail": "Daily quota exceeded."}`, `monthly_limit` (server
  default **2000**) → `"Monthly quota exceeded."`. No row means unlimited, and
  nothing in this tree creates one — a fresh install is bounded by the window
  alone.

No `X-RateLimit-*` headers are returned; the window `429` carries `retry_after`
and `limit`, the quota `429` is a plain `detail` string. (The per-tier
Free/Pro/Enterprise table the previous catalogue documented belonged to the
account product and is gone.)

### Pagination

List routes page with `limit` / `offset` — there are no cursors:

| Route | `limit` (default / max) | `offset` | Order |
|---|---|---|---|
| `GET /api/v1/memories` | 50 / 200 | yes (`ge=0`) | `sort=newest` (default) · `salience` · `last_used` |
| `GET /api/v1/agents` | no paging — every client | — | newest first |
| `GET /api/v1/agents/access-log` | 50 / 200 | yes | newest first |
| `GET /api/v1/erasure-receipts` | 50 / 200 | yes | newest first |

`items` come back with a `total`, counted with the same predicate as the page.

---

## 3. Health and readiness

| Route | Answers |
|---|---|
| `GET /health` | `{"status": "ok", "version": "1.1.0"}` — liveness; touches no dependency |
| `GET /ready` | the readiness payload with `200` when every check passes, `503` when one fails |

`/ready` checks the SQLite connection (`SELECT 1`), the vector store (in the
shipped lite shape, opening the process's ONE embedded Qdrant client is the
check; server mode polls `{QDRANT_URL}/readyz` with a 2-second bound), the
in-memory Redis shim (`ping`), and the upload storage backend.

---

## 4. Memories

The memory spine: create, list, read, update, delete, digest, recall, stats and
the public share link. All routes here serve the owner's own rows; a foreign id
is a `404`.

### POST /api/v1/memories

Create one memory. Returns `201` with the `MemoryResponse`.

| Field | Type | Notes |
|---|---|---|
| `content` | string | **required** |
| `title` | string ≤ 500 | optional |
| `summary` | string ≤ 4000 | optional |
| `source_type` | string | default `manual_note` |
| `source_ref` / `source_url` | string ≤ 500 / ≤ 1000 | optional |
| `tags` | string[] | optional |
| `captured_at` | datetime | optional |
| `parent_id` | uuid | must be an existing memory of the owner, else `404` |
| `pinned` | bool | default `false` |
| `metadata` | object | client metadata; server-owned `cm_*` keys are preserved |
| `auto_compress` | bool | default `false` |

```bash
curl -s -X POST http://localhost:8000/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{"content": "Orivory stores memories in SQLite and Qdrant.", "title": "Stack", "tags": ["stack"]}'
```

### GET /api/v1/memories

List the owner's memories, filtered and paged:

| Query | Type | Notes |
|---|---|---|
| `source_type` | enum | `manual_note`, `file_upload`, `google_drive`, `notion`, `gmail`, `web_clipper`, `rss`, `conversation_excerpt`, `chatgpt_import`, `claude_import`, `gemini_import`, `copilot_import`, `openclaw_import`, `generic_import`, `other` |
| `tag` | string | exact match |
| `query` | string | case-insensitive substring in title OR content |
| `pinned` | bool | optional |
| `sort` | enum | `newest` (default) · `salience` · `last_used` |
| `limit` / `offset` | int | 50 (max 200) / 0 |

Dirty rows are never listed; superseded rows are listed with
`state="superseded"` — history stays readable, labelled.

### GET /api/v1/memories/{memory_id}

One memory (`MemoryResponse`), or `404`.

### PATCH /api/v1/memories/{memory_id}

Partial update — send only the fields you are changing (`title`, `summary`,
`tags`, `salience`, `pinned`, `metadata`). Bumps `revision`, enqueues the
re-index intent, and writes through to the vector store best-effort: the
response's `indexing` is `"ready"` when the vector write landed, `"pending"`
when the durable outbox owns it, and omitted on a no-op PATCH.

### DELETE /api/v1/memories/{memory_id}

`204`. Runs the durable erasure path (row + derived closure + vectors, with a
receipt) — see §8.

### GET /api/v1/memories/digest

What you saved recently plus "on this day" resurfacing. `window_days` is
`7` by default (`ge=1, le=90`). Returns `generated_at`, `window_days`,
`recent_count`, `top_themes[]`, `recent_memories[]` and `resurfaced[]`
(each with `memory`, `age_label`, `age_days`).

### GET /api/v1/memories/stats

Aggregate counts for a dashboard, computed over the owner's visible rows:
`total_memories`, `entities`, `relationships`, `observations`, `concepts`,
`recent_activity[]`, `top_tags[]`.

### POST /api/v1/memories/recall

The retrieval entry point — **the only route with the rate limit/quota guard**.

```json
{"query": "what did I say about SQLite?", "top_k": 10, "include_personal_context": true}
```

| Field | Type | Default |
|---|---|---|
| `query` | string | required |
| `top_k` | int | 10 |
| `include_personal_context` | bool | true |

Pipeline: personal context (pinned + recent) → LLM query rewrite + entity
extraction → vector search in Qdrant → hydrate, entity boost, time decay →
`top_k` results with a `trace` (rewritten query, entities, `latency_ms`,
`num_candidates`, `num_results`, `stage_ms`, `counts`, rewrite fallback flags).

Every step degrades gracefully (empty `results` plus a `trace`) with the three
typed `503`s in §2 as the exceptions. On SQLite, a vector outage answers from
the FTS5 lexical leg with `trace.counts["lexical"]` set and no `dense` key
rather than refusing.

### GET /api/v1/memories/{memory_id}/share

**Public, no auth.** Returns `SharedMemoryResponse` (`id`, `title`, `content`,
`summary`, `tags`, `created_at`, `source_type`) for a memory with
`is_shared=true` in the personal namespace; anything else is `404`.

---

## 5. Users

### GET /api/v1/users/me

The local owner: `id`, `email`, `display_name`, `avatar_url`, `auth_provider`,
`role`, `is_active`, `is_deleted`, `is_verified`, `onboarding_done`,
`retention_enabled`, `retention_days`, `created_at`.

### PATCH /api/v1/users/me/settings

Retention settings for the owner:

| Field | Type | Notes |
|---|---|---|
| `retention_enabled` | bool | default `false` |
| `retention_days` | int \| null | `> 0`, max 36500 |

Returns the stored `{"retention_enabled": …, "retention_days": …}`.

---

## 6. Agent clients and the access ledger

The Open Memory Hub lets external AI agents (Claude Desktop, Claude Code, OpenClaw, Cursor, custom agents) read and write your memory over MCP (Model Context Protocol). Access is controlled by **agent clients**: per-agent tokens with scoped permissions (`memory:read` / `memory:write`), and every authorized call is recorded in an **access ledger** — which AI read or wrote what, and when.

> **Note:** The MCP endpoint is mounted at `/mcp` and is gated by the `MCP_HUB_ENABLED` setting (`true` by default; set `MCP_HUB_ENABLED=false` in `.env` to disable it).

### POST /api/v1/agents

Register an agent client. Returns the plaintext token — **shown exactly once**. Only a SHA-256 hash is stored; the token cannot be recovered or re-displayed afterwards.

**Request:**

```bash
curl -s -X POST http://localhost:8000/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{"name": "Claude Desktop", "scopes": ["memory:read", "memory:write"]}'
```

| Field    | Type     | Required | Description                                                       |
|----------|----------|----------|-------------------------------------------------------------------|
| `name`   | string   | Yes      | Display name (max 100 chars)                                      |
| `scopes` | string[] | No       | `memory:read` (search/get/list), `memory:write` (add/delete). Defaults to `["memory:read"]` |

**Response `201 Created`:**

```json
{
  "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "name": "Claude Desktop",
  "scopes": ["memory:read", "memory:write"],
  "status": "active",
  "created_at": "2026-09-02T10:30:00Z",
  "token": "oa_9f8e7d6c5b4a3210fedcba9876543210"
}
```

> **Warning: the token is shown once.** Copy it immediately and store it in your MCP client's config — it will never appear in any API response again. Anyone who loses it must revoke the client and register a new one.

### GET /api/v1/agents

List the owner's agent clients, newest first. Never includes token material.

**Response `200 OK`:**

```json
{
  "items": [
    {
      "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
      "name": "Claude Desktop",
      "scopes": ["memory:read", "memory:write"],
      "status": "active",
      "created_at": "2026-09-02T10:30:00Z",
      "last_used_at": "2026-09-02T14:22:00Z",
      "revoked_at": null
    }
  ],
  "total": 1
}
```

### DELETE /api/v1/agents/{client_id}

Revoke an agent client by `client_id` (uuid). The token stops working immediately and revocation is idempotent (re-revoking the caller's own already-revoked client still returns `204`).

**Response `204 No Content`**

### GET /api/v1/agents/access-log

The access ledger: every authorized MCP call, newest first. Which agent did what, when.

**Query Parameters:**

| Parameter         | Type    | Default | Description                              |
|-------------------|---------|---------|------------------------------------------|
| `agent_client_id` | string  | null    | Filter by agent client ID                |
| `limit`           | integer | 50      | Items per page (max 200)                 |
| `offset`          | integer | 0       | Offset for pagination                    |

**Response `200 OK`:**

```json
{
  "items": [
    {
      "id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
      "agent_client_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
      "action": "mcp_search",
      "memory_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
      "detail": {"query": "CRISPR specificity"},
      "created_at": "2026-09-02T14:22:00Z"
    }
  ],
  "total": 1
}
```

Ledger `action` values: `mcp_search`, `mcp_get`, `mcp_list`, `mcp_add`, `mcp_correct`, `mcp_delete`, `mcp_forget`, plus `import` for `POST /api/v1/imports` calls made with an agent token. `memory_id` is null for search/list actions.

---

## 7. MCP hub

The hub speaks **streamable HTTP MCP** at:

```
http://localhost:8000/mcp
```

Authentication uses your agent token in either header:

```http
Authorization: Bearer oa_9f8e7d6c5b4a3210fedcba9876543210
```

```http
X-Orivory-Agent-Token: oa_9f8e7d6c5b4a3210fedcba9876543210
```

Requests without a valid token — or with a revoked token — are rejected before any tool runs; only authorized calls are ledgered.

### Tools

| Tool            | Scope         | Description                                          |
|-----------------|---------------|------------------------------------------------------|
| `search_memory` | `memory:read` | Semantic search over the caller's memory hub          |
| `timeline`      | `memory:read` | Anchor + chronological neighbours around one memory   |
| `get_memory`    | `memory:read` | Fetch one memory by ID (full content + provenance, any state) |
| `list_recent`   | `memory:read` | List the caller's most recent memories                |
| `add_memory`    | `memory:write`| Store a new memory (title, content, optional tags)    |
| `correct_memory`| `memory:write`| Supersede a fact (correction) with provenance — a same-slot race answers `status: "conflict"` (see below) |
| `delete_memory` | `memory:write`| Delete one memory by ID (hard erase + receipt)        |
| `forget_memory` | `memory:write`| Soft-forget memories: invalidate them and pin their sources against re-import (see below and [§8](#8-erasure-receipts)) |

Every tool is scoped to the caller's namespace as well as to the token's
owner, and `timeline` filters its neighbours by that
namespace and by the dirty rule — a stale derived row no longer appears beside
the anchor, while superseded rows still do (labelled). A revoked token (or one
whose owner lost the scope) is refused before any tool runs, and the boundary is
enforced again below the identity layer, on the SQL that reads the rows.

Scopes are enforced per call: a token with only `memory:read` cannot `add_memory` or `delete_memory`.

### correct_memory: the correction state machine

`correct_memory` never overwrites: it creates a new version and, when the fact
is a slot it can match (same normalized `subject` + `attribute` + `scope`), it
supersedes the current one. `status` is one of:

| Status | Meaning |
|---|---|
| `added` | No exact candidate: the fact is new to the slot (or outside normalization) |
| `superseded` | The matched candidate(s) now point at the new version |
| `needs-check` | Ambiguous (empty scope with a sibling, wrong target, unparseable `valid_from`, a late old import): both rows stay, nothing is superseded |
| `conflict` | P4b CAS: the slot moved after this tool read it (or holds a candidate the caller did not name). The new version lands flagged `needs-check` and NOTHING is superseded — read the slot again and decide |

The CAS is the reason `status` grew a value: an explicit `memory_id` carries the
revision this tool just read of that row, and the supersede applies only if the
slot is still where the read found it. A second writer that lands in that
window therefore sees `conflict` instead of silently un-learning the winner. A
slot holding MORE than one exact candidate is refused the same way — the tool
never supersedes a candidate the caller did not name (the whole apply stands
down, the named candidate included).

### Connecting an MCP client

```json
{
  "mcpServers": {
    "orivory-memory": {
      "type": "http",
      "url": "http://localhost:8000/mcp",
      "headers": {
        "X-Orivory-Agent-Token": "oa_9f8e7d6c5b4a3210fedcba9876543210"
      }
    }
  }
}
```

> **Note — DNS-rebinding protection is on by default (localhost-only).** `/mcp` is constructed without explicit transport security, and in `mcp` 1.29.x FastMCP defaults `host="127.0.0.1"`, which auto-enables localhost-only DNS-rebind protection — so **any non-localhost `Host` header is answered `421` as shipped** (connection attempts against a public hostname will fail until configured).
>
> To accept other hostnames — e.g. behind a reverse proxy that forwards a public `Host` — set `MCP_HUB_ALLOWED_HOSTS=your.host.example` in the environment (comma-separated for several). This explicitly enables `TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=[...])` on the FastMCP instance.
>
> Until configured, keep the endpoint localhost-bound or proxy with `Host` preservation pointing at localhost.

---

## 8. Erasure receipts

Erasing a memory removes the row **and every derived artifact** (child memories, entity links, source links, vector-store entries), then runs a post-deletion verification pass: re-query the vector store and re-count residual DB rows per target. Each erasure call returns one **receipt** with per-target detail. Receipts are scoped to the owner and are deleted with the user row.

> **Two shapes share this receipt table (P4b).** The erasure endpoints and MCP `delete_memory` are HARD: rows deleted, vectors purged, absence positively verified. MCP `forget_memory` is SOFT ([§7](#7-mcp-hub)): rows are invalidated in place, every affected source is suppressed against re-import, and the verification is a serving-off readback. A soft receipt carries `detail.mode: "soft"` (hard receipts keep their earlier shape — no `mode` key) and the shared status vocabulary reads differently for it: `completed` means "no target is visible to a serving surface any more", never "the rows are gone".

> **Honest v0 verification:** v0 verifies erasure by **absence-checks** — the receipt confirms that vectors and DB rows are *gone*. It does not probe whether facts can be re-inferred from correlated knowledge-graph data (KG-correlation re-inference probing is a planned follow-up). Also note that `Entity`/`Relation` nodes themselves survive memory erasure in v0 (link counts are recorded in the receipt; orphan pruning is a follow-up). Don't market this as "adversarially verified" until the deeper protocol ships.

### POST /api/v1/erasure-receipts

Erase memories owned by the caller. Foreign or unknown ids are recorded in the receipt as `"not_found_or_foreign"` — never deleted, no existence leak. Erasure is best-effort per target: one failing target never aborts the other targets.

**Request:**

```bash
curl -s -X POST http://localhost:8000/api/v1/erasure-receipts \
  -H "Content-Type: application/json" \
  -d '{"memory_ids": ["3fa85f64-5717-4562-b3fc-2c963f66afa6"]}'
```

| Field        | Type   | Required | Description                                    |
|--------------|--------|----------|------------------------------------------------|
| `memory_ids` | UUID[] | Yes      | 1–100 memory ids (duplicates are deduplicated) |

**Response `201 Created`:** one receipt. `detail.targets[]` carries one entry per requested id — `affected_memory_ids` (transitive descendants erased with the target), `traversal_depth`, `entity_links` / `source_links` counts, `vectors_deleted`, `vector_residual`, `vector_residual_checked`, and `db_residual`:

```json
{
  "id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "user_id": "5f0b3a2e-1c4d-4e8f-9a7b-2d6c8e1f0a3b",
  "requested_memory_ids": ["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
  "status": "completed",
  "detail": {
    "requested_by": "rest_api",
    "targets": [
      {
        "memory_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
        "status": "deleted",
        "affected_memory_ids": [],
        "traversal_depth": 0,
        "entity_links": 2,
        "source_links": 1,
        "vectors_deleted": ["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
        "vector_residual": [],
        "vector_residual_checked": true,
        "db_residual": {"children": 0, "entity_links": 0, "source_links": 0}
      }
    ],
    "summary": {"requested": 1, "erased": 1, "skipped": 0, "errors": 0, "residual_vectors": 0, "residual_rows": 0}
  },
  "created_at": "2026-09-02T14:22:00Z"
}
```

**Receipt status semantics:**

| Status                    | Meaning                                                              |
|---------------------------|----------------------------------------------------------------------|
| `completed`               | Every target erased and positively verified — absence readback confirmed for every deleted target |
| `completed_unverified`    | Erasure succeeded and found no residual, but no positive presence readback (a target is `pending` or `unknown`); spec §5.4 / P1 gate: `completed` is never stored without one |
| `completed_with_residual` | Erasure succeeded, but the verification pass found leftover vectors or DB rows |
| `completed_with_errors`   | At least one target's erasure raised; the error is recorded per-target and the remaining targets were still erased |

Rollup precedence: `completed_with_errors` > `completed_with_residual` > `completed_unverified` > `completed`. `detail.verification`/`detail.index_pending` are omitted when the call erased nothing (a forget that deleted nothing verified nothing).

The no-op branch of a **soft** forget follows the same rule: a call that invalidated nothing (every requested id foreign or missing) reports `completed_unverified` — nothing was invalidated, so nothing could be verified, and it never claims `completed`. (The hard endpoints keep their earlier shape: nothing erased, no per-target verification, `completed`.)

`vector_residual_checked: false` means the vector-store re-query was unavailable during verification (the DB delete still succeeded — the SQL row is the source of truth). A `false` flag alone does not imply residual data.

**Reconciliation.** Open receipts (`completed_unverified`) are re-checked by the reconcile pass that rides the drain loop (`reconcile_erasure_receipts`, `app/retrieval/memory/drain_loop.py`), OLDEST first (`created_at ASC`, ties broken by id; at most 200 per pass — FIFO, so a sustained erase load cannot starve an old receipt out of the window). A pass that reads the index clean rewrites the SAME receipt to `completed` with the re-verified evidence. A pass that still finds residual vectors does NOT relabel the receipt: it records what it observed in `detail` (`vector_residual_checked: true`, `vector_residual: [...]`) and leaves `status` alone, so the receipt stays open and re-checkable by a later pass — never frozen into the terminal `completed_with_residual`. A pass whose readback failed observed nothing and writes nothing at all: the receipt is byte-identical afterwards. The pass is upgrade-only: `completed`, `completed_with_residual` and `completed_with_errors` receipts are not even scanned, and no open receipt is ever downgraded.

### GET /api/v1/erasure-receipts

List the owner's receipts, newest first.

**Query Parameters:**

| Parameter | Type    | Default | Description                              |
|-----------|---------|---------|------------------------------------------|
| `limit`   | integer | 50      | Items per page (max 200)                 |
| `offset`  | integer | 0       | Offset for pagination                    |

**Response `200 OK`:**

```json
{
  "items": [
    {
      "id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
      "user_id": "5f0b3a2e-1c4d-4e8f-9a7b-2d6c8e1f0a3b",
      "requested_memory_ids": ["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
      "status": "completed",
      "detail": {"targets": [], "summary": {}},
      "created_at": "2026-09-02T14:22:00Z"
    }
  ],
  "total": 1
}
```

```bash
curl -s "http://localhost:8000/api/v1/erasure-receipts?limit=50&offset=0"
```

### GET /api/v1/erasure-receipts/{id}

Fetch one receipt. Unknown or other users' receipts return `404` (no existence leak).

```bash
curl -s http://localhost:8000/api/v1/erasure-receipts/7c9e6679-7425-40de-944b-e07fc1f90ae7
```

### MCP: forget_memory

Agents with the `memory:write` scope can call the `forget_memory` MCP tool (endpoint `/mcp`, see [§7](#7-mcp-hub)) with `{"memory_ids": ["<uuid>", ...]}`. It returns a compact summary — `receipt_id`, `status`, `invalidated`, `suppressed`, `skipped`, `invalid` — instead of the full receipt; fetch the receipt via `GET /api/v1/erasure-receipts/{id}` for the per-target detail (`detail.mode == "soft"`, `detail.targets[].affected_memory_ids` = the invalidated closure). Every authorized call appends an `mcp_forget` row to the access ledger pointing at the receipt. Ids outside the caller's namespace are resolved out before the service sees them and counted in `skipped` — the same answer a missing id gets, so there is never an existence leak.

**`forget_memory` is SOFT (P4b, spec §12/§5.4).** It invalidates the target AND its transitive closure (parent + `cm_derived_from`) and writes a `memory_suppressions` row for every affected `source_ref`, so the content stops being served and cannot be re-imported. What it does NOT do:

- it does not delete rows — content, tags, provenance and evidence links stay (`state: "invalidated"`, readable through `get_memory`/`timeline` and labelled history);
- it does not purge the vector at once: the point stays, its payload state is refreshed to `invalidated` by the index drain (R37), and the SQL visibility rule is what closes serving immediately;
- `invalidated` counts the REQUESTED ids this call invalidated (one per id that reached the service) — not the closure size; `suppressed` counts every affected source, so it can exceed `invalidated`.

Hard deletion (row + descendants + vectors, with the full verification receipt) is `delete_memory` (MCP) or `DELETE /api/v1/memories/{id}` and `POST /api/v1/erasure-receipts` (REST). An explicit erase wins over a pin AND over an invalidated row.

> **Note — ledger rows survive erasure by design.** The access ledger is append-only and is never erased by an erasure call: it records that a memory was *accessed before* deletion, which is exactly what makes it an audit trail. Receipts, by contrast, quote personal memory ids and are deleted with the user.

---

## 9. Import paths

Bring an existing AI assistant's memory into Orivory in one call. The endpoint accepts a raw export file, auto-detects (or takes an explicit format), normalizes it into memories, and returns a summary.

### POST /api/v1/imports

```bash
curl -X POST http://localhost:8000/api/v1/imports \
  -F "file=@conversations.json" \
  -F "source_format=chatgpt"
```

| Form field | Required | Description |
|---|---|---|
| `file` | yes | The raw export file (JSON). Max **20 MiB** (larger → `413`). |
| `source_format` | no | `auto` (same as omitted/blank — detection) · `chatgpt` · `claude` · `gemini` · `copilot` · `openclaw` · `generic`. An explicit unknown value → `422`. |

**Accepted formats** (detection heuristics follow the [PAM importer mappings](https://github.com/portable-ai-memory/portable-ai-memory/blob/master/importer-mappings.md)):

| Format | Detected shape | Notes |
|---|---|---|
| `chatgpt` | OpenAI export: array of conversations with a `mapping` DAG of `{author.role, content.parts}` nodes | One memory per conversation; system/empty turns skipped; turns ordered by `create_time` |
| `claude` | Claude export: `chat_messages` with `sender` human/assistant and message-level `text` | Thinking/tool blocks dropped |
| `generic` | JSON array of `{title?, content, created_at?, url?, ref?, tags?}` | The escape hatch for anything else |
| PAM bundle | `{"schema": "portable-ai-memory", "memories": [...]}` | [Portable AI Memory](https://portable-ai-memory.org/spec/v1.0/) `memory-store.json` (auto-detects to `generic`) |
| `gemini` / `copilot` / `openclaw` | provider conversation / session dumps | Dedicated adapters in `app/ingestion/import_formats.py` — same defensive contract as the others: malformed entries are skipped, never fatal |

**Response `201`** — `ImportSummary`:

```json
{
  "parsed": 42,
  "created": 40,
  "skipped_duplicates": 2,
  "suppressed_skipped": 0,
  "failed": 0,
  "index_failures": 0
}
```

`suppressed_skipped` (P4b/T4) counts items whose source identity the user
FORGOT: the suppression ledger blocked the re-import, so they are neither
created nor folded into `skipped_duplicates`. An item that is both a duplicate
and suppressed stays attributed to the dedup check (it is read first) — the
order is the one the counters are read in, not a claim about intent.

Duplicates are detected per `(user, source_type, source_ref)` — re-uploading the same export skips what you already imported. Dedup runs per-request (select-then-insert): two concurrent uploads of the same file can both succeed, and generic items without a `ref` field are re-created on every re-upload (a unique index is the planned hardening). Conversation content is clipped at **10,000 characters** (truncation marker appended). Malformed entries inside an otherwise-valid file are **skipped, never fatal** — one bad conversation can't fail the whole import. Embedding is best-effort: `index_failures > 0` means those memories exist and are searchable by keyword but not yet vector-indexed (the SQL row is the source of truth; the drain loop recovers the vector).

**Errors:** `422` — unparseable file, undetectable format, explicit unknown format, undecodable bytes · `413` — file over 20 MiB.

> **Honest notes:**
> - **Rewind/Limitless: no adapter.** Their local history lives in a SQLCipher-encrypted SQLite database with no official export format (the app was sunset 2025-12-19). Convert manually to the generic JSON shape, or wait for a dedicated adapter.
> - **ChatGPT "Memory" feature contents are NOT in the data export** — only conversations.
> - **OpenRecall** stores an unencrypted local SQLite; one `sqlite3` query converts it to the generic JSON shape:
>
>   ```bash
>   sqlite3 recall.db "SELECT json_group_array(json_object('content', text, 'created_at', timestamp, 'title', title)) FROM entries" > openrecall.json
>   ```

---

## 10. Appendix: worked examples

Every request below runs against a default local install
(`http://localhost:8000`), with no credentials: the REST surface serves the
local owner.

**Store a memory and recall it**

```bash
curl -s -X POST http://localhost:8000/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{"title": "Stack", "content": "Orivory keeps SQLite as the canonical store and an embedded Qdrant for vectors.", "tags": ["stack"]}'

curl -s -X POST http://localhost:8000/api/v1/memories/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "where are vectors stored?", "top_k": 5}'
```

**Mint an agent token, then list the ledger**

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{"name": "Claude Desktop", "scopes": ["memory:read", "memory:write"]}' \
  | python -c "import json,sys; print(json.load(sys.stdin)['token'])")

curl -s "http://localhost:8000/api/v1/agents/access-log?limit=20"
```

The token is shown in that response and never again — only its SHA-256 hash is
stored. Revoke it with `DELETE /api/v1/agents/{client_id}`.

**Check readiness and version**

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/ready
```

**Register the MCP endpoint in an MCP client**

```json
{
  "mcpServers": {
    "orivory-memory": {
      "type": "http",
      "url": "http://localhost:8000/mcp",
      "headers": {
        "X-Orivory-Agent-Token": "oa_9f8e7d6c5b4a3210fedcba9876543210"
      }
    }
  }
}
```

**Import an export file**

```bash
curl -s -X POST http://localhost:8000/api/v1/imports \
  -F "file=@conversations.json" \
  -F "source_format=chatgpt"
```

**Erasure and its receipt**

```bash
curl -s -X POST http://localhost:8000/api/v1/erasure-receipts \
  -H "Content-Type: application/json" \
  -d '{"memory_ids": ["3fa85f64-5717-4562-b3fc-2c963f66afa6"]}'

curl -s "http://localhost:8000/api/v1/erasure-receipts?limit=50"
```
