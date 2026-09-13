# Changelog

All notable changes to Orivory are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/) and the
project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased] — Correctable Memory V1 (2026-09-12)

### Added
- **Evidence-first correctable memory** — `cm_*` metadata on `Memory`
  (subject/attribute/scope, valid_from, supersede chain, evidence IDs,
  derived_from/dirty), single atomic `resolve_correction` write path,
  recall hides superseded/dirty with `include_history` opt-in
  (spec: `docs/superpowers/specs/2026-09-12-correctable-memory-v1.md`).
- **`correct_memory` MCP tool** (7th tool) — sửa fact có evidence,
  tạo bản mới + link chain, không overwrite; mơ hồ → `needs-check`.
  `get_memory` trả provenance, `search_memory` trả `state` mỗi hit.
- **Latency V1** — fast-path skip LLM rewrite khi query rõ, bounded
  `timeline` queries (row-value `tuple_`), `RecallTrace` có `stage_ms`
  + `rewrite_skipped`. Không đổi stack.
- **ZeroMem trial (REVERTED)** — lexical-refinement + route-weights +
  same-slot-closure (0 LLM call) đo n=100 cùng protocol: 0.470
  (CI [0.375, 0.567]) vs 0.510 baseline, multi-session 13→10.
  Không tăng → revert cả 3, giữ code ngoài tree. Bài học: rerank tín
  hiệu trên pool top_k×3 đã bão hòa; muốn nhích phải đổi pool
  (retrieval), không phải sort lại pool cũ.
- **arctic-embed-xs local (DEFAULT)** — Snowflake xs fp32 90MB, query-prefix
  only, 384-dim, backend `local-arctic`. H1 probe tốt nhất trục local
  (update 0.545 vs stale 0.451, +0.094). n=100 store tươi: **0.490**
  (CI [0.394, 0.587]) — ngang band MiniLM, không breakthrough.
  Pattern đáng chú ý: single-session mạnh lên (user 12/13, assistant 14/14)
  nhưng multi-session sập 14→9 ở CẢ e5 lẫn arctic → nghi pool thiếu
  diversity (model single giỏi hơn lấn át evidence cross-session),
  không phải chất lượng embed. Hướng tiếp: điều tra multi riêng.
- **e5-multilingual local (OPT-IN, not default)** — custom ONNX int8
  (Xenova quantized 118MB, prefix query:/passage: cả hai phía, 384-dim,
  backend `local-e5` tách khỏi `local`) đo n=100 store tươi: **0.430**
  (CI [0.337, 0.528]) vs MiniLM 0.470–0.510, multi-session rớt 14→9.
  Kết luận: quantized e5 thua MiniLM trên bench tiếng Anh (nghi ngờ
  quantization + prefix-compression; bản fp32 470MB chưa thử vì vượt
  ngân sách Lite). Default flip về `minilm`; e5 giữ lại cho user Việt
  qua `LOCAL_EMBED_MODEL=e5`.
- **Local-first env** — `.env.example`: `USE_LOCAL_EMBEDDINGS=true`
  (ONNX MiniLM, no key/no cost), `RETRIEVAL_SEMANTIC_RERANK=false`
  (opt-in khi eval chứng minh cần).

### Removed
- **`frontend/`** — Next.js app khỏi tree + compose + CI (Lite không
  ship nó; agent là UI).
- **LangGraph agents** — 16 agent files khỏi `app/agents/` (giữ
  `llm_client` + `llm_parsing` + `state` + `routing` seams); test mồ
  côi đi theo (−7.8K dòng).
- **Celery khỏi path V1** — worker/beat/flower/redis/minio khỏi
  compose (còn app/migrate/postgres/chromadb); eager gọi trực tiếp,
  `.delay` còn lại bọc try/except best-effort.

## [1.1.0] — 2026-09-11 — benchmark era + full-repo remediation

### Added — benchmark era (PR #11–#20, 2026-09-05 → 2026-09-09)
- **OpenClaw auto-capture** (#11) — agent-token imports + capture daemon
  (`scripts/openclaw_capture.py`) + skill docs.
- **Compression + progressive disclosure** (#12) + one-command installer
  (`install.sh`, the `npx claude-mem install` equivalent).
- **Benchmark tuning ladder, all with Wilson 95% CIs** (#13–#19):
  session-level chunking (NEGATIVE, recorded honestly) → system-vs-baseline
  n=20 → semantic-dominant ranking (0.700 vs 0.600) → chunk overlap fusion
  (0.650) → official n=100 run (0.490 [0.394, 0.587]) → Jina semantic
  rerank (0.570 vs 0.490, CIs separate).
- **Map-reduce answering experiment** (#20) — NEGATIVE result (0.486 clean
  vs 0.570 single-pass), recorded honestly; resume/complete scripts kept
  for long runs.
- **Judge hardening** — exact-token match, env-aware gateway/model, real
  judged pilot fixture.

### Fixed — full-repo review + remediation batch (2026-09-10)
- **Full-stack Postgres was broken on first write**: models used JSON list
  columns while migrations created `varchar[]`; referral tables had no
  migration at all. New migration `a9b8c7d6e5f4` converts ARRAY→JSONB and
  creates the referral tables; `tests/migrations/` guards the drift class.
- **Prod compose exposed internal services**: `ports: []`/`volumes: []`
  are Compose merge no-ops — switched to `!override []`; `security_check.py`
  now validates merged `compose config` behavior instead of grepping YAML.
- **Cross-tenant insight mutation** via LLM-echoed IDs — ownership
  prefilter added. **CRAG web fallback** re-applies the context budget.
  **Celery graph tasks** no longer poison the shared LLM client across
  `asyncio.run` boundaries. **Embedding backend switches** fail loud via
  collection dim guard.
- **Test suite honesty**: `make test` was a silent no-op (648 skips, exit 0)
  without Postgres — now 634 pass / 46 honest per-test skips; 7 DB-free
  suites added to CI; a dozen never-green tests fixed (wrong auth-dep
  overrides, missing API keys, structlog-kwarg logging crash).
- **Frontend**: two infinite refetch/render loops fixed, dead Discovery
  journey + documents endpoints removed, FeatureHints token-key fix.

## [Unreleased] — Open Memory Hub MVP (2026-09-02 → 2026-09-04)

The strategic pivot from "AI second brain app" to an **open memory hub for
AI agents** (decision one-pager: [open-memory-hub.md](https://github.com/twilightt1/orivory-private/blob/main/docs/ideas/open-memory-hub.md) (private); research
corpus: [research/](https://github.com/twilightt1/orivory-private/tree/main/docs/research) (private)).

### Added
- **MCP server** at `/mcp` — six scoped memory tools (`search_memory`,
  `get_memory`, `list_recent`, `add_memory`, `delete_memory`,
  `forget_memory`) over the official `mcp` SDK (stateless HTTP, pinned
  `>=1.9.0,<2`), with per-agent identity, ASGI scope normalization
  (Starlette 1.6 307 header-strip workaround), and settings-driven
  transport security (`MCP_HUB_ALLOWED_HOSTS`).
- **Agent clients + access ledger** — `agent_clients` registry
  (`memory:read`/`memory:write` scopes, sha256-only token storage,
  plaintext shown once, instant revocation) and the append-only
  `memory_access_logs` ledger ("which AI saw what, when").
- **Erasure receipts** — `erase_memories` service with ownership-checked
  transitive cascade (BFS over `parent_id`), Chroma vector cleanup +
  adversarial verification pass, three honest receipt statuses
  (`completed` / `completed_with_residual` / `completed_with_errors`),
  per-target isolation with session rollback; `forget_memory` MCP tool and
  `POST/GET /api/v1/erasure-receipts`.
- **Import paths** — `POST /api/v1/imports`: ChatGPT / Claude / generic /
  PAM `memory-store.json` exports with auto-detection, per-item isolation
  (malformed entries skip, never fatal), 10k-char cap, dedup by
  (user, source_type, source_ref), best-effort embedding with counted
  failures.
- **Benchmark harness** — `eval/benchmarks/`: LongMemEval-S (primary) +
  MemoryAgentBench (secondary, selective-forgetting) adapters, phased
  runner CLI with a structural no-fabrication guarantee, dataset sha256
  integrity, leaderboard-hygiene fields reserved.
- **ClawHub skill package** — `skills/orivory/` (agent runbook, tool
  catalog, error runbook, worked examples).
- **Compose hardening** — one-shot `migrate` service gating `app` /
  `celery_worker` (`alembic upgrade head` before any server starts) and an
  `mcp_hub` readiness check.
- **CI** — DB-free hub/benchmark test suites wired into the workflow.

### Changed
- `memories.source_type` gained import/MCP values (`chatgpt_import`,
  `claude_import`, `generic_import`, `mcp_agent`) — no migration needed
  (unconstrained `String(32)`); Literals extended in schemas and routes.
- `POST /api/v1/memories` now validates `parent_id` ownership (404 on
  foreign/missing parent).
- Docs rewritten around the hub positioning; pre-pivot architecture/spec
  docs removed (history in git).

### Removed
- `OpenMemory`-era docs superseded by `docs/ARCHITECTURE.md`;
  sprint-progress trackers; stale demo script; root `main.py` stub.

## [Unreleased] — Phase 1-3 remediation (2026-06-01 → 2026-06-02)

### Security

- **Refresh tokens are now hashed in Redis** (see `auth_service`).
  - Tokens are stored under `refresh:{sha256(token)}` instead of
    `refresh:{raw}`, so a Redis snapshot no longer yields a list of
    usable tokens.
  - A per-user index set `refresh_user:{user_id}` is maintained so
    `_invalidate_all_refresh()` runs in O(N_user_tokens) instead of
    scanning the full `refresh:*` keyspace.
  - New `auth_service._invalidate_one_refresh()` revokes a single
    token (used by `/logout` and refresh-rotation).
  - The `/api/v1/auth/refresh` and `/api/v1/auth/logout` endpoints
    hash incoming tokens before any Redis interaction.
- **Email mock no longer logs token-bearing bodies**. The mock
  implementation that runs when `SENDGRID_API_KEY` is empty previously
  printed the full HTML (containing OTP and password-reset tokens)
  to stdout. The new behaviour logs only recipient, subject, and
  body length. Set `EMAIL_MOCK_VERBOSE=True` to opt back in to the
  full body at `DEBUG` level (dev environments only).
- **Graph snapshot empty-state hardening**: `graph_snapshot` returns
  an empty snapshot when the user has no entities, avoiding an
  empty `IN (...)` SQL clause that depended on dialect tolerance.

### Added

- `Settings.EMAIL_MOCK_VERBOSE: bool = False` — opt-in verbose
  logging for the email mock (gates full HTML body output).
- `auth_service._hash_refresh_token()` — internal helper that returns
  the SHA-256 hex of a refresh token.
- `auth_service._invalidate_one_refresh()` — single-token revocation
  helper used by `/logout` and refresh-rotation.

### Changed

- **Source sync endpoint** (`/api/v1/sources/{id}/sync`) now calls
  `SourceSyncService.sync()` instead of returning a stub.
  - Errors surface as `Source.status = "error"` with
    `Source.sync_error` populated, never as an unhandled 500.
  - Module docstring updated to reflect the real behaviour.
- **Module-level `rag_graph`** in `app.api.v1.chat` so test code can
  monkeypatch the compiled graph. The SSE stream resolves the graph
  through a lazy accessor.
- **Embedder module**: `embedder.async_client` and
  `embedder.sync_client` are exposed via module `__getattr__` for
  lazy resolution and test injection. Production callers see a
  normal module attribute.
- **Graph re-exports**: `MAX_RETRIES`, `route_from_router`,
  `route_after_grade_docs`, and `route_after_grade_gen` are now
  re-exported from `app.agents.graph` with an explicit `__all__`.

### Fixed

- **Prompt construction test** aligned with the actual renderer
  output: `[Source N] (source_type - filename)`.
- **Unauthenticated chat test** now accepts 401 or 403, since 401 is
  the correct RFC response for missing credentials.
- **Test fixtures** in `tests/rag/test_graph_routing.py` now provide
  non-empty `grounding_context_chunks` so retry branches in the
  routing helpers are actually exercised.
- **Linter**: dropped unused imports across `app/`, `tests/`,
  `eval/`, and `scripts/`; replaced lambdas assigned to names
  (`E731`), unwrapped one-line `try/except` chains (`E701`), and
  annotated intentional late imports in smoke scripts with
  `# noqa: E402`. Final ruff run reports zero issues.

## Historical Releases

Phase 16 (prior) — Security readiness audit added
`scripts/security_check.py` and the production guardrails enforced by
`Settings._validate_production_settings()`. The audit covers
JWT-secret strength, CORS, default MinIO credentials, provider key
requirements, production-internal port stripping, Flower under the
`ops` profile, admin-only diagnostics authorization, secret-safe
diagnostics output, FastAPI docs disabled in production, and
explicit demo placeholders in `.env.example`.
