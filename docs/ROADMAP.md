# Orivory Roadmap

> Status: 2026-09-12. History lives in git; this file tracks what shipped and
> what's open. Design rationale for the hub direction and positioning live in
> the private companion repo
> ([orivory-private](https://github.com/twilightt1/orivory-private):
> [ideas/](https://github.com/twilightt1/orivory-private/tree/main/docs/ideas) ·
> [research/](https://github.com/twilightt1/orivory-private/tree/main/docs/research)).

## Shipped

### Correctable Memory V1 (2026-09-12, unreleased)
- Evidence-first memory: `cm_*` metadata, atomic `resolve_correction`,
  `correct_memory` MCP tool (7th), provenance ở `get`, `state` ở `search`.
- Latency: fast-path skip-rewrite, bounded timeline, stage trace. Không đổi stack.
- Tối giản: −frontend, −16 LangGraph agents (−7.8K dòng), −Celery khỏi path V1.
- Spec + plan: `docs/superpowers/specs/2026-09-12-correctable-memory-v1.md`.

### Benchmark era (2026-09-05 → 2026-09-09, PR #11–#20)
- OpenClaw auto-capture, one-command installer, compression.
- Tuning ladder with Wilson CIs: 0.490 official n=100 → 0.570 Jina rerank;
  two NEGATIVE results recorded honestly (session chunking, map-reduce).
- Judge hardening (exact-token match, judged pilot).

### Review + remediation batch (2026-09-10)
- Schema reconciliation migration (`a9b8c7d6e5f4`) — full-stack Postgres
  deployable again; `tests/migrations/` guards the drift class.
- Prod compose lockdown (`!override`), behavior-validating security gate.
- Cross-tenant, CRAG-budget, Celery-loop, embedding-dim fixes.
- Test honesty: 634 pass / 0 fail without infra; CI extended.
- Frontend loop/dead-feature fixes; FeatureHints token fix.

### Foundation (P0–P4, 2026-06)
- Connector-synced memories embed; reindex/backfill task + admin endpoint.
- Documents → memories unification (hybrid granularity); `save_note` intent.
- Salience loop (bump-on-use + decay), proactive digest endpoint, graph perf fix.
- Provable quality: grounding confidence in SSE + admin quality-trend endpoint.
- Hardening: answer temperature 0.0, char-budget contexts, index migrations,
  email normalization, query length caps, dead code removal.

### Open Memory Hub MVP (2026-09, PR #7)
- **MCP server** at `/mcp` — six scoped memory tools, per-agent identity
  (sha256 token registry), streamable-HTTP with scope normalization.
- **Permissions + access ledger** — per-agent read/write scopes, append-only
  audit log of every authorized call.
- **Erasure receipts** — ownership-checked transitive cascade, Chroma
  verification pass, three honest statuses, MCP `forget_memory` + REST.
- **Import paths** — ChatGPT / Claude / generic / PAM upload with detection,
  dedup, 10k cap, per-item isolation.
- **Benchmark scaffold** — LongMemEval-S + MemoryAgentBench adapters, phased
  runner, no-fabrication guarantee, hygiene fields reserved.
- **ClawHub skill package** — `skills/orivory/` (runbook + examples + tool
  catalog), publishable via `clawhub skill publish` (founder-side step).
- **Hardened compose** — one-shot `migrate` service gating app/celery;
  `mcp_hub` readiness check; healthchecks on all infra.

## Open follow-ups (ranked)

1. **UI**: access-ledger page, upload/import page, erasure-receipts view.
   (Security dashboard + imports page exist; ledger/receipts views TBD.)
2. **Behavioral REST security tests** — partially closed 2026-09-10
   (cross-tenant refresh guard + endpoint tests, auth-dep override fixes);
   full HTTP-level matrix still open.
3. **LLM judge for benchmarks** (official LongMemEval prompt, version pinned)
   → first real public score after ≥3 judged runs. Judge robustness fixed
   (exact-token match, INFO-level logging crash, provenance tags).
4. **Live wiring for benchmark ingest/query** against a running stack.
5. **Ledger retention policy** — `ledger_tasks.py` prune exists; unique index
   on `(user_id, source_type, source_ref)` shipped in `c7d8e9f0a1b2`.
   Remaining: retention config surface + referral-code unique index (shipped
   in `a9b8c7d6e5f4` as partial `uq_referral_codes_user_active`).
6. **OpenClaw session-log import** (their memory is local Markdown — same
   generic-JSON path, needs a converter).
7. **Rewind/Limitless adapter** — blocked on a verified export format
   (SQLCipher-encrypted, no official format).
8. **Gemini/Copilot import adapters** (PAM documents the shapes).
9. **Re-revoke `revoked_at` guard; explicit `captured_at` in imports.**
10. **Parent ownership + depth-cap receipts polish** — partially done; see
    code TODOs.

### From claude-mem competitive analysis (see [CLAUDE_MEM_ANALYSIS.md](https://github.com/twilightt1/orivory-private/blob/main/docs/research/CLAUDE_MEM_ANALYSIS.md) (private))
- ✅ **Auto-capture for OpenClaw** (PR #11): agent-token imports + the
  stdlib-only `scripts/openclaw_capture.py` watcher — sessions auto-flow
  into the ingestion path without manual calls.
- ✅ **Compression-before-storage** (this PR): feature-flagged
  (`COMPRESSION_ENABLED`, default off) best-effort AI summarize before
  memory writes on both the REST and MCP seams; failures degrade to raw.
- ✅ **Progressive-disclosure search** (this PR): `search` returns an
  index (id/title/snippet, no full content); new `timeline` tool returns
  anchor + before/after windows; workflow baked into tool docstrings.
- ✅ **One-command installer** (this PR): `curl … install.sh | bash`
  bootstraps the lite container idempotently with health gating.
- ⏳ **LLM-judge real runs**: requires `OPENAI_API_KEY` — the runner
  honest-skips rather than fabricate scores (hygiene rule 1).
- **Watch**: claude-mem's server-runtime GA plan (#2685) — Docker+pg+redis+
  scopes is Orivory's full-stack lane; land ledger/receipts/benchmark
  differentiators before their GA drops.

## Explicitly out of scope (for now)

Meeting-notes AI (30+ player market), team-first positioning, Celery for
large imports (sync + 20 MiB cap covers v0), donation-based funding.
