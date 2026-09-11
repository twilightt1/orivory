# Draft: Show HN / showcase post — Orivory v1.1.0 (CHƯA POST — founder duyệt)

> Gợi ý tiêu đề: "Orivory — self-hosted memory hub for AI agents, with receipts (0.570 LongMemEval-S, negatives published)"

---

My agents kept forgetting. Worse — I couldn't see what they remembered.

Orivory (MIT, self-hosted) is a memory hub every MCP-capable agent shares:
Claude Desktop, Cursor, OpenClaw, your own scripts. One place stores, every
agent reads/writes through scoped tokens (`memory:read` / `memory:write`).

Three things I haven't seen combined elsewhere:

1. **Ledger** — every agent access recorded: which AI saw what, when.
   `forget_memory` returns a **verifiable erasure receipt** (cascade across
   rows, links, vectors — re-checked, not just claimed).
2. **Benchmarks with the negatives left in** — LongMemEval-S n=100:
   **0.570** (single-pass + rerank) vs 0.490 baseline, Wilson CIs in-repo.
   Two experiments that LOST (session chunking, map-reduce 0.486) are
   published in the same folder. (`eval/benchmarks/results/`)
3. **Leave anytime** — plain Postgres + JSON, one-command Docker install,
   import your ChatGPT/Claude exports with dedup.

Demo that convinced me it works: two agents, one memory — agent A writes a
decision, agent B recalls it next session, and the ledger shows both
accesses. Screenshot that page and you get the whole product in 10 seconds.

v1.1.0 is out today: benchmark era results + a full-repo remediation batch
(Postgres schema reconciliation, prod compose lockdown, cross-tenant fix,
honest test suite: 634 pass / 0 fail without infra).

Try: `curl -fsSL https://raw.githubusercontent.com/twilightt1/orivory/main/install.sh | bash`
Skill: `skills/orivory/` (ClawHub publish pending)
Good first issues: #21–#26 — contributors welcome, I respond within 48h.

Honest limits: tuning frozen at 0.570 (diminishing returns solo);
multi-session recall (8/26) is the known weak slice — next lever is
multi-hop fusion. Numbers, receipts, and gaps all in the repo.
