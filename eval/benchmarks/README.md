# Benchmark Harness — Orivory Memory Hub

This directory holds Orivory's **external benchmark harness** for long-term agent memory. It extends the existing eval tooling (`eval/metrics.py`, `eval/reporting.py`) — it does not touch application code and registers nothing in the app.

> **No scores ship in this repo until a real run happens.**
> A real run requires: the downloaded dataset (below), a live Orivory stack, and LLM keys for judged scoring. Any JSON or table claiming a benchmark number that was not produced by `eval/run_benchmark.py` against a real dataset does not belong here. The runner refuses to fabricate results: no dataset → clear error; no live stack → dry-plan exit 0.

## Which benchmarks, and why

| Benchmark | Role | Why this one |
|---|---|---|
| **LongMemEval-S** ([ICLR 2025](https://arxiv.org/abs/2410.10813), [repo](https://github.com/xiaowu0162/LongMemEval)) | **Primary** | Peer-reviewed and human-curated: 500 questions over timestamped multi-session chat histories, testing exactly what a time-aware memory hub claims — **knowledge updates, temporal reasoning, and abstention** (plus information extraction and multi-session reasoning). _S (~115k tokens / ~40 sessions per history) is long enough to stress retrieval rather than the context window, and is the variant the field actually runs (_M is ~1.5M tokens and rarely reported). Publishing here buys comparability with Zep's and Mem0's published runs. |
| **MemoryAgentBench** ([arXiv 2507.05257](https://arxiv.org/abs/2507.05257)) | **Secondary** | The **only benchmark that scores selective forgetting** (Orivory's differentiator), via its FactConsolidation set, alongside test-time learning, accurate retrieval, and long-range understanding, all in incremental multi-turn form. Neutral academic group; no commercial system has gamed it. v0 harness covers the selective-forgetting scenario; a full dataset rebuild is a follow-up. |
| **LoCoMo** (ACL 2024) | **Protocol-explicit only — never lead** | Usable as a secondary stress test, but not decision-grade: tiny n (10 conversations), fits modern context windows, broken adversarial category, and protocol sensitivity so high that Zep and Mem0 disagree on the same system's score by ±17 points (see [PAPERS_AGENT_MEMORY.md §3.1](https://github.com/twilightt1/orivory-private/blob/main/docs/research/PAPERS_AGENT_MEMORY.md) (private)). If community pressure demands a LoCoMo number, it ships only protocol-explicit (fixed judge prompt, ≥10 runs with variance, exclusions stated) and never in the headline. |

## Dataset download

Datasets are downloaded by the operator and **never committed** (`eval/benchmarks/data/` is gitignored).

```bash
mkdir -p eval/benchmarks/data
cd eval/benchmarks/data
# LongMemEval-S (~264MB uncompressed)
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
# Record the digest before every run — it goes into the results file (hygiene rules below):
shasum -a 256 longmemeval_s_cleaned.json   # macOS; sha256sum on Linux
```

The file is a JSON array of 500 instances, each with `question_id`, `question_type`, `question`, `answer`, `question_date`, `haystack_session_ids`, `haystack_dates`, `haystack_sessions` (parallel arrays; each session is a list of `{"role": "user"|"assistant", "content": str}` turns; evidence turns carry `has_answer: true`), and `answer_session_ids`. A tiny, schema-exact sample lives at `eval/benchmarks/fixtures/longmemeval_s_fixture.json` (2 instances: one temporal-reasoning, one knowledge-update abstention) for CI-safe tests — never the real dataset.

The MemoryAgentBench stub adapter consumes `eval/benchmarks/fixtures/memoryagentbench_fixture.json` (one record per competency; the `selective_forgetting` record carries the to-be-forgotten fact with `"forget": true`). Real MemoryAgentBench data is a follow-up (its released sets are reconstructed long-context corpora; see the plan's follow-ups).

## How to run

**LLM judge.** Answers not covered by the exact-match guard are scored by
the official LongMemEval judge semantics — implemented in
`eval/benchmarks/llm_judge.py` with a pinned prompt version
(`JUDGE_PROMPT_VERSION = longmemeval-official-v1`). Every results file
records the judge version; changing the prompt requires bumping it and
re-running.

The v0 boundary is: **download dataset → plan → score**. Live ingest/query/judge wiring is a pinned follow-up — in v0, `--phase ingest` and `--phase query` exit 3 with a refusal message and write nothing; `--phase score` aggregates an existing per-question results JSON (the one phase that needs no live stack):

```bash
.venv/bin/python eval/run_benchmark.py --benchmark longmemeval_s \
  --dataset eval/benchmarks/data/longmemeval_s_cleaned.json \
  --output-dir eval/benchmarks/results/longmemeval_s \
  --limit 20 --phase plan        # then: --phase score --results PATH
```

Without a dataset the CLI exits 2 with a pointer back to this README; without a live stack it prints the plan (instance count, phases) and exits 0 — it never fabricates results. Adapters are shipped: `eval/benchmarks/longmemeval_s.py` (loader + ingest/query interfaces + exact-match judge guard) and `eval/benchmarks/memoryagentbench.py` (selective-forgetting scenario driver). LLM-judge integration for answers that fail the exact-match guard is a pinned follow-up.

**MemoryAgentBench grouping contract (pinned for the adapter):** the v0 fixture stores **one self-contained record per competency** — each record yields exactly one scenario (its own ingest turns + its own question). Grouping is per-record/competency, NOT per-phase; do not merge consecutive query records across competencies. Because MemoryAgentBench sessions are unnamed, `answer_session_ids` use the synthetic ids `s0`, `s1`, … parallel to the record's sessions.

## Leaderboard hygiene

These rules exist because the LoCoMo controversy showed vendor memory scores swinging ±10–20 points on the same benchmark purely on protocol choices (see [PAPERS_AGENT_MEMORY.md §3.1](https://github.com/twilightt1/orivory-private/blob/main/docs/research/PAPERS_AGENT_MEMORY.md) (private)). Every result file produced by this harness must include:

1. **Judge prompt version committed** — the exact judge prompt text (or its committed hash + version tag) is pinned in the repo before any judged number is recorded. Judge changes invalidate prior comparability and must be stated.
2. **≥3 runs, mean ± variance** — any judged score is reported as the mean over at least 3 runs with variance (±). Single-run numbers never appear in results files.
3. **Dataset SHA-256 in results** — every result file records the SHA-256 of the exact dataset file it ran against, so a dataset revision can never silently change a score.
4. **Full-context baseline noted** — results note whether the full-context (no-memory-system) baseline was run alongside; where it was not, that absence is stated.
5. **Deviations stated** — any deviation from the benchmark's official protocol (subset of questions, modified prompts, excluded categories, different judge) is written into the results file explicitly. No silent deviations.
