# Evaluation Guide

How Orivory measures RAG quality, runs the eval pipeline, and interprets
results.

## TL;DR

```bash
# 1. Offline eval (no LLM, deterministic) — the only lane
.venv/bin/python eval/run_eval.py --mode offline --output-dir eval/results --top-k 5

# 2. Offline eval + RAGAS-style metrics (a no-op in this repo — see below)
.venv/bin/python eval/run_eval.py --mode offline --enable-ragas

# 3. Memory benchmarks (LongMemEval-S / MemoryAgentBench)
.venv/bin/python eval/run_benchmark.py \
    --benchmark longmemeval_s \
    --dataset eval/benchmarks/data/longmemeval_s_cleaned.json \
    --phase plan --limit 20
```

## What gets measured

### Core metrics (always computed)

| Metric | What it measures | Target |
|--------|------------------|--------|
| `source_hit_rate` | % of cases where expected source appears in top-K | ≥ 90% |
| `keyword_coverage` | % of expected keywords found in answer | ≥ 80% |
| `citation_rate` | % of answers containing `[Source N]` markers | ≥ 75% |
| `fallback_accuracy` | % of out-of-scope queries routed correctly | 100% |

### Self-correction metrics

| Metric | What it measures |
|--------|------------------|
| `hallucination_flag_rate` | How often the LLM-as-judge flagged an answer |
| `correction_rate` | Of the flagged ones, how many retries fixed it |

### RAGAS-style metrics (`--enable-ragas`)

> **Not implemented in this repo.** `--enable-ragas` imports
> `eval/ragas_metrics.py`, which does not exist here, so the flag is a silent
> no-op: the report carries no `context_recall@k`, `context_precision@k`,
> `mrr`, `ndcg@k`, `faithfulness` or `answer_relevancy` number. Do not cite
> them as measured. What the offline run DOES compute is the core table above
> (source-hit, keyword coverage, citation rate, fallback accuracy); LLM-judged
> fields come only from a judge run (`eval/llm_judge.py`).

### Retrieval ablation artifacts (the measured evidence)

The retrieval evidence the P2 gate reads is `eval/ablation_retrieval_p2.py`
plus its committed artifact `eval/ablation_retrieval_p2.json`: four arms
(dense-only / lexical-only / RRF hybrid / hybrid+rerank) over a frozen
311-row, 48-query fixture, per-slice `recall@{1,5,10}` and MRR, the enable-rule
verdict for `RETRIEVAL_HYBRID_ENABLED` and the late-vs-full hydration verdict.
It is **evidence at fixture scale** — read the artifact's `limitations` before
quoting any number (the whole `+0.1042` overall recall@5 gain sits in the
`exact_id` slice, the dense leg is exact cosine rather than Qdrant ANN, and the
rerank stage is a local stand-in). Shape is pinned by
`tests/retrieval/test_p2_ablation_contract.py`; the embedding-pooling ablation
from P1b is `eval/ablation_mean_vs_cls.py` + `eval/ablation_mean_vs_cls.json`.

## The eval dataset

`eval/orivory_eval_dataset.json` — 18 cases across 7 categories:

| Category | Cases | What it tests |
|----------|-------|---------------|
| api_auth | 3 | API authentication questions |
| billing | 3 | Billing and plan questions |
| webhooks | 3 | Webhook setup / troubleshooting |
| integrations | 2 | Third-party integrations |
| releases | 2 | Product release notes |
| incidents | 2 | Incident response runbook |
| out_of_scope | 3 | "I don't know" path |

Each case has:
- `id` — stable identifier
- `category`
- `query` — the user question
- `expected_sources` — document names that should appear in retrieval
- `expected_keywords` — terms that should appear in the answer
- `should_fallback` — whether the case expects the "not in my memories" path

## Reading the report

`eval/results/latest_report.md` contains:

1. **Summary table** — overall metrics
2. **Per-case results** — each case's status + sources
3. **Retrieval ablation evidence** — the committed artifacts, not the
   `--enable-ragas` flag (see "Retrieval ablation artifacts" above; that flag is
   a no-op in this repo)
4. **Failed cases** — anything that crossed a threshold
5. **Recommendations** — auto-generated next steps

## Adding new eval cases

```json
{
  "id": "my_new_case_001",
  "category": "api_auth",
  "query": "How do I rotate my API key without downtime?",
  "expected_sources": ["api_authentication_guide.md"],
  "expected_keywords": ["rotation", "overlap", "revoke"],
  "should_fallback": false
}
```

Add to `eval/orivory_eval_dataset.json`, then re-run.

## Custom thresholds

```bash
.venv/bin/python eval/run_eval.py --mode offline \
    --fail-under-source-hit 0.9 \
    --fail-under-keyword-coverage 0.8
```

The eval will exit non-zero if any threshold is missed — useful in CI.

## CI integration

```yaml
- name: Run RAG eval
  run: |
    .venv/bin/python eval/run_eval.py --mode offline \
      --output-dir eval/results --fail-under-source-hit 0.9
```

The exit code propagates; pull request is blocked on regression.

## Prompt A/B testing

Not implemented in this release: there is no versioned-prompt registry
(`app/agents/prompts/versions.py`) and no `scripts/eval_experiments.py`
sweeper in this repo. Prompts are module-level constants inside the module that
uses them — there is no registry and no A/B script
(see [RAG_TECHNIQUES.md](RAG_TECHNIQUES.md) §10). A prompt variant is a code
change, and evaluating it is an eval run (above).

## Benchmarks (memory hub)

The memory-hub benchmark harness lives in [`eval/benchmarks/`](../eval/benchmarks/README.md)
(see that README for the full protocol and leaderboard-hygiene rules):

- **LongMemEval-S** (primary) — 500 human-curated questions over ~115k-token
  chat histories; tests knowledge updates, temporal reasoning, abstention.
- **MemoryAgentBench** (secondary) — the only benchmark scoring selective
  forgetting; Orivory's `forget_memory` path is exercised two-sided
  (surviving answer recalled AND stale fact gone).

```bash
# Phase 1 — plan (no dataset needed with the committed fixtures)
.venv/bin/python eval/run_benchmark.py --benchmark longmemeval_s \
    --dataset eval/benchmarks/fixtures/longmemeval_s_fixture.json --phase plan

# Phase 2 — download the real dataset (~264 MB, HuggingFace
# xiaowu0162/longmemeval-cleaned) into eval/benchmarks/data/, then:
.venv/bin/python eval/run_benchmark.py --benchmark longmemeval_s \
    --dataset eval/benchmarks/data/longmemeval_s_cleaned.json --limit 20

# Phase 3 — score an existing results JSON
.venv/bin/python eval/run_benchmark.py --benchmark longmemeval_s \
    --results eval/benchmarks/results/<run>/results.json
```

No scores ship in the repo — results are written only from real runs, with
the dataset sha256 recorded. The LLM judge for non-exact-match answers is a
pinned follow-up; until then, aggregate numbers read artificially low by
design.

## Cost & latency analysis

```bash
# After running the eval
.venv/bin/python -c "from app.observability.cost import CostTracker; \
    t = CostTracker(); print(t.breakdown_by_agent())"
```

The ledger is the SQLite file `eval/costs.db` (run the command from the repo
root): the app's `llm_client` records every call there, so the same query works
after a session against the running API.

## Benchmarking different models

Model/cost comparisons come out of the RAG eval (every case records tokens
and latency per agent):

```bash
.venv/bin/python -c "from app.observability.cost import CostTracker; \
    t = CostTracker(); print(t.breakdown_by_agent())"
```
