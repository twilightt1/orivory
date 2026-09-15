# Evaluation Guide

How Orivory measures RAG quality, runs the eval pipeline, and interprets
results.

## TL;DR

```bash
# 1. Offline eval (no LLM, deterministic)
.venv/Scripts/python -m eval.run_eval --mode offline --output-dir eval/results

# 2. Offline eval + RAGAS metrics
.venv/Scripts/python -m eval.run_eval --mode offline --enable-ragas

# 3. Live API eval (requires running server + API key)
.venv/Scripts/python -m eval.run_eval --mode live --api-url http://localhost:8000

# 4. Sweep an experiment
.venv/Scripts/python scripts/eval_experiments.py --experiment topk_sweep \
    --variants topk_3,topk_5,topk_8

# 5. Memory benchmarks (LongMemEval-S / MemoryAgentBench)
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

`eval/Orivory_eval_dataset.json` — 18 cases across 6 categories:

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
- `should_cite` — whether the answer must contain citations
- `is_in_scope` — whether the case expects a RAG answer

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
  "should_cite": true,
  "is_in_scope": true
}
```

Add to `eval/Orivory_eval_dataset.json`, then re-run.

## Custom thresholds

```bash
.venv/Scripts/python -m eval.run_eval --mode offline \
    --fail-under-source-hit 0.9 \
    --fail-under-keyword-coverage 0.8
```

The eval will exit non-zero if any threshold is missed — useful in CI.

## CI integration

```yaml
- name: Run RAG eval
  run: |
    .venv/Scripts/python -m eval.run_eval --mode offline \
      --output-dir eval/results --fail-under-source-hit 0.9
```

The exit code propagates; pull request is blocked on regression.

## Prompt A/B testing

```bash
# 1. Register a new variant (edit app/agents/prompts/versions.py)
# 2. Run a sweep
.venv/Scripts/python scripts/eval_experiments.py \
    --experiment router_compare \
    --variants router_v1,router_v2
```

Output: `eval/experiments/router_compare_comparison.md` with
side-by-side metrics.

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
.venv/Scripts/python -c "from app.observability.cost import CostTracker; \
    t = CostTracker(); print(t.breakdown_by_agent())"
```

Or query the admin endpoint:
```
GET /admin/ai-costs?hours=24
```

## Benchmarking different models

Model/cost comparisons come out of the RAG eval (every case records tokens
and latency per agent):

```bash
.venv/bin/python -c "from app.observability.cost import CostTracker; \
    t = CostTracker(); print(t.breakdown_by_agent())"
```
