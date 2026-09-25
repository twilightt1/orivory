# RAG Evaluation Framework

## Start here (canonical entry points)

| Muốn gì | Chạy gì |
|---|---|
| Offline eval, deterministic, CI-safe | `python eval/run_eval.py --mode offline` (the only lane) |
| Benchmark LongMemEval-S / MemoryAgentBench | `python eval/run_benchmark.py --benchmark longmemeval_s --dataset <json> --output-dir <dir> --phase plan` → `ingest` → `query` → `score` |
| Resume interrupted system run (same local retrieval contract only) | `python eval/resume_system_run.py --results <results.json>` |

Legacy one-shots (paths cứng, không CLI args — chạy đúng như ghi, đừng
copy pattern): `run_judge_only.py` (extreme dataset), `run_real_sample.py`,
`run_system_benchmark.py`, `mindlayer_offline_eval.py`,
`pilot_judged_fixture.py`, `run_full_eval.py`. Đừng thêm script mới —
mở rộng 4 entrypoint trên.

This directory contains evaluation tooling for the Orivory RAG demo.

- **Offline mode** is deterministic and CI-safe. It uses `sample_docs/` directly
  and does not require the API server, a database, the vector store, or LLM
  keys.
- **Live API mode** is gone — it drove the chat surface (`/api/v1/chat/*`),
  which was removed with the full-stack product. `eval/live_api_eval.py` and
  its tests were deleted with it; the offline lane is the whole harness.

## Components

| File | Purpose |
|---|---|
| `orivory_eval_dataset.json` | Golden dataset of questions, expected sources, keywords, and fallback expectations. |
| `metrics.py` | Deterministic metric helpers for source hit, keyword coverage, citations, fallback accuracy, and summaries. |
| `reporting.py` | Markdown and JSON report generation helpers. |
| `run_eval.py` | CLI entrypoint for the offline evaluation lane. |
| `ablation_mean_vs_cls.py` | NOT an eval entrypoint: the P1b mean-vs-CLS ablation (both pooling contracts on one corpus, one embedded Qdrant) whose committed `ablation_mean_vs_cls.json` is the cutover's evidence artifact. Evidence only — never a gate; SKIPs without the arctic ONNX cache. |

## Benchmarks

External long-term-memory benchmarks (LongMemEval-S, MemoryAgentBench) live in
`eval/benchmarks/`, driven by the phased runner CLI `eval/run_benchmark.py`
(plan → ingest → query → score; it never fabricates scores). Dataset download,
hygiene rules, and per-phase usage: `eval/benchmarks/README.md`.

## Dataset Schema

Each item in `orivory_eval_dataset.json` uses this shape:

```json
{
  "id": "api_auth_001",
  "query": "How do I rotate an API key?",
  "category": "api_auth",
  "expected_sources": ["api_authentication_guide.md"],
  "expected_keywords": ["rotate", "API key", "Settings", "Developer"],
  "should_fallback": false
}
```

Out-of-scope examples set `expected_sources` and `expected_keywords` to empty
lists and use `should_fallback: true`.

## Metrics

| Metric | Meaning |
|---|---|
| `source_hit_rate` | Whether expected source files appeared in returned sources. |
| `keyword_coverage` | Whether answer/source text contains expected support keywords. |
| `citation_rate` | Whether answers include a citation marker or returned source metadata. |
| `fallback_accuracy` | Whether out-of-scope cases fallback and in-scope cases do not. |
| `avg_latency_ms` | Average runtime per case. |
| `hallucination_flag_rate` | Whether trace data indicates hallucination handling. |
| `correction_rate` | Whether trace/done metadata indicates self-correction or retry. |

## Offline Evaluation

From the repository root:

```bash
python eval/run_eval.py --mode offline --output-dir eval/results --top-k 5
```

The offline runner writes:

- `eval/results/latest_report.md`
- `eval/results/latest_report.json`

Optional threshold checks:

```bash
python eval/run_eval.py \
  --mode offline \
  --output-dir eval/results \
  --top-k 5 \
  --fail-under-source-hit 0.80 \
  --fail-under-keyword-coverage 0.70
```

If a threshold is not met, the command exits non-zero.

## Testing Metric Logic

```bash
python -m pytest --confcutdir=tests/eval \
  tests/eval/test_eval_metrics.py \
  -q
```

These tests do not call a running API; they validate scoring and metric
behavior.

## Continuous Evaluation Strategy

### 1. Offline Evaluation in CI

The default CI runs deterministic metric tests and an offline smoke evaluation
(`python eval/run_eval.py --mode offline --output-dir eval/results --top-k 5`).
This gives fast regression coverage without infrastructure or secrets.

### 2. Dataset Expansion

When a real query fails or produces weak citations, add it to the dataset with:

- the expected source document
- key phrases that should appear
- whether it should fallback
- the category impacted
---

## Historical baselines (v1.1.0; preserve artifacts, not current regression gates)

Config `orivory_stack`, seed `20260906`, judge `longmemeval-official-v1`:

| Run | File | Score |
|---|---|---|
| Single-pass + hosted rerank (historical), n=100 | `benchmarks/results/longmemeval_s_system_n100.json` | **0.570**, Wilson 95% CI [0.472, 0.663] |
| No-rerank baseline, n=100 | (PR #18) | 0.490, CI [0.394, 0.587] |
| Map-reduce answering, n=70 clean | `benchmarks/results/longmemeval_s_system_n100_mapreduce.json` | 0.486 — NEGATIVE, single-pass stays default |

The 0.570 result is a historical hosted-model measurement. The shipped default
retrieval lane is local, so that score cannot be reproduced by the current
benchmark config; keep the committed artifact as provenance, not as a current
regression target.

Rule: for retrieval-code changes, re-run the n=100 single-pass lane and compare
only against a baseline with the same local embedding/rerank configuration; the
historical hosted score above is not a regression gate. Tuning the score further
is explicitly out of scope until ≥5 active installs (see
[open-source-positioning.md §5](https://github.com/twilightt1/orivory-private/blob/main/docs/ideas/open-source-positioning.md) (private)).
