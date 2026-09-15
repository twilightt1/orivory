# RAG Evaluation Framework

## Start here (canonical entry points)

| Muốn gì | Chạy gì |
|---|---|
| Offline eval, deterministic, CI-safe | `python eval/run_eval.py --mode offline` |
| Benchmark LongMemEval-S / MemoryAgentBench | `python eval/run_benchmark.py --benchmark longmemeval_s --dataset <json> --output-dir <dir> --phase plan` → `ingest` → `query` → `score` |
| Resume run dài bị ngắt (rate limit) | `python eval/resume_system_run.py` (system) · `eval/complete_mapreduce_run.py` (map-reduce, FROZEN — run completed, do not extend) |

Legacy one-shots (paths cứng, không CLI args — chạy đúng như ghi, đừng
copy pattern): `run_judge_only.py` (extreme dataset), `run_real_sample.py`,
`run_system_benchmark.py`, `mindlayer_offline_eval.py`,
`pilot_judged_fixture.py`, `run_full_eval.py`. Đừng thêm script mới —
mở rộng 4 entrypoint trên.

This directory contains evaluation tooling for the Orivory RAG demo.

- **Offline mode** is deterministic and CI-safe. It uses `sample_docs/` directly
  and does not require the API server, database, Qdrant, Redis, MinIO, or LLM
  keys.
- **Live API mode** is opt-in. It exercises the running backend API, document
  upload/ingestion, SSE chat streaming, returned sources, and agent trace data.

## Components

| File | Purpose |
|---|---|
| `Orivory_eval_dataset.json` | Golden dataset of questions, expected sources, keywords, and fallback expectations. |
| `metrics.py` | Deterministic metric helpers for source hit, keyword coverage, citations, fallback accuracy, and summaries. |
| `reporting.py` | Markdown and JSON report generation helpers. |
| `run_eval.py` | CLI entrypoint for offline and live API evaluation modes. |
| `live_api_eval.py` | Live API evaluator, SSE parser, response collector, and live scoring helpers. |
| `Orivory_offline_eval.py` | Lightweight keyword sanity check used by `run_eval.py --mode offline`. |
| `ablation_mean_vs_cls.py` | NOT an eval entrypoint: the P1b mean-vs-CLS ablation (both pooling contracts on one corpus, one embedded Qdrant) whose committed `ablation_mean_vs_cls.json` is the cutover's evidence artifact. Evidence only — never a gate; SKIPs without the arctic ONNX cache. |

## Benchmarks

External long-term-memory benchmarks (LongMemEval-S, MemoryAgentBench) live in
`eval/benchmarks/`, driven by the phased runner CLI `eval/run_benchmark.py`
(plan → ingest → query → score; it never fabricates scores). Dataset download,
hygiene rules, and per-phase usage: `eval/benchmarks/README.md`.

## Dataset Schema

Each item in `Orivory_eval_dataset.json` uses this shape:

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

## Live API Evaluation

Live mode requires the full application path to be running:

1. Postgres, Redis, Qdrant, and MinIO
2. database migrations
3. FastAPI server
4. Celery ingestion worker
5. a verified/onboarded user or a valid access token
6. provider keys required by ingestion/chat, such as OpenAI, OpenRouter, and Jina

Example using email/password login:

```bash
python eval/run_eval.py --mode live-api \
  --api-base-url http://localhost:8000 \
  --email eval-user@example.com \
  --password EvalPassword123! \
  --sample-docs sample_docs \
  --output-dir eval/results
```

Example using an existing bearer token:

```bash
python eval/run_eval.py --mode live-api \
  --api-base-url http://localhost:8000 \
  --access-token "$ACCESS_TOKEN" \
  --sample-docs sample_docs \
  --output-dir eval/results
```

The live API runner writes:

- `eval/results/live_api_report.md`
- `eval/results/live_api_report.json`

Live mode uploads markdown documents from `sample_docs/`, waits until ingestion
marks them `ready`, sends each dataset query to the SSE chat endpoint, aggregates
streamed tokens, sources, trace, and done metadata, then generates the report.

## Testing Metric and Live-helper Logic

```bash
python -m pytest --confcutdir=tests/eval \
  tests/eval/test_eval_metrics.py \
  tests/eval/test_live_api_eval.py \
  -q
```

These tests do not call a running API; they validate parser, collector, scoring,
and metric behavior.

## Continuous Evaluation Strategy

### 1. Offline Evaluation in CI

The default CI runs deterministic metric tests, live-helper unit tests, and an
offline smoke evaluation. This gives fast regression coverage without
infrastructure or secrets.

### 2. Live/API Evaluation Manually or in a Separate Workflow

Keep live API evaluation separate from default CI because it depends on service
startup, Celery, API keys, and LLM latency. It is suitable for pre-demo checks,
staging smoke tests, or a manually triggered workflow.

### 3. Dataset Expansion

When a real query fails or produces weak citations, add it to the dataset with:

- the expected source document
- key phrases that should appear
- whether it should fallback
- the category impacted
---

## Frozen baseline (v1.1.0, do not tune — regression-gate only)

Config `orivory_stack`, seed `20260906`, judge `longmemeval-official-v1`:

| Run | File | Score |
|---|---|---|
| Single-pass + Jina rerank, n=100 | `benchmarks/results/longmemeval_s_system_n100.json` | **0.570**, Wilson 95% CI [0.472, 0.663] |
| No-rerank baseline, n=100 | (PR #18) | 0.490, CI [0.394, 0.587] |
| Map-reduce answering, n=70 clean | `benchmarks/results/longmemeval_s_system_n100_mapreduce.json` | 0.486 — NEGATIVE, single-pass stays default |

Rule: any retrieval-code change must re-run the n=100 single-pass config
and stay within the frozen CI before merge. Tuning the score further is
explicitly out of scope until ≥5 active installs (see
[open-source-positioning.md §5](https://github.com/twilightt1/orivory-private/blob/main/docs/ideas/open-source-positioning.md) (private)).
