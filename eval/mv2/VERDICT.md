# m-v2 verdict — keep XS (signed thresholds)

Signed gate (2026-09-19, §12): an m-v2 arm is eligible iff **non-inferior on
every slice** (recall@5 drop ≤ 0.02 vs XS) **and superior on the VI slices**
(gain ≥ +0.02). Verdict source: `eval/ablation_retrieval_mv2.json` (real run,
5 arms, reproducible byte-identically apart from the timestamp).

## Verdict: FAIL on all four m-v2 arms → the local default stays `arctic` (XS)

| slice (recall@5) | xs | fp_reference | onnx_fp | onnx_int8 | int8_mrl256 |
|---|---|---|---|---|---|
| exact_id | 0.375 | 0.9375 | 0.9375 | 1.0 | 0.75 |
| vi_diacritics | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| vi_no_diacritics | 1.0 | 0.625 | 0.625 | 0.625 | 0.625 |
| en / short / long | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

- Non-inferiority violated: `vi_no_diacritics` −0.375 on every arm.
- VI superiority unmet: `vi_diacritics` is at ceiling (1.0 for everyone,
  gain +0.0); `vi_no_diacritics` is a loss.
- The unaccented-VI loss is a measured capability difference, not a plumbing
  artifact: the XS tokenizer strips tone marks (accented ↔ unaccented queries
  embed identically, cosine 1.0), so that slice is the same query to XS; the
  m-v2 tokenizer preserves diacritics (query-form cosine 0.30–0.51) and ranks
  the golds lower on the unaccented form.

The ONNX FP32 export reproduces the official reference code exactly
(corpus + query cosine 1.0), and the INT8/MRL arms were measured with the
batch-composition policy the parity harness pinned — the numbers above are the
honest ones.

## What this closes

- Task 3 (opt-in `LOCAL_EMBED_MODEL=arctic-m-v2` registry work): **skipped by
  the signed rule** — no code, no default change. XS remains the shipped local
  model; the m-v2 harness (`eval/mv2/`) stays as experiment tooling for any
  future re-run.
- Follow-up option (not committed to): a VI-slice improvement that teaches the
  XS path nothing new would need an m-v2 *query-side* normalization strategy
  measured against this exact artifact — a new experiment, not a tweak to this
  verdict.
