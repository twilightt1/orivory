#!/usr/bin/env python
"""T2 / D3 — the m-v2 ablation on the p2 ablation's FROZEN fixture, with the signed D1 verdict.

What this artifact decides
--------------------------
The §12 budgets signed on 2026-09-19 pin the only rule under which the shipped
XS embedder may be replaced: an m-v2 arm must be **non-inferior on EVERY slice**
(drop <= 0.02 recall@5 vs ``xs_cls_baseline``) **AND superior on the VI slices**
(gain >= +0.02 recall@5). This script measures the arms and records the
verdict — it never flips a flag: no registry entry, no default change, no code
the app loads (that is T3, conditional on this verdict).

Arms (D3 "full"): ``xs_cls_baseline`` (the shipped arctic-embed-xs, 384-dim
CLS) · ``mv2_fp_reference`` (the OFFICIAL custom modeling code from the HF repo,
pinned revision + review in :mod:`eval.mv2.reference`) · ``mv2_onnx_fp`` (the
pinned ``onnx/model.onnx`` FP32 export) · ``mv2_onnx_int8`` (the pinned
``onnx/model_int8.onnx``) · ``mv2_onnx_int8_mrl256`` (the same INT8 session,
dim 256: slice + re-L2, never the 128-byte/4-bit claim).

Slices: the p2 ablation's six, on the p2 ablation's own fixture —
``build_fixture``/``corpus_hash``/``query_set_hash`` are imported from
``eval/ablation_retrieval_p2.py`` so this artifact's corpus hashes are the
committed p2 hashes (asserted in tests). Metrics: recall@{1,5,10}, MRR per
slice, and per-arm deltas vs ``xs_cls_baseline``.

The substitutions (recorded, because an ablation is only as honest as its seams)
------------------------------------------------------------------------------
- **Ranking**: exact cosine top-k in numpy over the arms' REAL vectors, filtered
  by tenant + served visibility, sorted ``(-score, memory_id)`` — the same seam
  as ``ExactCosineStore`` in the p2 ablation, which production's Qdrant HNSW
  approximates. Unlike p2's pipeline there is no SQL hydration / refill /
  rerank stage: the only thing varying between arms is the embedder, and every
  arm sees identical corpus text, identical queries and identical ranking.
- **Corpus view**: rows carrying ``cm_superseded_by`` are excluded BEFORE the
  top-k cut (the served-visibility filter), applied identically to every arm.
  p2's pipeline filtered them after the pool fetch with a refill; the counts can
  therefore differ from p2's ``dense_only`` arm on the slices where superseded
  near-misses compete (recorded in ``limitations``, not hidden).
- **Batch composition is a per-arm CONTRACT, not a default**: the T1 parity
  report measured that the INT8 export quantizes activations with a per-tensor
  dynamic scale, so a row's vector moves with its batch mates (cos 0.95-0.98 vs
  solo). The INT8 arms therefore embed ONE TEXT PER CALL (bit-exact per text),
  the float32 arms batch (``float32`` reassociation only). The MRL-256 arm runs
  the same INT8 session at dim 256. This decision is recorded per arm in
  ``batch_policy``.
- **FP reference deviations**: the published config ships
  ``unpad_inputs``/``use_memory_efficient_attention`` as the string ``"true"``
  (both xformers-only paths; no macOS build). Forced False — padded SDPA path,
  same math the ONNX exports compute. Full review in ``eval/mv2/reference.py``.

Scale, stated plainly
---------------------
EVIDENCE AT FIXTURE SCALE on a generated corpus: the p2 fixture's hundreds of
rows and 48 queries, local models, exact-cosine ranking, CPU. Never a
production claim; the shape of the decision is what it carries.

Run: ``.venv/bin/python eval/ablation_retrieval_mv2.py``
(writes ``eval/ablation_retrieval_mv2.json``). CI-safe checks live in
``tests/benchmarks/test_ablation_mv2.py`` (gate arithmetic needs no model; the
tiny real run skips when the model cache is absent).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.retrieval import e5_local  # noqa: E402
from eval import ablation_retrieval_p2 as p2  # noqa: E402
from eval.mv2 import runner as mv2  # noqa: E402

ARTIFACT_PATH = ROOT / "eval" / "ablation_retrieval_mv2.json"

# The signed gate (§12, 2026-09-19) — thresholds are constants, never arguments
# of the measurement: they were signed BEFORE the run and the script may not
# choose different ones after seeing numbers.
GATE_MAX_SLICE_DROP = 0.02
GATE_MIN_VI_GAIN = 0.02
VI_PRIMARY = p2.SLICE_VI
VI_SECONDARY = p2.SLICE_VI_NODIAC
_GATE_EPS = 1e-9

TOP_K = 10
SLICES = p2.SLICES
METRIC_KEYS = ("recall@1", "recall@5", "recall@10", "mrr@10")
# Re-exported slice names (the gate's own tests and any consumer read them here).
SLICE_EXACT_ID = p2.SLICE_EXACT_ID
SLICE_VI = p2.SLICE_VI
SLICE_VI_NODIAC = p2.SLICE_VI_NODIAC
SLICE_EN = p2.SLICE_EN
SLICE_SHORT = p2.SLICE_SHORT
SLICE_LONG = p2.SLICE_LONG

ARM_XS = "xs_cls_baseline"
ARM_REFERENCE = "mv2_fp_reference"
ARM_ONNX_FP = "mv2_onnx_fp"
ARM_ONNX_INT8 = "mv2_onnx_int8"
ARM_ONNX_MRL256 = "mv2_onnx_int8_mrl256"
MV2_ARMS = (ARM_REFERENCE, ARM_ONNX_FP, ARM_ONNX_INT8, ARM_ONNX_MRL256)

_SLICE_DEFINITIONS = {
    p2.SLICE_EXACT_ID: "exact identifier / path lookup (rare-token queries)",
    p2.SLICE_VI: "Vietnamese with diacritics (semantic queries)",
    p2.SLICE_VI_NODIAC: "Vietnamese typed WITHOUT tone marks — same rows, differently typed query",
    p2.SLICE_EN: "English semantic queries",
    p2.SLICE_SHORT: "one-token queries",
    p2.SLICE_LONG: "long prose queries (phrase-like, many tokens)",
}


# ── the signed gate (pure; unit-tested without any model) ────────────────────

def evaluate_gate(
    per_slice_deltas: dict[str, float],
    *,
    max_slice_drop: float = GATE_MAX_SLICE_DROP,
    min_vi_gain: float = GATE_MIN_VI_GAIN,
) -> dict:
    """The signed D1 rule as a pure function of per-slice recall@5 deltas.

    ``per_slice_deltas``: ``slice -> arm_recall@5 - baseline_recall@5`` (positive
    improves). ELIGIBLE iff every slice's drop is <= ``max_slice_drop`` AND both
    VI slices gain >= ``min_vi_gain`` (``vi_diacritics`` is the primary slice;
    ``vi_no_diacritics`` is recorded separately — if the two disagree the verdict
    FAILS and names the slice). Missing slices are a programming error: the
    non-inferiority clause is over EVERY slice, so an incomplete map cannot be
    evaluated.
    """
    missing = [name for name in SLICES if name not in per_slice_deltas]
    if missing:
        raise ValueError(
            f"missing slice deltas {missing!r} — the signed gate is non-inferiority on EVERY "
            f"slice ({list(SLICES)})"
        )
    deltas = {name: float(per_slice_deltas[name]) for name in SLICES}
    non_finite = [name for name, value in deltas.items() if not np.isfinite(value)]
    if non_finite:
        raise ValueError(
            f"non-finite slice deltas {non_finite!r} — NaN comparisons would read as "
            f"'not failing' and could falsely pass the signed gate"
        )
    failing = sorted(name for name in SLICES if deltas[name] < -max_slice_drop - _GATE_EPS)
    gains = {VI_PRIMARY: deltas[VI_PRIMARY], VI_SECONDARY: deltas[VI_SECONDARY]}
    vi_failing = [
        name for name in (VI_PRIMARY, VI_SECONDARY) if deltas[name] < min_vi_gain - _GATE_EPS
    ]
    eligible = not failing and not vi_failing
    if eligible:
        reason = (
            "eligible: every slice non-inferior (max drop <= "
            f"{max_slice_drop} recall@5) and both VI slices superior: "
            f"{VI_PRIMARY} {gains[VI_PRIMARY]:+.4f}, {VI_SECONDARY} {gains[VI_SECONDARY]:+.4f} "
            f"(>= +{min_vi_gain})"
        )
    else:
        parts = []
        if failing:
            parts.append(
                "non-inferiority violated on "
                + ", ".join(f"{name} ({deltas[name]:+.4f})" for name in failing)
            )
        if vi_failing:
            parts.append(
                "VI superiority not met on "
                + ", ".join(f"{name} ({deltas[name]:+.4f} < +{min_vi_gain})" for name in vi_failing)
            )
        reason = "not eligible: " + "; ".join(parts)
    return {
        "eligible": eligible,
        "verdict": "PASS" if eligible else "FAIL",
        "thresholds": {
            "max_slice_drop_recall@5": max_slice_drop,
            "min_vi_gain_recall@5": min_vi_gain,
            "vi_primary_slice": VI_PRIMARY,
            "vi_secondary_slice": VI_SECONDARY,
        },
        "non_inferior": {"passed": not failing, "failing_slices": failing},
        "vi_superiority": {"passed": not vi_failing, "failing_slices": vi_failing, "gains": gains},
        "per_slice_deltas": deltas,
        "reason": reason,
    }


# ── the fixture (imported verbatim from the p2 ablation) ─────────────────────

def build_fixture() -> dict:
    return p2.build_fixture()


corpus_hash = p2.corpus_hash
query_set_hash = p2.query_set_hash


def visibility_hash(fixture: dict) -> str:
    """sha256 over every ranking-relevant field, INCLUDING ``superseded``.

    ``p2.corpus_hash`` covers id/tenant/title/content only; flipping a row's
    visibility changes recall while that hash stays equal. This one pins the
    served set too, next to the corpus hash.
    """
    digest = hashlib.sha256()
    for row in sorted(fixture["rows"], key=lambda r: str(r["memory_id"])):
        digest.update(
            f"{row['memory_id']}\t{row['user_id']}\t{row['title']}\t{row['content']}\t"
            f"{1 if row['superseded'] else 0}\n".encode()
        )
    return f"sha256:{digest.hexdigest()}"


def corpus_documents(rows: list[dict]) -> list[str]:
    """The text every arm embeds (``vector_store._memory_to_document``, as p2)."""
    return p2._corpus_documents(rows)


# ── the ranking seam: exact cosine top-k, tenant-filtered, (-score, id) ──────

def exact_cosine_topk(
    query: np.ndarray,
    documents: np.ndarray,
    ids: list[str],
    tenants: np.ndarray,
    *,
    visible: np.ndarray,
    user_id: str,
    top_k: int,
) -> list[str]:
    """``search_memories`` over the arms' real vectors — the p2 ``ExactCosineStore`` seam.

    Same surface: tenant filter, served-visibility filter, ``(-score, memory_id)``
    deterministic sort, top-k. Production's Qdrant HNSW APPROXIMATES this exact
    order; at fixture scale the exact order is the one the store is trying to
    reproduce.
    """
    vector = np.asarray(query, dtype=np.float64)
    vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
    mask = (tenants == str(user_id)) & visible
    scores = np.asarray(documents, dtype=np.float64)[mask] @ vector
    candidates = [str(value) for value in np.asarray(ids, dtype=object)[mask]]
    order = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), candidates[i]))
    return [candidates[i] for i in order[:top_k]]


def query_metrics(served: list[str], golds: list[str]) -> dict:
    def recall(k: int) -> float:
        return sum(1 for gold in golds if gold in served[:k]) / len(golds)

    rank = next((i + 1 for i, sid in enumerate(served) if sid in golds), None)
    # `served` is the TOP_K-length ranking, so this is mrr@10 by construction —
    # a gold outside the served list counts 0, exactly like recall@10.
    return {
        "recall@1": recall(1),
        "recall@5": recall(5),
        "recall@10": recall(10),
        "mrr@10": 0.0 if rank is None else 1.0 / rank,
    }


# ── the arms' embedders ──────────────────────────────────────────────────────

class Mv2SoloText(mv2.Mv2Onnx):
    """The INT8 arms' batch policy: ONE text per call, so batch composition is fixed.

    T1 parity finding (eval/mv2/parity_report.json): the INT8 export quantizes
    activations with a per-TENSOR dynamic scale, so a row's vector depends on
    its batch mates (measured cos 0.95-0.98 vs the same row embedded alone). A
    single-row batch makes every vector bit-exact for its text; it is slower per
    text and that is the point. The FP32 arms batch (float reassociation only,
    inside the parity tolerance) — the policy is recorded per arm.
    """

    def _embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            out.extend(super()._embed([text]))
        return out


def _verify_cached_digests(pairs: dict[Path, str]) -> None:
    """Digest-check cached artifacts before anything constructs a session.

    Existence was the old guard; a substituted-but-loadable export would then
    silently feed the signed gate. Digests, never downloads (CI stays offline).
    """
    from eval.mv2 import runner as _mv2

    for path, expected in pairs.items():
        if not path.exists():
            raise RuntimeError(f"cached artifact missing: {path}")
        actual = _mv2._digest(path)
        if actual != expected:
            raise RuntimeError(f"cached artifact digest mismatch for {path}: {actual} != {expected}")


def _onnx_embedders(artifact: str, dim: int, *, solo: bool):
    model_dir = mv2.model_dir()
    _verify_cached_digests({
        model_dir / (mv2.INT8_FILE if artifact == "int8" else mv2.FP32_FILE):
            mv2.INT8_SHA256 if artifact == "int8" else mv2.FP32_SHA256,
        model_dir / mv2.TOKENIZER_FILE: mv2.TOKENIZER_SHA256,
    })
    path = mv2.model_dir() / (mv2.INT8_FILE if artifact == "int8" else mv2.FP32_FILE)
    embedding_class = Mv2SoloText if solo else mv2.Mv2Onnx
    embedder = embedding_class(path, dim=dim, prefix=mv2.QUERY_PREFIX)
    return embedder.embed_queries, embedder.embed_passages


def _reference_embedders():
    from eval.mv2 import reference

    embedder = reference.ReferenceEmbedder(dim=768)
    reference.verify_loaded_weights()
    return embedder.embed_queries, embedder.embed_passages


def _xs_embedders():
    model_dir = e5_local.model_dir()
    _verify_cached_digests({
        model_dir / e5_local.ARCTIC_MODEL_FILE: e5_local.ARCTIC_MODEL_SHA256,
        model_dir / e5_local.ARCTIC_TOKENIZER_FILE: e5_local.ARCTIC_TOKENIZER_SHA256,
    })
    return e5_local.arctic_embed_queries, e5_local.arctic_embed_passages


ARM_SPECS: dict[str, dict] = {
    ARM_XS: {
        "model": "arctic-embed-xs ONNX (the shipped local default)",
        "dim": 384,
        "pooling": "cls",
        "batch_policy": "passages batched (e5_local _BATCH=128, longest-padded); queries singleton",
        "build": _xs_embedders,
    },
    ARM_REFERENCE: {
        "model": "official custom code, pinned revision (eval/mv2/reference.py)",
        "dim": 768,
        "pooling": "cls",
        "batch_policy": f"passages batched (torch CPU float32, batch={16}); queries singleton",
        "build": _reference_embedders,
    },
    ARM_ONNX_FP: {
        "model": f"{mv2.FP32_FILE} (onnx/model.onnx, pinned revision)",
        "dim": 768,
        "pooling": "cls",
        "batch_policy": f"passages batched (Mv2Onnx _BATCH={mv2._BATCH}); queries singleton",
        "build": lambda: _onnx_embedders("fp32", 768, solo=False),
    },
    ARM_ONNX_INT8: {
        "model": f"{mv2.INT8_FILE} (onnx/model_int8.onnx, pinned revision)",
        "dim": 768,
        "pooling": "cls",
        "batch_policy": "ONE TEXT PER CALL (T1: per-tensor dynamic quantization)",
        "build": lambda: _onnx_embedders("int8", 768, solo=True),
    },
    ARM_ONNX_MRL256: {
        "model": f"{mv2.INT8_FILE} (onnx/model_int8.onnx, same session), MRL dim=256",
        "dim": 256,
        "pooling": "cls (slice to 256, then L2 AGAIN — not the 128-byte claim)",
        "batch_policy": "ONE TEXT PER CALL (T1: per-tensor dynamic quantization)",
        "build": lambda: _onnx_embedders("int8", 256, solo=True),
    },
}


def run_arm(fixture: dict, embed_queries, embed_passages) -> dict:
    rows = fixture["rows"]
    ids = [str(row["memory_id"]) for row in rows]
    tenants = np.array([str(row["user_id"]) for row in rows])
    visible = np.array([not row["superseded"] for row in rows])
    documents = np.asarray(embed_passages(corpus_documents(rows)), dtype=np.float64)
    if (documents.ndim != 2 or documents.shape[0] != len(rows)
            or not np.isfinite(documents).all()):
        raise ValueError(
            f"invalid passage embedding array: shape={documents.shape}, expected "
            f"({len(rows)}, dim) with finite values"
        )
    document_norms = np.linalg.norm(documents, axis=1, keepdims=True)
    if np.any(document_norms <= 0.0):
        raise ValueError("passage embeddings contain zero-norm rows")
    documents = documents / document_norms

    records: list[dict] = []
    queries = np.zeros((len(fixture["queries"]), documents.shape[1]), dtype=np.float64)
    for index, query in enumerate(fixture["queries"]):
        golds = [str(p2._memory_id(key)) for key in query["golds"]]
        embedded = np.asarray(embed_queries([query["text"]]), dtype=np.float64)
        if embedded.shape != (1, documents.shape[1]) or not np.isfinite(embedded).all():
            raise ValueError(f"invalid query embedding array: shape={embedded.shape}")
        norm = float(np.linalg.norm(embedded[0]))
        if norm <= 0.0:
            raise ValueError(f"query {index} embedding has zero norm")
        queries[index] = embedded[0] / norm
        served = exact_cosine_topk(
            queries[index], documents, ids, tenants,
            visible=visible, user_id=str(p2.TENANT_A), top_k=TOP_K,
        )
        records.append(
            {
                "key": query["key"],
                "slice": query["slice"],
                "served": served,
                "metrics": query_metrics(served, golds),
            }
        )
    return {"records": records, "documents": documents, "queries": queries}


def _cosine_pair(first: np.ndarray, second: np.ndarray) -> dict:
    cosines = np.clip((first * second).sum(axis=1), -1.0, 1.0)
    return {
        "rows": len(cosines),
        "mean_cosine": round(float(cosines.mean()), 6),
        "min_cosine": round(float(cosines.min()), 6),
    }


def _vi_query_form_cosines(fixture: dict, runs: dict[str, dict]) -> dict:
    """Cosine between the SAME VI query typed with diacritics vs without tone marks.

    This is the measurement that makes the decisive slice legible: an arm whose
    two forms collapse to one vector (cosine 1.0) is answering the no-diacritics
    slice with the identical query, while an arm below 1.0 is genuinely reading
    different tokens — so a delta there is a capability difference, not plumbing.
    """
    accented = [i for i, q in enumerate(fixture["queries"]) if q["slice"] == SLICE_VI]
    plain = [i for i, q in enumerate(fixture["queries"]) if q["slice"] == SLICE_VI_NODIAC]
    if len(accented) != len(plain) or not accented:
        return {"status": "not measured — the two VI slices do not pair up"}
    return {
        arm: _cosine_pair(run["queries"][accented], run["queries"][plain])
        for arm, run in runs.items()
    }


def _cross_checks(runs: dict[str, dict], fixture: dict) -> dict:
    """The sanity checks the D3 arm list invites.

    ``reference_vs_onnx_fp``: both arms consume identical text through their own
    pinned pipelines; their vectors agreeing is the evidence that the ONNX
    export IS the official code path (it is not a quality gate — the slices
    decide that).
    """
    checks = {
        "vi_query_form_cosine_per_arm": {
            "pairs": len(
                [q for q in fixture["queries"] if q["slice"] == SLICE_VI]
            ),
            "per_arm": _vi_query_form_cosines(fixture, runs),
            "note": (
                "cosine between each VI query typed with diacritics and the same query typed "
                "without tone marks: 1.0 means the two forms are the same vector to that "
                "embedder"
            ),
        }
    }
    if ARM_REFERENCE not in runs or ARM_ONNX_FP not in runs:
        checks["reference_vs_onnx_fp"] = {"status": "not run — both arms must have run"}
        return checks
    checks["reference_vs_onnx_fp"] = {
        "corpus_passages": _cosine_pair(
            runs[ARM_REFERENCE]["documents"], runs[ARM_ONNX_FP]["documents"]
        ),
        "queries": _cosine_pair(runs[ARM_REFERENCE]["queries"], runs[ARM_ONNX_FP]["queries"]),
        "note": (
            "row-wise cosine between the official custom code (fp32 torch) and the pinned "
            "onnx/model.onnx export on identical inputs; 1.0 means the export reproduces the "
            "reference"
        ),
    }
    return checks


def _aggregate(records: list[dict], key_of) -> dict:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(key_of(record), []).append(record["metrics"])
    return {
        name: {
            **{
                metric: round(statistics.fmean(item[metric] for item in items), 6)
                for metric in METRIC_KEYS
            },
            "n": len(items),
        }
        for name, items in sorted(groups.items())
    }


def _arm_payload(arm: str, records: list[dict]) -> dict:
    spec = ARM_SPECS[arm]
    metrics = [record["metrics"] for record in records]
    return {
        "status": "ran",
        "config": {
            "model": spec["model"],
            "dim": spec["dim"],
            "pooling": spec["pooling"],
            "prefix": mv2.QUERY_PREFIX if arm != ARM_XS else e5_local.ARCTIC_QUERY_PREFIX,
            "max_tokens": mv2.MAX_TOKENS,
            "batch_policy": spec["batch_policy"],
        },
        "per_slice": _aggregate(records, lambda record: record["slice"]),
        "overall": {
            **{
                metric: round(statistics.fmean(item[metric] for item in metrics), 6)
                for metric in METRIC_KEYS
            },
            "n": len(records),
        },
        "per_query": [
            {"key": record["key"], "slice": record["slice"], **record["metrics"]}
            for record in records
        ],
    }


def _arm_model_facts(arm: str) -> dict:
    """File-level fingerprint of what this arm actually loaded (best effort)."""
    try:
        if arm == ARM_XS:
            directory = e5_local.model_dir()
            files = {
                name: directory / name
                for name in (e5_local.ARCTIC_MODEL_FILE, e5_local.ARCTIC_TOKENIZER_FILE)
            }
        elif arm == ARM_REFERENCE:
            from eval.mv2 import reference

            return reference.artifact_facts()
        else:
            files = {
                name: mv2.model_dir() / name
                for name in (mv2.INT8_FILE if "int8" in arm else mv2.FP32_FILE, mv2.TOKENIZER_FILE)
            }
        return {
            name: {
                "path": str(path),
                "sha256": mv2._digest(path) if path.exists() else None,
                "size_bytes": path.stat().st_size if path.exists() else None,
            }
            for name, path in files.items()
        }
    except Exception as exc:  # fingerprinting must never sink a finished run
        return {"error": f"{type(exc).__name__}: {exc}"}


def _deltas(arms: dict) -> dict:
    reference_slices = arms[ARM_XS]["per_slice"]
    reference_overall = arms[ARM_XS]["overall"]
    deltas: dict[str, dict] = {}
    for arm, payload in arms.items():
        if arm == ARM_XS or payload.get("status") != "ran":
            continue
        deltas[arm] = {
            "per_slice": {
                slice_name: {
                    **{
                        metric: round(
                            payload["per_slice"][slice_name][metric]
                            - reference_slices[slice_name][metric],
                            6,
                        )
                        for metric in METRIC_KEYS
                    },
                    "recall@5_delta": round(
                        payload["per_slice"][slice_name]["recall@5"]
                        - reference_slices[slice_name]["recall@5"],
                        6,
                    ),
                }
                for slice_name in SLICES
            },
            "overall": {
                **{
                    metric: round(payload["overall"][metric] - reference_overall[metric], 6)
                    for metric in METRIC_KEYS
                },
                "recall@5_gain": round(
                    payload["overall"]["recall@5"] - reference_overall["recall@5"], 6
                ),
            },
        }
    return deltas


def gate_verdicts(arms: dict, deltas: dict) -> dict:
    """One signed-gate verdict per m-v2 arm; a failed arm is NOT_RUN, never invented."""
    verdicts: dict[str, dict] = {}
    for arm in MV2_ARMS:
        payload = arms.get(arm)
        if payload is None:
            verdicts[arm] = {
                "verdict": "NOT_RUN",
                "eligible": None,
                "reason": "arm not requested in this run",
            }
        elif payload.get("status") != "ran":
            verdicts[arm] = {
                "verdict": "NOT_RUN",
                "eligible": None,
                "reason": f"arm failed to run: {payload.get('error')}",
            }
        else:
            verdicts[arm] = evaluate_gate(
                {name: deltas[arm]["per_slice"][name]["recall@5_delta"] for name in SLICES}
            )
    return verdicts


# ── fingerprint (§10.1) ──────────────────────────────────────────────────────

def _git_state() -> dict:
    def _run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout

    porcelain = _run("status", "--porcelain").strip()
    return {
        "head": _run("rev-parse", "HEAD").strip(),
        "dirty": bool(porcelain),
        "dirty_paths": sorted(
            line[3:].strip() for line in porcelain.splitlines() if line.strip()
        ),
    }


def _memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def fingerprint() -> dict:
    import onnxruntime
    import tokenizers

    runtime: dict = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": _memory_bytes(),
        "onnxruntime": onnxruntime.__version__,
        "tokenizers": tokenizers.__version__,
        "numpy": np.__version__,
        "torch": None,
        "transformers": None,
    }
    try:
        from eval.mv2 import reference

        facts = reference.artifact_facts()
        runtime["torch"] = facts.get("torch")
        runtime["transformers"] = facts.get("transformers")
    except Exception:  # torch/transformers are experiment-only deps
        pass
    return {
        "git": _git_state(),
        "runtime": runtime,
        "model_pins": {"repo": mv2.HF_REPO, "revision": mv2.HF_REVISION},
        "seeds": {"fixture_seed": p2.SEED, "ranking": "deterministic; no other randomness"},
        "top_k": TOP_K,
        "concurrency": "single process, single thread of control (ORT session defaults)",
        "warmup": "none — every query is embedded on its own call",
    }


# ── the run ──────────────────────────────────────────────────────────────────

def run_ablation(*, arms_requested: list[str] | None = None) -> dict:
    fixture = build_fixture()
    arms: dict[str, dict] = {}
    runs: dict[str, dict] = {}
    for arm, spec in ARM_SPECS.items():
        if arms_requested and arm not in arms_requested:
            continue
        try:
            embed_queries, embed_passages = spec["build"]()
            run = run_arm(fixture, embed_queries, embed_passages)
        except Exception as exc:  # a failed arm is recorded, never fabricated
            arms[arm] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "config": {
                    "model": spec["model"],
                    "dim": spec["dim"],
                    "batch_policy": spec["batch_policy"],
                },
            }
            print(f"[arm] {arm}: FAILED — {type(exc).__name__}: {exc}")
            continue
        arms[arm] = _arm_payload(arm, run["records"])
        arms[arm]["fingerprint"] = _arm_model_facts(arm)
        runs[arm] = run
        print(f"[arm] {arm}: ran ({len(run['records'])} queries)")

    if arms.get(ARM_XS, {}).get("status") != "ran":
        raise RuntimeError(
            "the xs_cls_baseline arm must run — every delta and the whole signed gate are "
            "relative to it"
        )
    deltas = _deltas(arms)
    verdicts = gate_verdicts(arms, deltas)
    eligible = sorted(arm for arm, verdict in verdicts.items() if verdict.get("eligible"))

    return {
        "artifact": "m-v2-ablation-retrieval",
        "plan": "2026-09-19-model-scale-gates (T2 / D3)",
        "scales": {
            "kind": "fixture",
            "memories": len(fixture["rows"]),
            "queries": len(fixture["queries"]),
            "top_k": TOP_K,
            "note": "EVIDENCE AT FIXTURE SCALE — never a production claim",
        },
        "fingerprint": fingerprint(),
        "fixture": {
            "source": "eval/ablation_retrieval_p2.py:build_fixture() — imported, not copied",
            "seed": fixture["seed"],
            "corpus_hash": corpus_hash(fixture),
            "visibility_hash": visibility_hash(fixture),
            "query_set_hash": query_set_hash(fixture),
            "slices": list(SLICES),
            "slice_definitions": _SLICE_DEFINITIONS,
            "queries_per_slice": {
                name: sum(1 for query in fixture["queries"] if query["slice"] == name)
                for name in SLICES
            },
            "query_ids": [query["key"] for query in fixture["queries"]],
            "rows": len(fixture["rows"]),
            "superseded_rows_excluded_before_topk": sum(
                1 for row in fixture["rows"] if row["superseded"]
            ),
        },
        "arms": arms,
        "deltas_vs_xs_cls_baseline": deltas,
        "cross_checks": _cross_checks(runs, fixture),
        "verdict": {
            "gate": (
                "SIGNED D1 (§12, 2026-09-19): eligible iff non-inferior on EVERY slice "
                "(drop <= 0.02 recall@5 vs xs_cls_baseline) AND superior on the VI slices "
                "(gain >= +0.02 recall@5); vi_diacritics is the primary VI slice, "
                "vi_no_diacritics is recorded separately — disagreement fails"
            ),
            "thresholds": {
                "max_slice_drop_recall@5": GATE_MAX_SLICE_DROP,
                "min_vi_gain_recall@5": GATE_MIN_VI_GAIN,
                "vi_primary_slice": VI_PRIMARY,
                "vi_secondary_slice": VI_SECONDARY,
            },
            "arms": verdicts,
            "eligible_arms": eligible,
            "authority": (
                "this artifact never flips a flag: T3 (opt-in LOCAL_EMBED_MODEL=arctic-m-v2) is "
                "conditional on an eligible arm, the default stays `arctic` regardless"
            ),
        },
        "substitutions": {
            "ranking": (
                "exact cosine top-k in numpy over real vectors, tenant + visibility filtered, "
                "sorted (-score, memory_id) — the p2 ExactCosineStore seam; production's Qdrant "
                "HNSW approximates this order"
            ),
            "pipeline": (
                "no SQL hydration/refill/rerank: identical for every arm, and the only variable "
                "under test is the embedder"
            ),
            "batch_composition_decision": (
                "fixed per arm (T1 parity finding): INT8 arms embed ONE text per call "
                "(per-tensor dynamic quantization makes vectors batch-dependent), float32 arms "
                "batch — see each arm's config.batch_policy"
            ),
            "reference": (
                "the official custom modeling code, pinned revision, downloaded + sha256-checked "
                "+ reviewed; classes registered locally, transformers never fetches remote code"
            ),
            "rewrite_modifiers": "identity / uniform by construction (as p2: one captured_at, one salience, no pins)",
        },
        "limitations": [
            "fixture scale (hundreds of memories, 48 queries): the §12.2 budgets are signed at this scale",
            "generated corpus (the p2 fixture), not user traffic",
            "the vi_diacritics slice sits at ceiling for BOTH xs (1.0) and every m-v2 arm (1.0) — the signed gate's superiority clause has no headroom there at this scale; what discriminates is the no-diacritics form of the same queries (see cross_checks.vi_query_form_cosine_per_arm)",
            "exact-cosine ranking, not Qdrant ANN; no SQL hydration/refill/rerank stage",
            "superseded rows are excluded before the top-k cut (p2's pipeline filtered them after a deeper pool fetch + refill), so recall counts can differ from p2's dense_only arm where superseded near-misses compete (measured: the xs arm reproduces p2's dense_only per-slice recall@5 exactly)",
            "CPU only; no latency/SLO claim — this artifact is about quality, T4/T5 measure scale",
            "no LLM query rewrite (identity, as p2)",
        ],
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _print_summary(artifact: dict, out: Path) -> None:
    arms = artifact["arms"]
    print(f"\nfixture: seed={artifact['fixture']['seed']} rows={artifact['scales']['memories']} "
          f"queries={artifact['scales']['queries']} corpus={artifact['fixture']['corpus_hash'][:19]}…")
    header = f"{'arm':<22}" + "".join(f"{name:>10}" for name in METRIC_KEYS) + f"{'n':>5}  status"
    print(header)
    for arm, payload in arms.items():
        if payload.get("status") != "ran":
            print(f"{arm:<22}{'—':>40}     failed")
            continue
        overall = payload["overall"]
        row = f"{arm:<22}" + "".join(f"{overall[name]:>10.4f}" for name in METRIC_KEYS)
        print(f"{row}{overall['n']:>5}  ran")
    print()
    for arm, verdict in artifact["verdict"]["arms"].items():
        print(f"gate {arm:<22} {verdict['verdict']:<7} {verdict['reason']}")
    print(f"\neligible arms: {artifact['verdict']['eligible_arms'] or 'none'}")
    print(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="m-v2 ablation (T2/D3) with the signed D1 gate")
    parser.add_argument("--out", default=str(ARTIFACT_PATH), help="artifact path")
    parser.add_argument(
        "--arms",
        default=None,
        help="comma-separated subset of arms (debugging only; the real run uses defaults)",
    )
    args = parser.parse_args()
    requested = [name.strip() for name in args.arms.split(",")] if args.arms else None

    artifact = run_ablation(arms_requested=requested)
    out = Path(args.out)
    out.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _print_summary(artifact, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
