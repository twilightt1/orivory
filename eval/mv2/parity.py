#!/usr/bin/env python
"""T1 parity harness — arctic-embed-m-v2.0 ONNX (INT8 + FP32), fixture scale.

What this artifact decides
--------------------------
Nothing about retrieval quality — that is T2's ablation. This harness decides
whether the two pinned exports are USABLE at all through :class:`Mv2Onnx`:
deterministic shape, finite unit-norm vectors, bit-exact repeats, the MRL-256
contract (slice → renormalize), the query/passage prefix contract, and — the
one that is not obvious — how much a row's vector moves when its batch mates
change. Every number below is measured on this box, on this run, from the
pinned artifacts. A FAIL blocks T2's arms.

Measured reality this harness records (do not "fix" it silently)
----------------------------------------------------------------
The INT8 export quantizes activations with a per-TENSOR dynamic scale, so a
row's vector depends on the other rows in its batch (and on padding): measured
cosine ~0.95–0.98 vs the same row embedded alone. FP32 is batch-invariant up to
float reassociation. Repeats and identical batch mates are bit-exact. The T2
ablation should therefore hold batch composition fixed per arm (or embed one
text per call) — this report exists so that is a recorded decision, not a
surprise.

Scale, stated plainly
---------------------
A handful of short fixtures (VI with diacritics, EN, empty string, and one
>MAX_TOKENS text) per export, one process, CPUExecutionProvider, default ORT
threads. Absolute latencies are not recorded — correctness harness, not a
benchmark. ARM64 local is the measured platform; x86_64 parity is `not-claimed`
(no support is published, so no heavy CI job is added).

Run: ``python eval/mv2/parity.py`` — downloads what is missing (~1.5 GB once,
into ``~/.cache/orivory/mv2``), writes ``eval/mv2/parity_report.json`` and
exits non-zero if any check fails.
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.mv2 import runner as mv2  # noqa: E402

ARTIFACT = Path(__file__).resolve().parent / "parity_report.json"
TOL = 1e-4
FIXTURES = [
    "Tôi đã gặp Huy ở quán cà phê gần hồ Tây vào chiều thứ Sáu.",
    "the app crashed when i tapped export after the update",
    "",
]
LONG_TEXT = " ".join(f"token{i}" for i in range(3000))

# Per-export envelope for "row i of a mixed batch vs that row alone" (see the
# module docstring). Gated, recorded, never silently relaxed.
ENVELOPE = {
    "int8": {
        "cos_floor": 0.95,
        "max_abs_tol": None,
        "why": "per-tensor dynamic quantization: batch mates and padding move the activation scale",
    },
    "fp32": {
        "cos_floor": None,
        "max_abs_tol": TOL,
        "why": "float32 reassociation only",
    },
}


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _collect(runner, label: str, path: Path) -> list[dict]:
    checks: list[dict] = []
    envelope = ENVELOPE[label]

    def add(name: str, passed: bool, **detail) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    add(
        "empty_batch_contract",
        bool(runner.embed_queries([]) == [] and runner.embed_passages([]) == []),
    )

    short = np.asarray(runner.embed_passages(FIXTURES), dtype=float)
    norms = np.linalg.norm(short, axis=1)
    add(
        "short_batch_shape_norms",
        bool(
            short.shape == (len(FIXTURES), 768)
            and np.isfinite(short).all()
            and np.allclose(norms, 1.0, atol=TOL)
        ),
        shape=list(short.shape),
        max_norm_error=float(np.abs(norms - 1.0).max()),
    )

    add(
        "determinism_repeat_call",
        bool(np.array_equal(np.asarray(runner.embed_passages(FIXTURES)), short)),
    )

    t0 = FIXTURES[0]
    dup = np.asarray(runner.embed_passages([t0, t0]), dtype=float)
    single_t0 = np.asarray(runner.embed_passages([t0]), dtype=float)[0]
    add(
        "identical_batch_mates_bit_exact",
        bool(np.array_equal(dup[0], dup[1]) and np.array_equal(dup[0], single_t0)),
        max_abs_drift_vs_single=float(np.abs(dup[0] - single_t0).max()),
        max_abs_drift_vs_mixed_batch=float(np.abs(dup[0] - short[0]).max()),
    )

    rows = []
    for i, text in enumerate(FIXTURES):
        single = np.asarray(runner.embed_passages([text]), dtype=float)[0]
        rows.append(
            {
                "row": i,
                "cos": _cos(short[i], single),
                "max_abs_drift": float(np.abs(short[i] - single).max()),
            }
        )
    worst_cos = min(r["cos"] for r in rows)
    worst_abs = max(r["max_abs_drift"] for r in rows)
    gated = (
        worst_cos >= envelope["cos_floor"]
        if envelope["cos_floor"] is not None
        else worst_abs <= envelope["max_abs_tol"]
    )
    add(
        "batch_mate_envelope",
        gated and bool(np.isfinite(short).all()),
        rows=rows,
        worst_cos=worst_cos,
        worst_max_abs_drift=worst_abs,
        cos_floor=envelope["cos_floor"],
        max_abs_tol=envelope["max_abs_tol"],
        why=envelope["why"],
    )

    mrl = mv2.Mv2Onnx(path, dim=256)
    sliced = np.asarray(mrl.embed_passages(FIXTURES), dtype=float)
    expected = short[:, :256] / np.linalg.norm(short[:, :256], axis=1, keepdims=True)
    add(
        "mrl_256_slice_then_renormalize",
        bool(
            sliced.shape == (len(FIXTURES), 256)
            and np.allclose(sliced, expected, atol=TOL)
            and np.allclose(np.linalg.norm(sliced, axis=1), 1.0, atol=TOL)
        ),
        max_abs_diff=float(np.abs(sliced - expected).max()),
    )
    del mrl

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(mv2.ensure_mv2_files("tokenizer")["tokenizer"]))
    measured = len(tok.encode(LONG_TEXT).ids)
    long_vec = np.asarray(runner.embed_passages([LONG_TEXT]), dtype=float)
    add(
        "long_input_capped",
        bool(
            min(measured, mv2.MAX_TOKENS) == mv2.MAX_TOKENS
            and long_vec.shape == (1, 768)
            and np.isfinite(long_vec).all()
            and abs(float(np.linalg.norm(long_vec[0])) - 1.0) <= TOL
        ),
        measured_tokens=measured,
        cap=mv2.MAX_TOKENS,
    )

    raises = []
    for bad in ("dim", "pool"):
        try:
            if bad == "dim":
                mv2.Mv2Onnx(path, dim=128)
            else:
                mv2.Mv2Onnx(path, pool="mean")
        except ValueError:
            raises.append(True)
        else:
            raises.append(False)
    add("dim_and_pool_contract", all(raises))

    probe = FIXTURES[0]
    add(
        "prefix_contract",
        bool(
            mv2.QUERY_PREFIX == "query: "
            and runner.embed_queries([probe])[0]
            == runner.embed_passages([mv2.QUERY_PREFIX + probe])[0]
        ),
    )

    # The export ships its own pooled output (sentence_embedding) — confirm it is
    # CLS-pooled and not, say, mean-pooled (which would invalidate T2's arms).
    enc = runner._tok.encode_batch([probe])
    ids = np.zeros((1, len(enc[0].ids)), dtype=np.int64)
    mask = np.zeros_like(ids)
    ids[0] = enc[0].ids
    mask[0] = enc[0].attention_mask
    token_emb, pooled = runner._sess.run(None, mv2._feed(runner._sess, ids, mask))
    cls = token_emb[:, 0, :]
    cls_n = cls / np.linalg.norm(cls, axis=1, keepdims=True)
    add(
        "export_pooled_output_is_cls",
        bool(_cos(pooled[0], cls_n[0]) >= 1.0 - TOL),
        cos=float(_cos(pooled[0], cls_n[0])),
        max_abs_diff=float(np.abs(pooled[0] - cls_n[0]).max()),
    )
    return checks


def _read_varint(f) -> int:
    shift = 0
    val = 0
    while True:
        b = f.read(1)
        if not b:
            raise EOFError("truncated varint")
        b = b[0]
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            return val
        shift += 7


def _read_opset_imports(path) -> list[int]:
    """Top-level ModelProto scan for field 8 (opset_import), seeking past the
    graph payload (field 7) instead of parsing it. Minimal wire-format reader —
    ModelProto top-level fields only; versions in file order."""
    opsets: list[int] = []
    with open(path, "rb") as f:
        while True:
            tag = f.read(1)
            if not tag:
                break
            field, wire = tag[0] >> 3, tag[0] & 7
            if wire == 0:
                _read_varint(f)
            elif wire == 1:
                f.seek(8, 1)
            elif wire == 5:
                f.seek(4, 1)
            elif wire == 2:
                length = _read_varint(f)
                if field != 8:
                    f.seek(length, 1)
                    continue
                payload = f.read(length)
                i = 0  # repeated OperatorSetIdProto {1: domain string, 2: version varint}
                while i < len(payload):
                    t = payload[i]
                    i += 1
                    if t >> 3 == 2 and t & 7 == 0:
                        v = sh = 0
                        while True:
                            b = payload[i]
                            i += 1
                            v |= (b & 0x7F) << sh
                            if not (b & 0x80):
                                break
                            sh += 7
                        opsets.append(v)
                    elif t & 7 == 2:
                        ln = sh = 0
                        while True:
                            b = payload[i]
                            i += 1
                            ln |= (b & 0x7F) << sh
                            if not (b & 0x80):
                                break
                            sh += 7
                        i += ln
                    else:
                        break
            else:
                break
    return opsets


def main() -> int:
    import onnxruntime as ort

    paths = mv2.ensure_mv2_files()
    report: dict = {
        "artifact": str(ARTIFACT.relative_to(ROOT)),
        "generated_at": datetime.now(UTC).isoformat(),
        "model": {"repo": mv2.HF_REPO, "revision": mv2.HF_REVISION},
        "runtime": {
            "ort_version": ort.__version__,
            "providers": ["CPUExecutionProvider"],
            "python": platform.python_version(),
            "machine": platform.machine(),
        },
        "config": {
            "pooling": "cls",
            "dim": 768,
            "max_tokens": mv2.MAX_TOKENS,
            "batch_size": mv2._BATCH,
            "tolerance": TOL,
            "x86_parity": "not-claimed",
        },
        "findings": {
            "batch_mate_sensitivity": (
                "INT8 vectors depend on batch composition (per-tensor dynamic "
                "quantization scale); see each model's batch_mate_envelope check "
                "for the measured cosine/drift. Hold batch composition fixed per "
                "arm, or embed one text per call, when the exact vector matters."
            ),
            "pooling": (
                "CLS pooling confirmed against the export's own sentence_embedding "
                "output (normalized CLS == sentence_embedding within tolerance)."
            ),
        },
        "artifacts": {},
        "checks": {},
    }
    failed = 0
    for label in ("int8", "fp32"):
        path = paths[label]
        runner = mv2.Mv2Onnx(path)
        report["artifacts"][label] = {
            "file": path.name,
            "path": str(path),
            "sha256": mv2._digest(path),
            "size_bytes": path.stat().st_size,
            "opset_import": _read_opset_imports(path),
            "inputs": [i.name for i in runner._sess.get_inputs()],
            "outputs": [o.name for o in runner._sess.get_outputs()],
            "output_used": runner.output_name or "first",
        }
        checks = _collect(runner, label, path)
        report["checks"][label] = checks
        failed += sum(1 for c in checks if not c["passed"])
        print(
            f"\n== {label} ({path.name}) — "
            f"{'PASS' if all(c['passed'] for c in checks) else 'FAIL'}"
        )
        for c in checks:
            print(f"  [{'ok' if c['passed'] else 'FAIL'}] {c['name']} {c['detail']}")
        del runner

    total = sum(len(c) for c in report["checks"].values())
    report["summary"] = {
        "checks": total,
        "passed": total - failed,
        "failed": failed,
        "result": "PASS" if failed == 0 else "FAIL",
    }
    ARTIFACT.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"\nsummary: {total - failed}/{total} checks passed — {report['summary']['result']}"
        f"\nort {ort.__version__} · {platform.machine()} · {ARTIFACT}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
