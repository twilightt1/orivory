# THROWAWAY SPIKE — không import từ app runtime
"""Measure the current Arctic mean baseline against a CLS reference."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.retrieval import e5_local  # noqa: E402

CASE_LABELS = ("short", "empty", "long", "padded-batch")
CASES = ["ngắn", "", "dài " * 400, "  padded   spacing  "]


def _cls_reference(texts: list[str], *, query: bool) -> tuple[list[list[float]], list[tuple[int, ...]]]:
    sess, tok = e5_local._asession(), e5_local._atokenizer()
    prefix = e5_local.ARCTIC_QUERY_PREFIX if query else ""
    out: list[list[float]] = []
    shapes: list[tuple[int, ...]] = []
    for text in [prefix + value for value in texts]:
        enc = tok.encode(text)
        ids = np.array([enc.ids[:512]], dtype=np.int64)
        mask = np.array([enc.attention_mask[:512]], dtype=np.int64)
        last = np.asarray(sess.run(None, e5_local._feed(sess, ids, mask))[0])
        shapes.append(tuple(int(size) for size in last.shape))
        if last.ndim != 3 or last.shape[0] != 1 or last.shape[1] < 1:
            raise ValueError(
                f"CLS reference unavailable: ONNX output is not token embeddings; actual shape={last.shape}"
            )
        value = last[:, 0, :]
        out.append((value / max(np.linalg.norm(value), 1e-12)).tolist()[0])
    return out, shapes


def _cosines(mean: list[list[float]], cls: list[list[float]]) -> np.ndarray:
    mean_array = np.asarray(mean)
    cls_array = np.asarray(cls)
    if mean_array.shape != cls_array.shape:
        raise ValueError(f"mean/CLS shape mismatch: {mean_array.shape} != {cls_array.shape}")
    cosines = (mean_array * cls_array).sum(axis=1)
    if not np.all(np.isfinite(cosines)):
        raise ValueError(f"non-finite cosine values: {cosines}")
    return cosines


def _format_shapes(shapes: list[tuple[int, ...]]) -> str:
    return ", ".join(f"{label}={shape}" for label, shape in zip(CASE_LABELS, shapes, strict=True))


def main() -> int:
    if not e5_local.arctic_files_cached():
        print("SKIP: arctic ONNX cache missing — no model download attempted")
        return 0

    means = {
        "query": e5_local.arctic_embed_queries(CASES),
        "passage": e5_local.arctic_embed_passages(CASES),
    }
    cls_query, query_shapes = _cls_reference(CASES, query=True)
    cls_passage, passage_shapes = _cls_reference(CASES, query=False)
    cosines = {
        "query": _cosines(means["query"], cls_query),
        "passage": _cosines(means["passage"], cls_passage),
    }

    print("XS Arctic mean-vs-CLS parity probe")
    print(f"model_dir: {e5_local.model_dir()}")
    print(f"ONNX output shape (query):   {_format_shapes(query_shapes)}")
    print(f"ONNX output shape (passage): {_format_shapes(passage_shapes)}")
    print("case                  query_cosine    passage_cosine")
    for label, query_cos, passage_cos in zip(CASE_LABELS, cosines["query"], cosines["passage"], strict=True):
        print(f"{label:<20} {query_cos:>13.9f} {passage_cos:>16.9f}")

    baseline_locked = all(np.any(values < 0.99) for values in cosines.values())
    print(f"PARITY: {'PASS' if baseline_locked else 'FAIL'}")
    return 0 if baseline_locked else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"PARITY: FAIL ({exc})")
        raise SystemExit(1) from exc
