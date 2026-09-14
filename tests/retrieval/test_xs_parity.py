"""XS parity gate: production CLS must equal the CLS reference (P1b Task 4)."""

from __future__ import annotations

import numpy as np
import pytest

from app.retrieval import e5_local

pytestmark = pytest.mark.skipif(
    not e5_local.arctic_files_cached(),
    reason="arctic onnx cache missing — run local, do not download in CI",
)

CASES = ["ngắn", "", "dài " * 400, "  padded   spacing  "]


def _cls_reference(texts: list[str], *, query: bool) -> list[list[float]]:
    sess, tok = e5_local._asession(), e5_local._atokenizer()
    prefix = e5_local.ARCTIC_QUERY_PREFIX if query else ""
    out = []
    for text in [prefix + value for value in texts]:
        enc = tok.encode(text)
        ids = np.array([enc.ids[:512]], dtype=np.int64)
        mask = np.array([enc.attention_mask[:512]], dtype=np.int64)
        last = sess.run(None, e5_local._feed(sess, ids, mask))[0]
        if last.ndim != 3:
            raise AssertionError(f"CLS reference requires token embeddings; got shape {last.shape}")
        value = last[:, 0, :]
        out.append((value / max(np.linalg.norm(value), 1e-12)).tolist()[0])
    return out


def test_parity_report_shape_and_finiteness():
    for embed in (e5_local.arctic_embed_queries, e5_local.arctic_embed_passages):
        got = np.asarray(embed(CASES))
        assert got.shape == (4, 384)
        assert np.issubdtype(got.dtype, np.floating)
        assert np.all(np.isfinite(got))
        assert np.allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-5)


def test_query_and_passage_outputs_are_deterministic():
    for embed in (e5_local.arctic_embed_queries, e5_local.arctic_embed_passages):
        first = np.asarray(embed(CASES))
        second = np.asarray(embed(CASES))
        assert np.array_equal(first, second)


def _mean_of(texts: list[str]) -> list[list[float]]:
    return e5_local._encode_with(
        texts, e5_local._asession, e5_local._atokenizer, pooling="mean"
    )


def test_production_arctic_is_the_cls_reference():
    """Cutover: the batched production path IS the reference (float noise only)."""
    for query in (True, False):
        texts = [e5_local.ARCTIC_QUERY_PREFIX + t for t in CASES] if query else CASES
        embed = e5_local.arctic_embed_queries if query else e5_local.arctic_embed_passages
        production = np.asarray(embed(CASES))
        reference = np.asarray(_cls_reference(CASES, query=query))
        assert np.allclose(production, reference, atol=1e-5), np.abs(
            production - reference
        ).max()
        assert np.allclose(production, _mean_of(texts)) is False


def test_legacy_mean_differs_from_production_cls():
    """The preserved mean contract documents a real, measurable change."""
    for query in (True, False):
        texts = [e5_local.ARCTIC_QUERY_PREFIX + t for t in CASES] if query else CASES
        embed = e5_local.arctic_embed_queries if query else e5_local.arctic_embed_passages
        cos = (np.asarray(_mean_of(texts)) * np.asarray(embed(CASES))).sum(1)
        assert float(cos.max()) < 0.99, f"mean≈cls on all cases: {cos}"
