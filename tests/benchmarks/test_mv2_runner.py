"""m-v2 runner contract (fixture scale): real ONNX session when cached, else skip.

Every claim below embeds through the REAL arctic-embed-m-v2 INT8 ONNX session —
no mocks — so a cold cache SKIPS instead of downloading (house pattern from
tests/retrieval/test_event_loop_responsiveness.py). The cache lives outside the
repo (~/.cache/orivory/mv2); populate it once with ``python eval/mv2/parity.py``.

One measured reality is baked in below, not assumed (parity report holds the
exact numbers): the INT8 export quantizes activations with a per-TENSOR dynamic
scale, so a row's vector shifts when its batch mates (or padding) change the
tensor scale — cos 0.95–0.98 vs the same row embedded alone. Identical batches
and repeated calls are bit-exact. :data:`BATCH_MATE_COS_FLOOR` is an envelope
that fails loudly if a future artifact degrades beyond it; it is not a quality
gate (that is T2's ablation).
"""
from __future__ import annotations

import numpy as np
import pytest

from eval.mv2 import runner as mv2

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(
        not mv2.mv2_files_cached("int8", "tokenizer"),
        reason="m-v2 onnx cache missing — run eval/mv2/parity.py locally, do not download in CI",
    ),
]

BATCH_MATE_COS_FLOOR = 0.95
NORM_TOL = 1e-4
SLICE_TOL = 1e-4

TEXTS = [
    "Tôi đã gặp Huy ở quán cà phê gần hồ Tây vào chiều thứ Sáu.",
    "the app crashed when i tapped export after the update",
    "",
]


@pytest.fixture(scope="module")
def runner():
    return mv2.Mv2Onnx(mv2.model_dir() / mv2.INT8_FILE)


def _as_array(vectors) -> np.ndarray:
    return np.asarray(vectors, dtype=float)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_shape_and_finite_norms(runner):
    arr = _as_array(runner.embed_queries(TEXTS))
    assert arr.shape == (len(TEXTS), 768)
    assert np.isfinite(arr).all()
    norms = np.linalg.norm(arr, axis=1)
    assert np.allclose(norms, 1.0, atol=NORM_TOL), norms


def test_passages_and_queries_have_same_shape(runner):
    arr = _as_array(runner.embed_passages(TEXTS))
    assert arr.shape == (len(TEXTS), 768)
    assert np.isfinite(arr).all()


def test_determinism(runner):
    first = _as_array(runner.embed_queries(TEXTS))
    second = _as_array(runner.embed_queries(TEXTS))
    assert np.array_equal(first, second)


def test_identical_batch_mates_are_bit_exact(runner):
    dup = _as_array(runner.embed_passages([TEXTS[0], TEXTS[0]]))
    alone = _as_array(runner.embed_passages([TEXTS[0]]))[0]
    assert np.array_equal(dup[0], dup[1])
    assert np.array_equal(dup[0], alone)


def test_batch_mate_envelope(runner):
    """Row i of a mixed batch vs that row alone: cosine above the pinned floor.

    Fails loudly if the INT8 artifact ever drifts materially beyond the measured
    envelope; exact per-row drift/cosine values live in eval/mv2/parity_report.json.
    """
    batch = _as_array(runner.embed_passages(TEXTS))
    for row, text in enumerate(TEXTS):
        single = _as_array(runner.embed_passages([text]))[0]
        similarity = _cos(batch[row], single)
        assert similarity >= BATCH_MATE_COS_FLOOR, f"row {row!r} cosine {similarity}"
        assert np.isfinite(batch[row]).all()


def test_dim_256_is_renormalized_slice_of_768(runner):
    mrl = mv2.Mv2Onnx(mv2.model_dir() / mv2.INT8_FILE, dim=256)
    full = _as_array(runner.embed_passages(TEXTS))
    sliced = _as_array(mrl.embed_passages(TEXTS))
    assert sliced.shape == (len(TEXTS), 256)
    expected = full[:, :256] / np.linalg.norm(full[:, :256], axis=1, keepdims=True)
    assert np.allclose(sliced, expected, atol=SLICE_TOL)
    assert np.allclose(np.linalg.norm(sliced, axis=1), 1.0, atol=NORM_TOL)


def test_prefix_contract(runner):
    assert mv2.QUERY_PREFIX == "query: "
    for text in TEXTS:
        # Same single-row input either way, so this is bit-exact, not approximate.
        assert runner.embed_queries([text])[0] == runner.embed_passages(
            [mv2.QUERY_PREFIX + text]
        )[0]


def test_empty_batch_is_empty(runner):
    assert runner.embed_queries([]) == []
    assert runner.embed_passages([]) == []


def test_unknown_dim_and_pool_raise():
    path = mv2.model_dir() / mv2.INT8_FILE
    with pytest.raises(ValueError):
        mv2.Mv2Onnx(path, dim=128)
    with pytest.raises(ValueError):
        mv2.Mv2Onnx(path, pool="mean")
