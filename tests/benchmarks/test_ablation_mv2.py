"""m-v2 ablation (T2): the SIGNED gate function, slice plumbing, tiny real run.

The gate the §12.2 budgets signed (2026-09-19) is pure arithmetic on per-slice
recall@5 deltas — so its tests need no model, no cache and no network, and they
are the FIRST thing written (RED before :func:`evaluate_gate` existed). The
fixture-plumbing tests pin the ablation to the p2 ablation's frozen fixture by
hash. The last test is a skipif-guarded tiny REAL run through the cached
INT8/FP32 ONNX exports; it never downloads (house pattern,
tests/benchmarks/test_mv2_runner.py).

Nothing here flips a flag: the ablation only reports eligibility, and these
tests only check the report's arithmetic.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eval import ablation_retrieval_mv2 as abl
from eval.mv2 import runner as mv2

pytestmark = [pytest.mark.eval]

P2_ARTIFACT = Path(abl.__file__).resolve().parent / "ablation_retrieval_p2.json"
MAX_DROP = 0.02
MIN_VI_GAIN = 0.02


def _deltas(**overrides: float) -> dict[str, float]:
    """A passing baseline: no slice moves, both VI slices clear the gain."""
    baseline = {name: 0.0 for name in abl.SLICES}
    baseline[abl.SLICE_VI] = MIN_VI_GAIN
    baseline[abl.SLICE_VI_NODIAC] = MIN_VI_GAIN
    baseline.update(overrides)
    return baseline


# ── the signed gate (no model needed) ────────────────────────────────────────

def test_gate_passes_when_every_slice_non_inferior_and_vi_superior():
    verdict = abl.evaluate_gate(_deltas(**{abl.SLICE_VI: 0.08, abl.SLICE_VI_NODIAC: 0.06}))
    assert verdict["eligible"] is True
    assert verdict["non_inferior"]["passed"] is True
    assert verdict["non_inferior"]["failing_slices"] == []
    assert verdict["vi_superiority"]["passed"] is True
    assert verdict["vi_superiority"]["failing_slices"] == []


def test_gate_allows_a_slice_drop_of_exactly_the_signed_threshold():
    verdict = abl.evaluate_gate(_deltas(**{abl.SLICE_EN: -MAX_DROP}))
    assert verdict["eligible"] is True
    assert verdict["non_inferior"]["failing_slices"] == []


def test_gate_fails_when_any_slice_drops_past_the_threshold():
    verdict = abl.evaluate_gate(_deltas(**{abl.SLICE_EN: -0.0201}))
    assert verdict["eligible"] is False
    assert verdict["non_inferior"]["failing_slices"] == [abl.SLICE_EN]
    assert abl.SLICE_EN in verdict["reason"]


def test_gate_requires_the_vi_gain_at_the_signed_threshold():
    just_under = abl.evaluate_gate(
        _deltas(**{abl.SLICE_VI: MIN_VI_GAIN - 1e-4, abl.SLICE_VI_NODIAC: MIN_VI_GAIN - 1e-4})
    )
    assert just_under["eligible"] is False
    assert just_under["vi_superiority"]["passed"] is False
    assert set(just_under["vi_superiority"]["failing_slices"]) == {
        abl.SLICE_VI,
        abl.SLICE_VI_NODIAC,
    }

    at_threshold = abl.evaluate_gate(_deltas(**{abl.SLICE_VI: MIN_VI_GAIN}))
    assert at_threshold["eligible"] is True


def test_gate_fails_when_the_two_vi_slices_disagree_and_names_the_slice():
    verdict = abl.evaluate_gate(_deltas(**{abl.SLICE_VI: 0.05, abl.SLICE_VI_NODIAC: 0.01}))
    assert verdict["eligible"] is False
    assert verdict["vi_superiority"]["failing_slices"] == [abl.SLICE_VI_NODIAC]
    assert abl.SLICE_VI_NODIAC in verdict["reason"]
    assert verdict["vi_superiority"]["gains"][abl.SLICE_VI] == pytest.approx(0.05)


def test_gate_reports_both_conditions_independently():
    """A VI win never buys back a slice loss, and vice versa (the signed AND)."""
    losing = abl.evaluate_gate(_deltas(**{abl.SLICE_LONG: -0.5, abl.SLICE_VI: 0.5}))
    assert losing["eligible"] is False
    assert losing["non_inferior"]["passed"] is False
    assert losing["vi_superiority"]["passed"] is True

    flat = abl.evaluate_gate({name: 0.0 for name in abl.SLICES})
    assert flat["eligible"] is False
    assert flat["non_inferior"]["passed"] is True
    assert flat["vi_superiority"]["passed"] is False


def test_gate_rejects_incomplete_slice_maps():
    deltas = _deltas()
    del deltas[abl.SLICE_VI_NODIAC]
    with pytest.raises(ValueError, match=abl.SLICE_VI_NODIAC):
        abl.evaluate_gate(deltas)


# ── fixture plumbing: the p2 frozen fixture, byte for byte ───────────────────

def test_fixture_matches_the_p2_artifact_hashes():
    artifact = json.loads(P2_ARTIFACT.read_text(encoding="utf-8"))
    fixture = abl.build_fixture()
    assert abl.corpus_hash(fixture) == artifact["fixture"]["corpus_hash"]
    assert abl.query_set_hash(fixture) == artifact["fixture"]["query_set_hash"]


def test_fixture_slices_and_golds_resolve():
    fixture = abl.build_fixture()
    assert {query["slice"] for query in fixture["queries"]} == set(abl.SLICES)
    gold_keys = {row["key"]: row for row in fixture["rows"]}
    for query in fixture["queries"]:
        assert query["golds"], query["key"]
        for key in query["golds"]:
            assert key in gold_keys, f"{query['key']} -> missing gold {key}"
            # the no-diacritics slice re-uses the VI rows; every other slice's
            # golds are rows of that same slice.
            if query["slice"] != abl.SLICE_VI_NODIAC:
                assert gold_keys[key]["slice"] == query["slice"], query["key"]
    assert len(fixture["queries"]) == sum(
        json.loads(P2_ARTIFACT.read_text(encoding="utf-8"))["fixture"]["queries_per_slice"].values()
    )


def test_exact_cosine_topk_filters_tenant_sorts_and_ties_on_id():
    query = np.array([1.0, 0.0])
    docs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    ids = ["b", "a", "c", "d"]
    tenants = np.array(["A", "A", "B", "A"])
    visible = np.array([True, True, True, False])  # "d" is superseded
    got = abl.exact_cosine_topk(
        query, docs, ids, tenants, visible=visible, user_id="A", top_k=5
    )
    assert got == ["a", "b"]  # tie on score -> id ascending; B and superseded rows out


def test_corpus_documents_mirror_production_shape():
    fixture = abl.build_fixture()
    documents = abl.corpus_documents(fixture["rows"])
    assert len(documents) == len(fixture["rows"])
    titled = next(row for row in fixture["rows"] if row["title"])
    assert documents[fixture["rows"].index(titled)].startswith(f"Title: {titled['title']}\n")


# ── the tiny REAL run (cached artifacts only; never downloads) ───────────────

_ONNX_READY = False
if mv2.mv2_files_cached("int8", "fp32", "tokenizer"):
    try:
        abl._verify_cached_digests({
            mv2.model_dir() / mv2.INT8_FILE: mv2.INT8_SHA256,
            mv2.model_dir() / mv2.FP32_FILE: mv2.FP32_SHA256,
            mv2.model_dir() / mv2.TOKENIZER_FILE: mv2.TOKENIZER_SHA256,
        })
        _ONNX_READY = True
    except RuntimeError:
        _ONNX_READY = False  # cached but substituted/stale: skip, never download
TINY_TEXTS = [
    "Tôi đã gặp Huy ở quán cà phê gần hồ Tây vào chiều thứ Sáu.",
    "the app crashed when i tapped export after the update",
    "ORIVORY-4417",
]


@pytest.mark.skipif(
    not _ONNX_READY,
    reason="m-v2 onnx cache missing — run eval/mv2/parity.py locally, do not download in CI",
)
def test_tiny_real_run_int8_solo_fp32_batch():
    """The arms' own batch policies, on 3 real texts, through the real exports.

    INT8 arms embed ONE text per call (T1 finding: per-tensor dynamic
    quantization makes a row's vector depend on its batch mates) — so a solo
    call and the same text inside a stated batch must be BIT-EXACT here. FP32
    batches (float reassociation only) and must stay within the parity
    harness's tolerance against the INT8 arm's vectors.
    """
    int8 = abl.Mv2SoloText(mv2.model_dir() / mv2.INT8_FILE, dim=768)
    fp32 = mv2.Mv2Onnx(mv2.model_dir() / mv2.FP32_FILE, dim=768)

    batch = np.asarray(int8.embed_passages(TINY_TEXTS), dtype=float)
    assert batch.shape == (len(TINY_TEXTS), 768)
    assert np.isfinite(batch).all()
    assert np.allclose(np.linalg.norm(batch, axis=1), 1.0, atol=1e-4)

    for row, text in enumerate(TINY_TEXTS):
        solo = np.asarray(int8.embed_passages([text]), dtype=float)[0]
        assert np.array_equal(batch[row], solo), f"row {row!r} moved with its batch mates"

    reference = np.asarray(fp32.embed_passages(TINY_TEXTS), dtype=float)
    cosine = [float(np.dot(batch[i], reference[i])) for i in range(len(TINY_TEXTS))]
    assert min(cosine) >= 0.9, cosine

    mrl = abl.Mv2SoloText(mv2.model_dir() / mv2.INT8_FILE, dim=256)
    sliced = np.asarray(mrl.embed_passages(TINY_TEXTS), dtype=float)
    assert sliced.shape == (len(TINY_TEXTS), 256)
    assert np.allclose(np.linalg.norm(sliced, axis=1), 1.0, atol=1e-4)


def test_gate_rejects_non_finite_deltas():
    """NaN comparisons are False, so a NaN delta would pass as 'not failing'."""
    clean = {name: 0.0 for name in abl.SLICES}
    clean[abl.VI_PRIMARY] = 0.02      # ≥ +0.02 on both VI slices -> eligible
    clean[abl.VI_SECONDARY] = 0.02
    verdict = abl.evaluate_gate(clean)
    assert verdict["eligible"] is True, verdict  # the clean map is eligible by construction

    for bad in (float("nan"), float("inf"), float("-inf")):
        poisoned = dict(clean)
        poisoned[abl.SLICES[0]] = bad
        with pytest.raises(ValueError):
            abl.evaluate_gate(poisoned)
