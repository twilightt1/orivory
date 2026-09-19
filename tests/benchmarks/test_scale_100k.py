"""CI-safe contracts for the 100K milestone: split math, artifact naming, filtered recall.

Nothing here runs the heavy milestone — the 100K artifact is committed and the
harness's own tiny check is ``filtered_ann.py --self-check`` (local Qdrant,
random vectors, no model needed), which one test below drives as a subprocess.
"""
from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "eval" / "scale" / "run_10k.py"
FILTERED = REPO_ROOT / "eval" / "scale" / "filtered_ann.py"

pytestmark = pytest.mark.eval


# ── corpus split math (gen_corpus) ──────────────────────────────────────────


def test_split_counts_100k_does_not_saturate_the_real_pool():
    """The recorded arithmetic behind the 100K real/synthetic split.

    100K rows at the default 0.6 share asks for 60,000 real turns out of a
    ~120K-turn pool: the task's \"the real part saturates the dataset\" premise
    is FALSE at this milestone and the artifact must say so (saturation starts
    around 200K rows at this share, or 120K rows at share 1.0 — which would
    evict the synthetic part).
    """
    from eval.scale.gen_corpus import split_counts

    split = split_counts(rows=100_000, real_share=0.6, pool_size=120_052)
    assert split["requested_real"] == 60_000
    assert split["selected_real"] == 60_000
    assert split["selected_synthetic"] == 40_000
    assert split["pool_saturated"] is False
    assert split["pool_share_used"] == pytest.approx(0.4998, abs=5e-4)


def test_split_counts_records_saturation_and_the_synthetic_shift():
    """At 1M rows the pool IS the ceiling: real drops to the pool size."""
    from eval.scale.gen_corpus import split_counts

    split = split_counts(rows=1_000_000, real_share=0.6, pool_size=120_052)
    assert split["requested_real"] == 600_000
    assert split["selected_real"] == 120_052
    assert split["selected_synthetic"] == 879_948
    assert split["pool_saturated"] is True
    assert split["pool_share_used"] == 1.0


def test_manifest_records_pool_usage(tmp_path):
    """The manifest carries the pool arithmetic, not just the counts."""
    from eval.scale.gen_corpus import generate

    fixture = REPO_ROOT / "eval" / "benchmarks" / "fixtures" / "longmemeval_s_fixture.json"
    manifest = generate(rows=50, seed=7, dataset=fixture, out=tmp_path / "c.jsonl",
                        real_share=0.6)
    dataset = manifest["dataset"]
    pool = dataset["pool_user_turns"]
    assert dataset["selected_real"] == manifest["parts"]["real"]["count"]
    assert dataset["pool_share_used"] == pytest.approx(dataset["selected_real"] / pool)
    assert dataset["pool_saturated"] == (dataset["selected_real"] >= pool)
    assert dataset["requested_real"] >= dataset["selected_real"]


# ── 100K-runner plumbing (run_10k is the milestone harness) ─────────────────


def test_artifact_path_names_the_milestone():
    from eval.scale.run_10k import default_artifact_path, milestone_label

    assert milestone_label(10_000) == "10k"
    assert milestone_label(100_000) == "100k"
    assert milestone_label(1_000_000) == "1m"
    assert milestone_label(300) == "300"
    assert default_artifact_path("daf2f4126e38939e", 100_000).name == "100k_daf2f4126e38.json"


def test_runner_exposes_rows_and_corpus_flags():
    """The 100K invocation is the 10K one with --rows: pin that surface."""
    proc = subprocess.run([sys.executable, str(RUNNER), "--help"],
                          capture_output=True, text=True, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stderr[-2000:]
    flat = " ".join(proc.stdout.split())  # argparse wraps at the terminal width
    assert "--rows" in flat and "--corpus" in flat and "--reuse" in flat
    assert "10000 (default)" in flat


# ── filtered-recall math (filtered_ann) ─────────────────────────────────────


def test_recall_at_k_is_intersection_over_k():
    from eval.scale.filtered_ann import recall_at_k

    assert recall_at_k(["a", "b", "c"], ["a", "b", "c"], 3) == 1.0
    assert recall_at_k(["a", "x", "y"], ["a", "b", "c"], 3) == pytest.approx(1 / 3)
    assert recall_at_k(["d", "e", "f"], ["a", "b", "c"], 3) == 0.0
    assert recall_at_k([], ["a"], 1) == 0.0
    assert recall_at_k(["a", "b"], ["a", "b"], 0) == 0.0
    # Only the first k of each list counts: a longer ANN list cannot inflate it.
    assert recall_at_k(["a", "x", "y", "z"], ["a", "b"], 2) == pytest.approx(0.5)


def test_exact_top_k_is_cosine_ordered_and_id_tie_broken():
    from eval.scale.filtered_ann import exact_top_k

    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    ids = ["c", "b", "a"]
    query = np.array([2.0, 0.0], dtype=np.float32)  # unnormalized on purpose
    assert exact_top_k(vectors, query, 2, ids) == ["a", "c"]  # equal cosine -> id order
    assert exact_top_k(vectors, query, 3, ids) == ["a", "c", "b"]


def test_selectivity_window_start_scales_with_the_fraction():
    from eval.scale.filtered_ann import window_start

    last = datetime(2026, 9, 1, 9, 0, tzinfo=UTC) + timedelta(seconds=6_000_000)
    span = 6_000_000.0
    assert window_start(last, span, 1.0) == last - timedelta(seconds=span)
    assert window_start(last, span, 0.001) == last - timedelta(seconds=6_000)
    assert window_start(last, span, 0.5) == last - timedelta(seconds=3_000_000)
    # Payload stamps read back through SQLite are naive: the bound must come
    # back tz-aware UTC (the app's filter builder refuses naive bounds).
    naive = window_start(last.replace(tzinfo=None), span, 0.1)
    assert naive.tzinfo is not None and naive.utcoffset() == timedelta(0)
    assert naive == last - timedelta(seconds=600_000)
    with pytest.raises(ValueError):
        window_start(last, span, 0.0)
    with pytest.raises(ValueError):
        window_start(last, span, 1.5)


# ── the harness's own tiny check (local Qdrant, random vectors, no model) ───


def test_filtered_ann_self_check_passes(tmp_path):
    """Runs the real code path on 64 points: exact store in, exact store out."""
    proc = subprocess.run(
        [sys.executable, str(FILTERED), "--self-check", "--workdir", str(tmp_path / "sc")],
        capture_output=True, text=True, timeout=300, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    assert "self-check: OK" in proc.stdout
