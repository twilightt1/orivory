"""The combined report's answer-quality verdict must come from JUDGED cases.

``generate_full_report`` read the blended ``avg_overall_score``, which mixes
heuristic-fallback rows (constants that reach ~0.9) in with judged ones — so a
run where EVERY judge call failed over to heuristics (``llm_cases == 0``)
still printed "✅ Excellent" for answer quality.
"""
from __future__ import annotations

import json

import pytest

from eval.run_full_eval import generate_full_report

pytestmark = pytest.mark.eval

OFFLINE = {"source_hit_rate": 0.95, "keyword_coverage": 0.9, "fallback_accuracy": 1.0}


def _judge_summary(**kw):
    base = {
        "total_cases": 3,
        "heuristic_cases": 3,
        "llm_cases": 0,
        "avg_overall_score_llm": 0.0,
        "avg_overall_score": 0.92,  # blended heuristic constants
        "avg_reasoning_quality": 0.9,
        "by_difficulty": {},
    }
    base.update(kw)
    return base


def test_heuristic_only_summary_cannot_earn_an_excellent_verdict(tmp_path):
    report = generate_full_report(OFFLINE, _judge_summary(), tmp_path)

    assert "Excellent" not in report["answer_quality_status"]
    assert report["answer_quality_score_source"] == "heuristic_fallback"
    # Nothing was judged, so the answer-quality weight is not invented from
    # heuristic constants: only the retrieval weight (0.3) is real.
    assert report["combined_score"] == pytest.approx(0.3)
    # The report on disk carries the same verdict.
    written = json.loads((tmp_path / "combined_report.json").read_text())
    assert written["answer_quality_status"] == report["answer_quality_status"]
    assert written["answer_quality_score_source"] == "heuristic_fallback"


def test_verdict_comes_from_the_llm_only_score_and_says_so(tmp_path):
    # Blended 0.92 (Excellent-looking) but judged cases average 0.4.
    summary = _judge_summary(
        heuristic_cases=2, llm_cases=1, avg_overall_score_llm=0.4, avg_overall_score=0.92
    )
    report = generate_full_report(OFFLINE, summary, tmp_path)

    assert report["answer_quality_score_source"] == "llm_judge"
    assert report["answer_quality_score"] == pytest.approx(0.4)
    assert report["answer_quality_status"] == "❌ Poor"
    assert report["combined_score"] == pytest.approx(0.4)


def test_fully_judged_excellent_run_still_reads_excellent(tmp_path):
    summary = _judge_summary(heuristic_cases=0, llm_cases=3, avg_overall_score_llm=0.9)
    report = generate_full_report(OFFLINE, summary, tmp_path)

    assert report["answer_quality_status"] == "✅ Excellent"
    assert report["answer_quality_score_source"] == "llm_judge"
    assert report["combined_score"] == pytest.approx(0.7)
