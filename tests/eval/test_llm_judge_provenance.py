"""Provenance honesty for eval/llm_judge.py (found in full-repo review, HIGH).

`evaluate_case_offline` assigned `reasoning_quality` from a constant keyed
on the case's difficulty LABEL (easy=0.85 regardless of the answer), and
`run_llm_judge_evaluation` fell back to these heuristics on ANY exception —
then `summarize_judge_results` blended both provenances into headline
aggregates with no trace of how many cases were heuristic. A published
report could present fabricated constants as judged quality.

Rules enforced here:
1. Every result carries `judged_by` ("llm" | "heuristic").
2. Offline scores never depend on the difficulty label.
3. Summaries report the heuristic count and split LLM-only aggregates.
"""
from __future__ import annotations

import pytest

from eval.llm_judge import (
    CaseJudgeResult,
    evaluate_case_offline,
    run_llm_judge_evaluation,
    summarize_judge_results,
)


def _case(difficulty="easy", **kw):
    base = {
        "id": "c1",
        "query": "What did we decide about pgvector?",
        "difficulty": difficulty,
        "reasoning_type": "general",
        "expected_keywords": ["pgvector"],
    }
    base.update(kw)
    return base


def test_offline_result_is_tagged_heuristic():
    res = evaluate_case_offline(_case(), "We chose pgvector for vectors.", "pgvector context")
    assert res.judged_by == "heuristic"


def test_offline_scores_ignore_difficulty_label():
    """Same answer+context must score the same whatever the label says —
    label-derived constants are fabricated scores, not measurements."""
    answer = "We chose pgvector because it is simple."
    easy = evaluate_case_offline(_case(difficulty="easy"), answer, "pgvector context")
    hard = evaluate_case_offline(_case(difficulty="hard"), answer, "pgvector context")
    assert easy.reasoning_quality == hard.reasoning_quality
    assert easy.overall_score == hard.overall_score


def _judged(case_id: str, score: float) -> CaseJudgeResult:
    return CaseJudgeResult(
        case_id=case_id,
        query="q",
        answer="a",
        context="c",
        reasoning_type="general",
        difficulty="medium",
        faithfulness=score,
        answer_relevancy=score,
        reasoning_quality=score,
        overall_score=score,
        judge_reasoning="llm",
        errors=[],
        suggestions=[],
        judged_by="llm",
    )


def test_summary_splits_heuristic_cases():
    llm_ok = _judged("l1", 0.9)
    llm_bad = _judged("l2", 0.2)
    heur = evaluate_case_offline(_case(), "", "")
    assert heur.judged_by == "heuristic"

    summary = summarize_judge_results([llm_ok, llm_bad, heur])
    assert summary["total_cases"] == 3
    assert summary["heuristic_cases"] == 1
    assert summary["llm_cases"] == 2
    # LLM-only aggregates must ignore the heuristic row.
    assert summary["avg_overall_score_llm"] == pytest.approx((0.9 + 0.2) / 2)
    assert summary["pass_rate_llm"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_run_fallback_path_tags_heuristic(monkeypatch):
    """When the LLM call blows up, the fallback result must say so."""
    import eval.llm_judge as judge_mod

    async def _boom(**kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(judge_mod, "evaluate_with_llm_judge", _boom)
    results = await run_llm_judge_evaluation([_case()], {"c1": "a"}, {"c1": "c"})
    assert len(results) == 1
    assert results[0].judged_by == "heuristic"
