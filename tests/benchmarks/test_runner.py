"""Runner: aggregation, sha256 hygiene, honest dataset errors, CLI plan phase, polarity."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.benchmarks.runner import (
    RunnerConfig,
    aggregate,
    interpret_result,
    record_sha256,
    run_ingest,
    run_score,
)

FIXTURES = Path(__file__).resolve().parents[2] / "eval" / "benchmarks" / "fixtures"
LONGMEMEVAL_FIXTURE = FIXTURES / "longmemeval_s_fixture.json"
MAB_FIXTURE = FIXTURES / "memoryagentbench_fixture.json"


async def _no_call(session) -> int:  # pragma: no cover - must never be reached
    raise AssertionError("ingest callable must not be called when the dataset is missing")


pytestmark = pytest.mark.eval


# --- aggregate -----------------------------------------------------------


def test_aggregate_three_results_mean_and_questions():
    results = [
        {"question_id": "q1", "correct": True},
        {"question_id": "q2", "correct": True},
        {"question_id": "q3", "correct": False},
    ]
    summary = aggregate(results)
    assert summary["mean"] == pytest.approx(2 / 3)
    assert summary["questions"] == 3


def test_aggregate_empty_results_mean_zero():
    summary = aggregate([])
    assert summary == {"mean": 0.0, "questions": 0}


def test_aggregate_ignores_non_bool_correct_values():
    """A missing/malformed 'correct' flag never counts as a fabricated pass."""
    summary = aggregate(
        [
            {"question_id": "q1", "correct": True},
            {"question_id": "q2", "correct": None},
        ]
    )
    assert summary["mean"] == pytest.approx(1 / 2)
    assert summary["questions"] == 2


# --- record_sha256 -------------------------------------------------------


def test_record_sha256_is_deterministic_and_hex():
    digest = record_sha256(LONGMEMEVAL_FIXTURE)
    assert digest == record_sha256(LONGMEMEVAL_FIXTURE)
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


def test_record_sha256_matches_known_value():
    import hashlib

    expected = hashlib.sha256(LONGMEMEVAL_FIXTURE.read_bytes()).hexdigest()
    assert record_sha256(LONGMEMEVAL_FIXTURE) == expected


# --- dataset-not-found ---------------------------------------------------


def test_missing_dataset_raises_runtime_error_with_readme_pointer(tmp_path):
    config = RunnerConfig(
        benchmark="longmemeval_s",
        dataset_path=tmp_path / "missing.json",
        output_dir=tmp_path / "out",
    )
    with pytest.raises(RuntimeError, match=r"dataset not found — download per eval/benchmarks/README\.md"):
        record_sha256(config.dataset_path)


def test_missing_dataset_message_mentions_readme(tmp_path):
    config = RunnerConfig(
        benchmark="memoryagentbench",
        dataset_path=tmp_path / "nope.json",
        output_dir=tmp_path / "out",
    )
    with pytest.raises(RuntimeError) as excinfo:
        record_sha256(config.dataset_path)
    assert "eval/benchmarks/README.md" in str(excinfo.value)


# --- run_ingest / run_score (injected callables, no live stack) ----------


def test_config_fields_and_defaults(tmp_path):
    config = RunnerConfig(
        benchmark="longmemeval_s", dataset_path=LONGMEMEVAL_FIXTURE, output_dir=tmp_path
    )
    assert config.limit is None
    assert config.benchmark == "longmemeval_s"


async def test_run_ingest_counts_sessions_via_injected_history(tmp_path):
    from eval.benchmarks.longmemeval_s import load_instances

    config = RunnerConfig(
        benchmark="longmemeval_s", dataset_path=LONGMEMEVAL_FIXTURE, output_dir=tmp_path
    )
    instances = load_instances(LONGMEMEVAL_FIXTURE)
    ingested_sessions: list[str] = []

    async def ingest_history(session) -> int:
        # A real caller ingests each turn via a live create_memory; the runner
        # only sums what the injected callable actually reports.
        ingested_sessions.append(session.session_id)
        return len(session.turns)

    count = await run_ingest(config, instances, ingest_history)
    # 3 sessions × 4 turns + 2 sessions × 4 turns = 20 turns total.
    assert count == 20
    assert ingested_sessions == ["s_2026_05_02", "s_2026_06_14", "s_2026_07_30",
                                  "s_2026_04_11", "s_2026_07_05"]


async def test_run_ingest_respects_limit(tmp_path):
    from eval.benchmarks.longmemeval_s import load_instances

    config = RunnerConfig(
        benchmark="longmemeval_s", dataset_path=LONGMEMEVAL_FIXTURE, output_dir=tmp_path, limit=1
    )
    instances = load_instances(LONGMEMEVAL_FIXTURE)

    async def ingest_history(session) -> int:
        return len(session.turns)

    count = await run_ingest(config, instances, ingest_history)
    # Only the first instance (3 sessions × 4 turns).
    assert count == 12


async def test_run_ingest_handles_memoryagentbench_scenarios(tmp_path):
    from eval.benchmarks.memoryagentbench import load_instances as load_scenarios

    config = RunnerConfig(
        benchmark="memoryagentbench", dataset_path=MAB_FIXTURE, output_dir=tmp_path
    )
    scenarios = load_scenarios(MAB_FIXTURE)

    async def ingest_history(session) -> int:
        return len(session.turns)

    count = await run_ingest(config, scenarios, ingest_history)
    # 4 scenarios; lru has 2 sessions × 4 turns, others 1 session × 4 turns.
    assert count == 4 * 4 + 4


async def test_run_ingest_missing_dataset_raises(tmp_path):
    config = RunnerConfig(
        benchmark="longmemeval_s", dataset_path=tmp_path / "missing.json", output_dir=tmp_path
    )
    with pytest.raises(RuntimeError, match="dataset not found"):
        await run_ingest(config, [], _no_call)


async def test_run_score_aggregates_injected_results(tmp_path):
    config = RunnerConfig(
        benchmark="longmemeval_s", dataset_path=LONGMEMEVAL_FIXTURE, output_dir=tmp_path
    )
    results = [
        {"question_id": "q1", "correct": True},
        {"question_id": "q2", "correct": False},
    ]
    summary = await run_score(config, results)
    assert summary["mean"] == pytest.approx(0.5)
    assert summary["questions"] == 2
    # Hygiene rule: the dataset digest travels with the score.
    assert "dataset_sha256" in summary
    assert len(summary["dataset_sha256"]) == 64


def test_write_results_writes_json_payload(tmp_path):
    from eval.benchmarks.runner import write_results

    out = tmp_path / "nested" / "results.json"
    write_results(out, {"mean": 0.5, "questions": 2})
    assert json.loads(out.read_text()) == {"mean": 0.5, "questions": 2}


async def test_run_score_output_carries_hygiene_placeholders(tmp_path):
    """Hygiene rules 1/4/5 surface as explicit placeholders, never silently omitted."""
    config = RunnerConfig(
        benchmark="longmemeval_s", dataset_path=LONGMEMEVAL_FIXTURE, output_dir=tmp_path
    )
    summary = await run_score(config, [{"question_id": "q1", "correct": True}])
    assert summary["judge_prompt_version"] is None
    assert summary["full_context_baseline"] is None
    assert summary["deviations"] == []


# --- interpret_result: the polarity handoff, pinned ----------------------


def test_selective_forgetting_fixture_polarity_true_is_pass():
    """CRITICAL: THIS fixture's gold is the SURVIVING updated fact.

    recalled_after_forget=True means memory kept the update while forgetting
    the stale fact — correct behavior. A dedicated forgetting-question dataset
    would invert this mapping; do not hardcode True=BAD.
    """
    assert interpret_result("selective_forgetting", True) == "pass"
    assert interpret_result("selective_forgetting", False) == "fail"


def test_selective_forgetting_with_the_stale_fact_still_present_is_not_pass():
    """Both the updated fact AND the verbatim forgotten fact in one response.

    The adapter flags it (stale_fact_still_present=True — pinned in
    test_memoryagentbench); scoring that response 'pass' would read a failed
    forget as a success.
    """
    assert (
        interpret_result("selective_forgetting", True, stale_fact_still_present=True)
        == "fail"
    )
    assert (
        interpret_result("selective_forgetting", True, stale_fact_still_present=False)
        == "pass"
    )


async def test_run_score_stale_fact_leak_never_counts_as_pass(tmp_path):
    config = RunnerConfig(
        benchmark="memoryagentbench", dataset_path=MAB_FIXTURE, output_dir=tmp_path
    )
    results = [
        {
            "question_id": "fixture_sf_0001",
            "competency": "selective_forgetting",
            "recalled_after_forget": True,
            "stale_fact_still_present": True,
        },
        {
            "question_id": "fixture_sf_0002",
            "competency": "selective_forgetting",
            "recalled_after_forget": True,
            "stale_fact_still_present": False,
        },
    ]
    summary = await run_score(config, results)
    assert summary["questions"] == 2
    assert summary["mean"] == pytest.approx(0.5)


def test_other_competencies_are_pending_interpretation():
    for competency in (
        "accurate_retrieval",
        "test_time_learning",
        "long_range_understanding",
    ):
        assert interpret_result(competency, True) == "pending_interpretation"
        assert interpret_result(competency, False) == "pending_interpretation"


def test_unknown_competency_is_pending_interpretation():
    assert interpret_result("some_future_competency", True) == "pending_interpretation"


def test_polarity_is_per_competency_not_global():
    """The runner must never interpret True/False globally."""
    mappings = {
        c: (interpret_result(c, True), interpret_result(c, False))
        for c in ("selective_forgetting", "accurate_retrieval")
    }
    assert mappings["selective_forgetting"] == ("pass", "fail")
    assert mappings["accurate_retrieval"] == (
        "pending_interpretation",
        "pending_interpretation",
    )
