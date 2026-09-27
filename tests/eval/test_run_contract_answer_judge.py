"""The run-artifact seam: one place that decides what a results file means.

Interface tests only. A per-key whitelist test is gone on purpose: the seam
compares every key that is not provenance, so these assert the RULE (a new
score-affecting key blocks a resume; provenance alone does not), not a list.
"""

import pytest

from eval.run_contract import (
    PROVENANCE_KEYS,
    recorded_context_policy,
    recorded_graph_builds,
    recorded_recall_top_k,
    run_contracts_match,
    run_verdict,
)

_CONTEXT_POLICY = {"session_level": True, "chunk_chars": 4000, "fuse": False}
_ANSWER = {
    "model": "stealth/space-bunny-alpha",
    "temperature": 0.0,
    "max_tokens": 2048,
    "prompt_version": "no-refusal-v1+date-hint",
    "passes_reference_date": True,
}
_JUDGE = {
    "model": "stealth/space-bunny-alpha",
    "prompt_version": "longmemeval-official-v1",
    "temperature": 0.0,
    "max_tokens": 8,
}


def _stack():
    return {
        "database": "sqlite (lite mode)",
        "vector_store": "qdrant local (in-process)",
        "embedding_backend": "local-arctic",
        "embeddings_actual": {"model_id": "Snowflake/snowflake-arctic-embed-xs", "dim": 384},
        "query_prefix": "query: ",
        "passage_prefix": "",
        "retriever": "MemoryRetriever",
        "dataset_path": "/some/checkout/eval/benchmarks/data/longmemeval_s_cleaned.json",
        "dataset_source": "worktree",
        "dataset_sha256": "test-dataset",
        "selected_question_ids": ["q1"],
        "sample_seed": 20260906,
        "git_head": "73ad85e",
        "git_dirty": False,
        "runtime": {"python": "3.13.1"},
        "timeouts_seconds": {"answer_and_judge_client": 240},
        "write_index_costs": {"memories_ingested": 12},
        "recall_top_k": 15,
        "rerank": {"enabled": True, "model": "local", "top_n": 15},
        "graph_builds": "off (bench ingest)",
        "retrieval": {"hybrid_enabled": False, "rerank_pool_multiplier": 2.0, "rrf_k": 60},
        "answer": dict(_ANSWER),
        "judge": dict(_JUDGE),
        "execution": {
            "requested_concurrency": 1,
            "actual_concurrency": 1,
            "thread_limits": {"OMP_NUM_THREADS": None},
            "warmup_performed": False,
            "cache_state": "not_reset; process/model/filesystem cache may be warm",
            "rewrite_policy": "personal-context plus pronoun policy; fast-path when eligible",
            "context_policy": dict(_CONTEXT_POLICY),
        },
    }


def _record(qid="q1", *, correct=True, recalled=1, error=None, qtype="single-session-user"):
    return {
        "question_id": qid,
        "question_type": qtype,
        "correct": correct,
        "memories_recalled": recalled,
        "error": error,
    }


# --- the compatibility rule itself ----------------------------------------


def test_an_unchanged_run_is_resumable():
    assert run_contracts_match(_stack(), _stack())


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("answer", "max_tokens", 300),
        ("answer", "prompt_version", "old-prompt-v0"),
        ("answer", "model", "some-other-model"),
        ("answer", "passes_reference_date", False),
        ("judge", "prompt_version", "judge-v0"),
        ("judge", "max_tokens", 32),
    ],
)
def test_a_changed_answer_or_judge_config_blocks_a_resume(section, key, value):
    changed = _stack()
    changed[section][key] = value

    assert not run_contracts_match(_stack(), changed)
    assert not run_contracts_match(changed, _stack())


def test_a_new_score_affecting_key_blocks_a_resume_without_being_listed():
    """Default-deny: a field nobody remembered to whitelist still stops the merge."""
    changed = _stack()
    changed["recall_pool_shape"] = "v2"

    assert not run_contracts_match(_stack(), changed)


@pytest.mark.parametrize(
    "key",
    [
        "git_head",
        "git_dirty",
        "runtime",
        "timeouts_seconds",
        "write_index_costs",
        "dataset_path",
        "dataset_source",
        "sample_seed",
    ],
)
def test_provenance_alone_never_blocks_a_resume(key):
    """Provenance is where the run HAPPENED, not what it MEASURED."""
    changed = _stack()
    changed[key] = {"python": "3.14.7", "note": "different machine"} if key == "runtime" else "elsewhere"

    assert run_contracts_match(_stack(), changed)


@pytest.mark.parametrize(
    "key", ["requested_concurrency", "actual_concurrency", "thread_limits", "warmup_performed",
            "cache_state", "rewrite_policy"]
)
def test_execution_provenance_alone_never_blocks_a_resume(key):
    changed = _stack()
    changed["execution"][key] = {"OMP_NUM_THREADS": "8"} if key == "thread_limits" else "changed"

    assert run_contracts_match(_stack(), changed)


def test_a_different_dataset_blocks_a_resume_though_its_path_does_not():
    """The dataset's PATH is provenance — the same bytes at another checkout
    resume fine. Its CONTENT is the measurement: scoring questions from a
    different file is not the run being resumed."""
    moved = _stack()
    moved["dataset_path"] = "/somewhere/else/longmemeval_s_cleaned.json"
    assert run_contracts_match(_stack(), moved)

    edited = _stack()
    edited["dataset_sha256"] = "0" * 64
    assert not run_contracts_match(_stack(), edited)


def test_a_context_policy_change_blocks_a_resume_and_a_missing_one_does_not_hide():
    changed = _stack()
    changed["execution"]["context_policy"] = dict(_CONTEXT_POLICY, chunk_chars=0)
    assert not run_contracts_match(_stack(), changed)

    missing = _stack()
    missing["execution"].pop("context_policy")
    assert not run_contracts_match(missing, _stack())


def test_malformed_or_absent_run_metadata_is_not_a_compatible_run():
    assert not run_contracts_match(None, _stack())
    assert not run_contracts_match(_stack(), "not-a-dict")


def test_a_run_policy_the_resume_path_cannot_honour_is_refused_on_either_side():
    """The resume path RE-APPLIES the recorded graph switch and context
    policy, so a policy it cannot read blocks the resume whichever side it is
    missing from. Same corrupt value on both sides still has to fail: equality
    alone would call two identical un-honourable policies compatible."""
    garbage = _stack()
    garbage["graph_builds"] = ["on"]  # a list where a policy belongs
    assert not run_contracts_match(garbage, _stack())
    assert not run_contracts_match(_stack(), garbage)
    assert not run_contracts_match(garbage, garbage)

    current_without_policy = _stack()
    del current_without_policy["execution"]["context_policy"]
    assert not run_contracts_match(_stack(), current_without_policy)
    assert not run_contracts_match(current_without_policy, current_without_policy)


# --- the recorded-policy readers the resume path applies ------------------


@pytest.mark.parametrize("policy", ["off (bench ingest)", "on"])
def test_recorded_graph_builds_returns_a_policy_the_resume_path_can_apply(policy):
    stack = _stack()
    stack["graph_builds"] = policy

    assert recorded_graph_builds(stack) == policy


@pytest.mark.parametrize("policy", [None, [], {}, "sometimes"])
def test_recorded_graph_builds_refuses_a_policy_it_does_not_understand(policy):
    stack = _stack()
    stack["graph_builds"] = policy

    assert recorded_graph_builds(stack) is None


def test_recorded_context_policy_returns_only_a_complete_typed_triple():
    assert recorded_context_policy(_stack()) == _CONTEXT_POLICY

    for broken in ({"session_level": True}, dict(_CONTEXT_POLICY, chunk_chars="4000"),
                   dict(_CONTEXT_POLICY, fuse=1), dict(_CONTEXT_POLICY, session_level=None)):
        stack = _stack()
        stack["execution"]["context_policy"] = broken
        assert recorded_context_policy(stack) is None


def test_recorded_recall_top_k_returns_a_positive_int():
    assert recorded_recall_top_k(_stack()) == 15

    for broken in (0, -1, "15", True, None):
        stack = _stack()
        stack["recall_top_k"] = broken
        assert recorded_recall_top_k(stack) is None


# --- the verdict: one place that says what a record set adds up to --------


def test_a_clean_run_is_complete_and_needs_no_re_run():
    records = [_record("q1"), _record("q2", correct=False)]

    verdict = run_verdict(records, expected_count=2, recorded_partial=False)

    assert verdict["complete"] is True
    assert verdict["failed"] == []
    assert verdict["totals"] == {
        "mean": 0.5,
        "questions": 2,
        "correct": 1,
        "errors": 0,
        "wilson_95": [0.095, 0.905],
        "by_type": {"single-session-user": {"n": 2, "correct": 1}},
    }


def test_a_fresh_run_and_a_resume_agree_on_the_same_records():
    """Two commands, one set of records, one number — or the artifact lies."""
    before = [_record("q1"), _record("q2", correct=False, recalled=0), _record("q3", error="boom")]
    after = [_record("q1"), _record("q2", correct=False), _record("q3", correct=True)]

    # The resume path saw the failed set from the pre-resume records...
    pending = run_verdict(before, expected_count=3, recorded_partial=True)
    # ...and the final artifact is scored by the SAME function, so the numbers
    # a resumed artifact reports equal a fresh run's over the same records.
    fresh = run_verdict(after, expected_count=3, recorded_partial=False)
    resumed = run_verdict(
        after,
        expected_count=3,
        recorded_partial=False,
    )

    assert pending["failed"] == [1, 2]  # recalled=0, then the errored record
    assert pending["complete"] is False
    assert fresh["failed"] == resumed["failed"] == []
    assert fresh["totals"] == resumed["totals"]
    assert fresh["comparison_delta"] == resumed["comparison_delta"]
    assert fresh["totals"]["mean"] == 0.667
    assert fresh["totals"]["errors"] == 0


def test_an_unrecorded_partial_state_leaves_the_last_record_unverified():
    """A run that died mid-flight never wrote its flag: the last row is suspect."""
    records = [_record("q1"), _record("q2")]

    assert run_verdict(records, recorded_partial=True)["failed"] == []
    assert run_verdict(records, recorded_partial=False)["failed"] == []
    # Only an artifact that predates the flag is ambiguous, and the resume path
    # tells the seam so explicitly rather than passing ``None`` by accident.
    assert run_verdict(records, recorded_partial=None)["failed"] == [1]
    assert run_verdict(records, recorded_partial=None)["complete"] is False


def test_a_record_that_errored_or_recalled_nothing_needs_a_re_run():
    records = [_record("q1", error="provider error"), _record("q2", recalled=0), _record("q3")]

    verdict = run_verdict(records, expected_count=3, recorded_partial=True)

    assert verdict["failed"] == [0, 1]
    assert verdict["complete"] is False
    # Only an ERROR is excluded from the mean; recalled=0 is a real, scored
    # answer (that is exactly what the resume path re-runs).
    assert verdict["totals"]["questions"] == 2
    assert verdict["totals"]["mean"] == 1.0
    assert verdict["totals"]["errors"] == 1


def test_a_corrupt_row_counts_as_an_error_instead_of_crashing_the_totals():
    """A record that is not even a dict is counted as unscoreable; the verdict
    must still produce totals rather than raise, because a half-written
    artifact is exactly when resume needs an answer."""
    verdict = run_verdict([_record("q1"), "half-written"], recorded_partial=False)

    assert verdict["totals"]["questions"] == 1
    assert verdict["totals"]["errors"] == 1


def test_a_short_or_contaminated_run_is_not_complete():
    records = [_record("q1")]

    assert run_verdict(records, expected_count=2, recorded_partial=False)["complete"] is False
    assert run_verdict(
        records, expected_count=1, recorded_partial=False, purge_failed=True
    )["complete"] is False
    assert run_verdict([], expected_count=0)["complete"] is True  # nothing to measure, nothing to hide


def test_forced_and_from_index_re_runs_narrow_the_same_failed_set():
    records = [_record("q1"), _record("q2")]

    assert run_verdict(records, recorded_partial=True, from_index=1)["failed"] == [1]
    assert run_verdict(records, recorded_partial=True, forced=[0])["failed"] == [0]
    assert run_verdict(records, recorded_partial=True, from_index=0)["failed"] == [0, 1]


def test_a_from_index_cursor_does_not_excuse_a_dropped_row_before_it():
    """--from-index means "from here on is unverified", not "everything before
    here is fine". A row that recalled nothing before the cursor still owes a
    re-run: quoting it would score a question the system could not answer."""
    records = [_record("q0", recalled=0), *(_record(f"q{i}") for i in range(1, 4))]

    assert run_verdict(records, recorded_partial=False)["failed"] == [0]
    assert run_verdict(records, recorded_partial=False, from_index=2)["failed"] == [0, 2, 3]


def test_the_baseline_delta_is_computed_once_from_the_verdict():
    records = [_record("q1"), _record("q2", correct=False)]

    assert run_verdict(records, recorded_partial=False, baseline_mean=0.0)["comparison_delta"] == 0.5
    assert run_verdict(records, recorded_partial=False, baseline_mean=0.5)["comparison_delta"] == 0.0
    assert run_verdict(records, baseline_mean=float("inf"))["comparison_delta"] is None
    assert run_verdict(
        [_record("q1", error="boom")], recorded_partial=True, baseline_mean=0.5
    )["comparison_delta"] is None
    assert run_verdict(
        records, recorded_partial=False, baseline_mean="n/a"
    )["comparison_delta"] is None
    # Nothing scored at all: 0 - baseline is a comparison against nothing.
    assert run_verdict([], recorded_partial=False, baseline_mean=0.5)["comparison_delta"] is None


def test_provenance_keys_are_dotted_paths_not_a_flat_blocklist():
    """The rule is 'ignore these exact paths', so a new top-level field is judged."""
    assert "runtime" in PROVENANCE_KEYS
    assert "execution.cache_state" in PROVENANCE_KEYS
    assert "answer" not in PROVENANCE_KEYS
    assert "recall_top_k" not in PROVENANCE_KEYS
