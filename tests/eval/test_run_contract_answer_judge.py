"""The resume seam must reject a run whose answer/judge config changed."""

import pytest

from eval.run_contract import run_contracts_match

_CONTEXT_POLICY = {"session_level": True, "chunk_chars": 4000, "fuse": False}
_RETRIEVAL_POLICY = {
    "hybrid_enabled": False,
    "rerank_pool_multiplier": 2.0,
    "rrf_k": 60,
}
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


def _stack(context_policy=None, answer=None, judge=None):
    return {
        "embedding_backend": "local-arctic",
        "embeddings_actual": {"model_id": "Snowflake/snowflake-arctic-embed-xs", "dim": 384},
        "dataset_sha256": "test-dataset",
        "selected_question_ids": ["q1"],
        "rerank": {"enabled": True, "model": "local", "top_n": 15},
        "recall_top_k": 15,
        "graph_builds": "off (bench ingest)",
        "retrieval": dict(_RETRIEVAL_POLICY),
        "answer": dict(_ANSWER if answer is None else answer),
        "judge": dict(_JUDGE if judge is None else judge),
        "execution": {"context_policy": dict(_CONTEXT_POLICY if context_policy is None else context_policy)},
    }


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
def test_resume_rejects_a_changed_answer_or_judge_config(section, key, value):
    recorded = _stack()
    changed = _stack()
    changed[section][key] = value

    assert not run_contracts_match(recorded, changed)
    assert not run_contracts_match(changed, recorded)


def test_resume_accepts_an_identical_answer_and_judge_config():
    assert run_contracts_match(_stack(), _stack())


@pytest.mark.parametrize("section", ["answer", "judge"])
def test_resume_rejects_missing_answer_or_judge_metadata(section):
    recorded = _stack()
    del recorded[section]

    assert not run_contracts_match(recorded, _stack())


@pytest.mark.parametrize("section", ["answer", "judge"])
def test_resume_rejects_a_malformed_answer_or_judge_metadata(section):
    recorded = _stack()
    recorded[section] = "not-a-dict"
    current = _stack()
    current[section] = "not-a-dict"

    assert not run_contracts_match(recorded, current)
