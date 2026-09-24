import pytest

from eval.retrieval_contract import retrieval_contract_matches

_CONTEXT_POLICY = {"session_level": True, "chunk_chars": 4000, "fuse": False}
_RETRIEVAL_POLICY = {
    "hybrid_enabled": False,
    "rerank_pool_multiplier": 2.0,
    "rrf_k": 60,
}


def _stack(context_policy, retrieval_policy=None):
    return {
        "embedding_backend": "local-arctic",
        "embeddings_actual": {"model_id": "Snowflake/snowflake-arctic-embed-xs", "dim": 384},
        "rerank": {"enabled": True, "model": "local", "top_n": 15},
        "recall_top_k": 15,
        "retrieval": dict(_RETRIEVAL_POLICY if retrieval_policy is None else retrieval_policy),
        "execution": {"context_policy": context_policy},
    }


def test_resume_accepts_an_identical_retrieval_contract():
    stack = _stack(dict(_CONTEXT_POLICY))

    assert retrieval_contract_matches(stack, stack)


@pytest.mark.parametrize(
    "key,value",
    [("session_level", False), ("chunk_chars", 0), ("fuse", True)],
)
def test_resume_rejects_a_changed_execution_policy(key, value):
    recorded_policy = dict(_CONTEXT_POLICY)
    recorded_policy[key] = value

    assert not retrieval_contract_matches(_stack(recorded_policy), _stack(_CONTEXT_POLICY))


def test_resume_rejects_a_missing_execution_policy():
    recorded = _stack(dict(_CONTEXT_POLICY))
    recorded["execution"].pop("context_policy")

    assert not retrieval_contract_matches(recorded, _stack(_CONTEXT_POLICY))


@pytest.mark.parametrize(
    "key,value",
    [("hybrid_enabled", True), ("rerank_pool_multiplier", 3.0), ("rrf_k", 30)],
)
def test_resume_rejects_changed_retrieval_settings(key, value):
    recorded_policy = dict(_RETRIEVAL_POLICY)
    recorded_policy[key] = value

    assert not retrieval_contract_matches(_stack(dict(_CONTEXT_POLICY), recorded_policy), _stack(_CONTEXT_POLICY))


def test_resume_rejects_missing_retrieval_policy():
    recorded = _stack(dict(_CONTEXT_POLICY))
    current = _stack(dict(_CONTEXT_POLICY))
    recorded.pop("retrieval")
    current.pop("retrieval")

    assert not retrieval_contract_matches(recorded, current)
