import json
from types import SimpleNamespace

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
        "graph_builds": "off (bench ingest)",
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


def test_resume_rejects_changed_graph_build_policy():
    recorded = _stack(dict(_CONTEXT_POLICY))
    current = _stack(dict(_CONTEXT_POLICY))
    current["graph_builds"] = "on"

    assert not retrieval_contract_matches(recorded, current)


def test_resume_rejects_missing_graph_build_policy():
    recorded = _stack(dict(_CONTEXT_POLICY))
    current = _stack(dict(_CONTEXT_POLICY))
    recorded.pop("graph_builds")
    current.pop("graph_builds")

    assert not retrieval_contract_matches(recorded, current)


@pytest.mark.parametrize("policy", [None, [], {}])
def test_resume_rejects_malformed_graph_build_policy(policy):
    recorded = _stack(dict(_CONTEXT_POLICY))
    current = _stack(dict(_CONTEXT_POLICY))
    current["graph_builds"] = policy

    assert not retrieval_contract_matches(recorded, current)


@pytest.mark.asyncio
@pytest.mark.parametrize("graph_builds", [None, [], {}])
async def test_resume_refuses_invalid_recorded_graph_build_policy(tmp_path, graph_builds):
    import eval.resume_system_run as resume

    stack = _stack(dict(_CONTEXT_POLICY))
    stack["graph_builds"] = graph_builds
    results = tmp_path / "run.json"
    results.write_text(json.dumps({"stack": stack, "per_question": []}))
    args = SimpleNamespace(
        results=results,
        session=None,
        chunk_chars=None,
        fuse=None,
        top_k=None,
        from_index=None,
        dry_run=True,
    )

    assert await resume.main_async(args) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(("graph_builds", "enabled"), [("off (bench ingest)", False), ("on", True)])
async def test_resume_applies_recorded_graph_build_policy(tmp_path, monkeypatch, graph_builds, enabled):
    import eval.resume_system_run as resume

    stack = _stack(dict(_CONTEXT_POLICY))
    stack["graph_builds"] = graph_builds
    results = tmp_path / "run.json"
    results.write_text(json.dumps({"stack": stack, "per_question": [{"question_id": "q1", "memories_recalled": 0}]}))
    monkeypatch.setattr(resume, "build_stack_metadata", lambda **_kwargs: stack)

    async def no_bootstrap():
        return None

    monkeypatch.setattr(resume, "bootstrap_sqlite", no_bootstrap)
    applied = []
    monkeypatch.setattr(resume, "_apply_graph_build_switch", applied.append)
    args = SimpleNamespace(
        results=results,
        session=None,
        chunk_chars=None,
        fuse=None,
        top_k=None,
        from_index=None,
        dry_run=True,
    )

    assert await resume.main_async(args) == 0
    assert applied == [enabled]
