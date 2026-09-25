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
        "dataset_sha256": "test-dataset",
        "selected_question_ids": ["q1"],
        "rerank": {"enabled": True, "model": "local", "top_n": 15},
        "recall_top_k": 15,
        "graph_builds": "off (bench ingest)",
        "retrieval": dict(_RETRIEVAL_POLICY if retrieval_policy is None else retrieval_policy),
        "execution": {"context_policy": context_policy},
    }


@pytest.fixture(autouse=True)
def _set_test_benchmark_model(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "test-model")


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
    results.write_text(
        json.dumps(
            {
                "stack": stack,
                "sample": {"n": 1, "question_ids": ["q1"]},
                "per_question": [{
                    "question_id": "q1",
                    "question_type": "single-session-user",
                    "correct": False,
                    "memories_recalled": 0,
                    "error": None,
                }],
            }
        )
    )
    monkeypatch.setattr(resume, "build_stack_metadata", lambda **_kwargs: stack)
    monkeypatch.setattr(
        resume,
        "load_instances",
        lambda _path: [SimpleNamespace(question_id="q1", question_type="single-session-user")],
    )

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


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_purge", [False, True])
async def test_resume_purges_each_question_before_scoring_the_next(
    tmp_path, monkeypatch, fail_purge
):
    import argparse

    import openai

    import app.database as database
    import eval.resume_system_run as resume

    qids = ["q1"] if fail_purge else ["q1", "q2"]
    stack = _stack(dict(_CONTEXT_POLICY))
    stack["selected_question_ids"] = qids
    records = [
        {
            "question_id": qid,
            "question_type": "single-session-user",
            "correct": False,
            "response": "",
            "memories_recalled": 0,
            "error": "temporary provider error" if qid == "q1" else None,
        }
        for qid in qids
    ]
    results = tmp_path / "run.json"
    results.write_text(
        json.dumps(
            {
                "stack": stack,
                "sample": {"n": len(qids), "question_ids": qids},
                "per_question": records,
                "errors": 1,
                "comparison_to_baseline": {"baseline_mean": 0.5, "delta": -0.5},
            }
        )
    )
    monkeypatch.setattr(resume, "build_stack_metadata", lambda **_kwargs: stack)

    async def no_bootstrap():
        return None

    monkeypatch.setattr(resume, "bootstrap_sqlite", no_bootstrap)
    monkeypatch.setattr(resume, "_apply_graph_build_switch", lambda _enabled: None)
    monkeypatch.setattr(resume, "_RESULTS_DIR", tmp_path / "results")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **_kwargs: object())

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def add(self, _row):
            pass

        async def commit(self):
            pass

    monkeypatch.setattr(database, "AsyncSessionLocal", FakeSession)
    instances = [
        SimpleNamespace(question_id=qid, question_type="single-session-user")
        for qid in qids
    ]
    monkeypatch.setattr(resume, "load_instances", lambda _path: instances)
    memories = {}
    purged = []
    scored = []
    purge_failures_remaining = [int(fail_purge)]

    async def ingest(user_id, instance, *_args, **_kwargs):
        memories.setdefault(user_id, []).append(instance.question_id)
        return 1

    async def answer(user_id, instance, *_args, **_kwargs):
        assert memories[user_id] == [instance.question_id]
        scored.append(instance.question_id)
        return "answer", 1

    async def judge(_client, _instance, _response):
        return True

    async def purge(user_id, qid):
        assert memories[user_id] == [qid]
        purged.append(qid)
        if purge_failures_remaining[0]:
            purge_failures_remaining[0] -= 1
            return None
        memories[user_id].clear()
        return 1

    monkeypatch.setattr(resume, "ingest_instance", ingest)
    monkeypatch.setattr(resume, "answer_from_stack", answer)
    monkeypatch.setattr(resume, "judge_one", judge)
    monkeypatch.setattr(resume, "purge_instance_memories", purge, raising=False)

    args = argparse.Namespace(
        results=results,
        session=None,
        chunk_chars=None,
        fuse=None,
        top_k=None,
        from_index=None,
        dry_run=False,
    )
    exit_code = await resume.main_async(args)
    updated = json.loads(results.read_text())
    assert purged == (qids[:1] if fail_purge else qids)
    assert scored == (qids[:1] if fail_purge else qids)
    if fail_purge:
        assert exit_code == 2
        assert updated["partial"] is True
        assert updated["resumed"]["complete"] is False
        assert updated["per_question"][0]["correct"] is True
        failed_user_id = next(iter(memories))
        assert updated["resumed"]["purge_failed_user_id"] == str(failed_user_id)
        assert await resume.main_async(args) == 0
        assert scored == qids[:1] * 2
        assert purged == qids[:1] * 3
        assert memories[failed_user_id] == []
        retried = json.loads(results.read_text())
        assert retried["partial"] is False
        assert retried["resumed"]["complete"] is True
        return

    assert exit_code == 0
    updated = json.loads(results.read_text())
    assert updated["partial"] is False
    assert updated["resumed"]["complete"] is True
    assert updated["resumed"]["questions"] == len(qids)
    assert updated["errors"] == 0
    assert updated["wilson_95"] == [0.342, 1.0]
    assert updated["comparison_to_baseline"]["delta"] == 0.5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sample_ids,record_ids",
    [(["q1", "q2"], ["q1"]), (["q2", "q1"], ["q1", "q2"])],
)
async def test_resume_refuses_an_incomplete_sample_before_working(
    tmp_path, monkeypatch, capsys, sample_ids, record_ids
):
    import argparse

    import eval.resume_system_run as resume

    stack = _stack(dict(_CONTEXT_POLICY))
    stack["selected_question_ids"] = sample_ids
    results = tmp_path / "run.json"
    results.write_text(
        json.dumps(
            {
                "stack": stack,
                "sample": {"n": len(sample_ids), "question_ids": sample_ids},
                "per_question": [
                    {"question_id": qid, "memories_recalled": 0} for qid in record_ids
                ],
            }
        )
    )
    monkeypatch.setattr(resume, "build_stack_metadata", lambda **_kwargs: stack)

    bootstrap_calls = []

    async def no_bootstrap():
        bootstrap_calls.append(True)
        return None

    monkeypatch.setattr(resume, "bootstrap_sqlite", no_bootstrap)
    args = argparse.Namespace(
        results=results,
        session=None,
        chunk_chars=None,
        fuse=None,
        top_k=None,
        from_index=None,
        dry_run=True,
    )

    assert await resume.main_async(args) == 2
    assert bootstrap_calls == []
    assert "sample" in capsys.readouterr().err


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["unknown_id", "missing_question_type"])
async def test_resume_validates_ids_and_records_before_work(tmp_path, monkeypatch, invalid):
    import argparse

    import openai

    import app.database as database
    import eval.resume_system_run as resume

    qid = "unknown" if invalid == "unknown_id" else "q1"
    record = {
        "question_id": qid,
        "correct": False,
        "memories_recalled": 1,
        "error": None,
    }
    if invalid != "missing_question_type":
        record["question_type"] = "single-session-user"
    stack = _stack(dict(_CONTEXT_POLICY))
    stack["selected_question_ids"] = [qid]
    results = tmp_path / "run.json"
    results.write_text(
        json.dumps(
            {
                "stack": stack,
                "sample": {"n": 1, "question_ids": [qid]},
                "partial": False,
                "per_question": [record],
            }
        )
    )
    monkeypatch.setattr(resume, "build_stack_metadata", lambda **_kwargs: stack)
    monkeypatch.setattr(resume, "_RESULTS_DIR", tmp_path / "results")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        resume,
        "load_instances",
        lambda _path: [SimpleNamespace(question_id="q1", question_type="single-session-user")],
    )

    bootstrap_calls = []
    client_calls = []
    work_calls = []

    async def no_bootstrap():
        bootstrap_calls.append(True)

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def add(self, _row):
            pass

        async def commit(self):
            pass

    async def ingest(*_args, **_kwargs):
        work_calls.append("ingest")
        return 1

    async def answer(*_args, **_kwargs):
        work_calls.append("answer")
        return "answer", 1

    async def judge(*_args, **_kwargs):
        work_calls.append("judge")
        return True

    async def purge(*_args, **_kwargs):
        work_calls.append("purge")
        return 0

    monkeypatch.setattr(resume, "bootstrap_sqlite", no_bootstrap)
    monkeypatch.setattr(database, "AsyncSessionLocal", FakeSession)
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **_kwargs: client_calls.append(True))
    monkeypatch.setattr(resume, "ingest_instance", ingest)
    monkeypatch.setattr(resume, "answer_from_stack", answer)
    monkeypatch.setattr(resume, "judge_one", judge)
    monkeypatch.setattr(resume, "purge_instance_memories", purge, raising=False)
    args = argparse.Namespace(
        results=results,
        session=None,
        chunk_chars=None,
        fuse=None,
        top_k=None,
        from_index=None,
        dry_run=False,
    )

    assert await resume.main_async(args) == 2
    assert bootstrap_calls == client_calls == work_calls == []


@pytest.mark.asyncio
async def test_resume_refuses_sample_different_from_stack_metadata(tmp_path, monkeypatch):
    import argparse

    import eval.resume_system_run as resume

    stack = _stack(dict(_CONTEXT_POLICY))
    stack["selected_question_ids"] = ["q1", "q2"]
    results = tmp_path / "run.json"
    results.write_text(
        json.dumps(
            {
                "stack": stack,
                "sample": {"n": 1, "question_ids": ["q1"]},
                "per_question": [{
                    "question_id": "q1",
                    "question_type": "single-session-user",
                    "correct": False,
                    "memories_recalled": 0,
                    "error": None,
                }],
            }
        )
    )
    monkeypatch.setattr(resume, "build_stack_metadata", lambda **_kwargs: stack)
    bootstrap_calls = []

    async def no_bootstrap():
        bootstrap_calls.append(True)

    monkeypatch.setattr(resume, "bootstrap_sqlite", no_bootstrap)
    args = argparse.Namespace(
        results=results,
        session=None,
        chunk_chars=None,
        fuse=None,
        top_k=None,
        from_index=None,
        dry_run=True,
    )

    assert await resume.main_async(args) == 2
    assert bootstrap_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("partial_status", "expected_rerun"), [(None, ["q1"]), (False, [])])
async def test_resume_uses_json_partial_status_for_filename_markers(
    tmp_path, monkeypatch, capsys, partial_status, expected_rerun
):
    import argparse

    import eval.resume_system_run as resume

    stack = _stack(dict(_CONTEXT_POLICY))
    results = tmp_path / "longmemeval_s_system_partial.json"
    payload = {
        "stack": stack,
        "sample": {"n": 1, "question_ids": ["q1"]},
        "per_question": [{
            "question_id": "q1",
            "question_type": "single-session-user",
            "correct": True,
            "memories_recalled": 1,
            "error": None,
        }],
    }
    if partial_status is not None:
        payload["partial"] = partial_status
    results.write_text(json.dumps(payload))
    monkeypatch.setattr(resume, "build_stack_metadata", lambda **_kwargs: stack)
    monkeypatch.setattr(
        resume,
        "load_instances",
        lambda _path: [SimpleNamespace(question_id="q1", question_type="single-session-user")],
    )
    bootstrap_calls = []

    async def no_bootstrap():
        bootstrap_calls.append(True)

    monkeypatch.setattr(resume, "bootstrap_sqlite", no_bootstrap)
    args = argparse.Namespace(
        results=results,
        session=None,
        chunk_chars=None,
        fuse=None,
        top_k=None,
        from_index=None,
        dry_run=True,
    )

    assert await resume.main_async(args) == 0
    assert bootstrap_calls == []
    assert f"would re-run: {expected_rerun}" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_resume_retries_last_record_of_legacy_partial_with_other_failures(
    tmp_path, monkeypatch, capsys
):
    import argparse

    import eval.resume_system_run as resume

    qids = ["q1", "q2"]
    stack = _stack(dict(_CONTEXT_POLICY))
    stack["selected_question_ids"] = qids
    results = tmp_path / "longmemeval_s_system_partial.json"
    results.write_text(
        json.dumps(
            {
                "stack": stack,
                "sample": {"n": 2, "question_ids": qids},
                "per_question": [
                    {
                        "question_id": "q1",
                        "question_type": "single-session-user",
                        "correct": False,
                        "memories_recalled": 0,
                        "error": "provider error",
                    },
                    {
                        "question_id": "q2",
                        "question_type": "single-session-user",
                        "correct": True,
                        "memories_recalled": 1,
                        "error": None,
                    },
                ],
            }
        )
    )
    monkeypatch.setattr(resume, "build_stack_metadata", lambda **_kwargs: stack)
    monkeypatch.setattr(
        resume,
        "load_instances",
        lambda _path: [
            SimpleNamespace(question_id=qid, question_type="single-session-user")
            for qid in qids
        ],
    )
    args = argparse.Namespace(
        results=results,
        session=None,
        chunk_chars=None,
        fuse=None,
        top_k=None,
        from_index=None,
        dry_run=True,
    )

    assert await resume.main_async(args) == 0
    assert "would re-run: ['q1', 'q2']" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_resume_refuses_dataset_drift_before_bootstrap(tmp_path, monkeypatch):
    import argparse

    import eval.resume_system_run as resume

    recorded = _stack(dict(_CONTEXT_POLICY))
    current = {**recorded, "dataset_sha256": "different-dataset"}
    results = tmp_path / "run.json"
    results.write_text(
        json.dumps(
            {
                "stack": recorded,
                "sample": {"n": 1, "question_ids": ["q1"]},
                "per_question": [{
                    "question_id": "q1",
                    "question_type": "single-session-user",
                    "correct": False,
                    "memories_recalled": 0,
                    "error": None,
                }],
            }
        )
    )
    monkeypatch.setattr(resume, "build_stack_metadata", lambda **_kwargs: current)
    bootstrap_calls = []

    async def no_bootstrap():
        bootstrap_calls.append(True)

    monkeypatch.setattr(resume, "bootstrap_sqlite", no_bootstrap)
    args = argparse.Namespace(
        results=results,
        session=None,
        chunk_chars=None,
        fuse=None,
        top_k=None,
        from_index=None,
        dry_run=True,
    )

    assert await resume.main_async(args) == 2
    assert bootstrap_calls == []
