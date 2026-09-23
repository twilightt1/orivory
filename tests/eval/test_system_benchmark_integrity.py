"""Eval-integrity guards for eval/run_system_benchmark.py.

Three defects an independent reader verified against the script:

1. ``ingest_instance`` dropped ``index_new_memory``'s bool return, so a failed
   vector upsert was counted as an ingested memory.
2. A partial run (some instances errored) still wrote the committed result
   artifact name and exited 0 — its mean covers fewer questions.
3. ONE ``benchmark_user_id`` for the whole run with no cleanup between
   instances: the retriever is user-scoped, so the next instance's recall can
   surface the previous instance's haystack (order-dependent scores).
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from sqlalchemy import select

from eval.benchmarks.longmemeval_s import load_instances
from eval.run_system_benchmark import (
    ingest_instance,
    purge_instance_memories,
    result_artifact_path,
    run_exit_code,
)

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "eval" / "benchmarks" / "fixtures" / "longmemeval_s_fixture.json"
)

pytestmark = pytest.mark.eval


async def _make_user(user_id: uuid.UUID, prefix: str) -> None:
    from app.database import AsyncSessionLocal
    from app.models.user import User

    async with AsyncSessionLocal() as db:
        db.add(
            User(
                id=user_id,
                email=f"{prefix}-{user_id}@example.com",
                hashed_password="x",
                is_verified=True,
                is_active=True,
            )
        )
        await db.commit()


# --- finding 1: only real index successes count as ingested --------------


async def test_ingest_counts_only_real_index_successes(tmp_path, monkeypatch):
    """A False return (or a raise) from the indexer is a FAILED upsert:
    it must not be counted as created, and the failure must be reported."""
    import app.retrieval.memory.write_back as write_back

    user_id = uuid.uuid4()
    await _make_user(user_id, "bench-index")
    instance = load_instances(FIXTURE)[0]  # 3 sessions → 3 memories
    assert len(instance.sessions) == 3

    outcomes: list[object] = [False, RuntimeError("qdrant down"), True]
    calls: list[object] = []

    async def fake_index(memory) -> bool:
        calls.append(memory)
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(write_back, "index_new_memory", fake_index)

    created = await ingest_instance(user_id, instance, tmp_path)

    assert len(calls) == 3
    assert created == 1  # only the third upsert landed
    report = json.loads((tmp_path / f"ingested_{instance.question_id}.json").read_text())
    assert report["memories"] == 1
    assert report["index_failed"] == 2


# --- finding 2: partial runs never masquerade as the clean artifact ------


def test_partial_run_targets_a_partial_artifact_not_the_committed_one():
    clean = result_artifact_path(n=20, chunking="session_level", partial=False)
    assert clean.name == "longmemeval_s_system_session.json"
    partial = result_artifact_path(n=20, chunking="session_level", partial=True)
    assert partial.name == "longmemeval_s_system_session_partial.json"
    assert partial != clean
    assert result_artifact_path(n=100, chunking="session_level", partial=True).name == (
        "longmemeval_s_system_n100_partial.json"
    )
    assert result_artifact_path(n=20, chunking="per_turn", partial=False).name == (
        "longmemeval_s_system.json"
    )


def test_errored_run_does_not_exit_clean():
    assert run_exit_code([]) == 0
    assert run_exit_code([{"error": None}, {"error": None}]) == 0
    assert run_exit_code([{"error": None}, {"error": "RateLimitError: slow down"}]) != 0
    # A quota-aborted run keeps the caller's own code.
    assert run_exit_code([{"error": None}], base_code=3) == 3


# --- finding 3: per-instance cleanup keeps recall order-independent ------


async def test_purge_instance_memories_deletes_only_that_instances_haystack(
    tmp_path, monkeypatch
):
    import app.retrieval.memory.vector_store as vector_store
    from app.database import AsyncSessionLocal
    from app.models.memory import Memory

    user_id = uuid.uuid4()
    await _make_user(user_id, "bench-purge")

    refs = ["bench:qA:s0", "bench:qA:s1", "bench:qA2:s0", "bench:qB:s0", "note:kept"]
    async with AsyncSessionLocal() as db:
        db.add_all(
            [
                Memory(
                    id=uuid.uuid4(),
                    user_id=user_id,
                    title="t",
                    content="c",
                    tags=[],
                    source_type="other",
                    source_ref=ref,
                )
                for ref in refs
            ]
        )
        await db.commit()

    vector_deletes: list[list[str]] = []

    async def fake_delete_memories(ids):
        vector_deletes.append(list(ids))
        return True

    monkeypatch.setattr(vector_store, "delete_memories", fake_delete_memories)

    removed = await purge_instance_memories(user_id, "qA")

    assert removed == 2
    async with AsyncSessionLocal() as db:
        remaining = (
            await db.execute(
                select(Memory.source_ref).where(Memory.user_id == user_id)
            )
        ).scalars().all()
    # qA2 is a DIFFERENT question id whose prefix must not collide with qA's.
    assert sorted(remaining) == ["bench:qA2:s0", "bench:qB:s0", "note:kept"]
    assert len(vector_deletes) == 1
    assert len(vector_deletes[0]) == 2


async def test_run_loop_purges_each_instance_before_the_next(tmp_path, monkeypatch):
    """The loop itself must purge: a haystack left behind is recallable by the
    next instance (same user, user-scoped retriever)."""
    import argparse

    import eval.run_system_benchmark as rsb
    from app.database import AsyncSessionLocal
    from app.models.memory import Memory

    instances = load_instances(FIXTURE)
    monkeypatch.setattr(rsb, "load_instances", lambda _path: instances)
    monkeypatch.setattr(rsb, "result_artifact_path", lambda **_kw: tmp_path / "results.json")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    ingested_user_ids: list[uuid.UUID] = []
    purges: list[str] = []
    real_purge = rsb.purge_instance_memories

    async def fake_run_instance(client, user_id, instance, top_k, run_dir, **_kw):
        ingested_user_ids.append(user_id)
        async with AsyncSessionLocal() as db:  # the haystack this instance leaves
            db.add(
                Memory(
                    id=uuid.uuid4(),
                    user_id=user_id,
                    title="hay",
                    content="haystack",
                    tags=[],
                    source_type="other",
                    source_ref=f"bench:{instance.question_id}:s0",
                )
            )
            await db.commit()
        return {
            "question_id": instance.question_id,
            "question_type": instance.question_type,
            "correct": True,
            "response": "a",
            "memories_ingested": 1,
            "memories_recalled": 1,
            "ingest_seconds": 0.0,
            "recall_seconds": 0.0,
            "seconds": 0.0,
            "error": None,
        }

    async def counting_purge(user_id, question_id):
        purges.append(question_id)
        return await real_purge(user_id, question_id)

    monkeypatch.setattr(rsb, "run_instance", fake_run_instance)
    monkeypatch.setattr(rsb, "purge_instance_memories", counting_purge)

    args = argparse.Namespace(
        n=2, seed=1, top_k=5, session=True, chunk_chars=0, fuse=False, concurrency=1
    )
    exit_code = await rsb.main_async(args)

    assert exit_code == 0
    assert purges == [inst.question_id for inst in instances]
    async with AsyncSessionLocal() as db:
        left = (
            await db.execute(
                select(Memory.source_ref).where(
                    Memory.user_id == ingested_user_ids[0],
                    Memory.source_ref.startswith("bench:"),
                )
            )
        ).scalars().all()
    assert left == []
