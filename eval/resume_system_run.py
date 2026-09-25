#!/usr/bin/env python3
"""Resume a system benchmark run: re-run only failed questions and merge.

A long n=100 run can hit transient embedding-provider failures near the end.
Questions whose recall failed (recalled=0) say nothing about retrieval quality —
recording them as wrong would misreport the system. This resume path pins the
current local embedding/rerank lane and rejects result files from a different
retrieval contract.

This script:
1. loads the run's results JSON,
2. picks records with ``memories_recalled == 0`` (or --from index N),
3. deletes those instances' ingested memories (by source_ref prefix),
4. re-ingests, re-answers, re-judges them with the SAME config,
5. merges the fresh records back, recomputes mean/Wilson/by-type, and
   records the resume in the payload (``resumed`` block) — full honesty.

Usage: python3 eval/resume_system_run.py --results <results.json> [--dry-run]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ENV_FILE = ROOT / ".env"
if not ENV_FILE.exists():
    ENV_FILE = ROOT.parent.parent / ".env"
# No .env anywhere — an archive checkout, a bare container, a worktree whose
# parent has none: run on the ambient environment instead of dying in a
# FileNotFoundError the caller cannot act on.
lines = ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []
for line in lines:
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())

os.environ["QDRANT_MODE"] = "local"
_RESULTS_DIR = ROOT / "eval/benchmarks/results"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_RESULTS_DIR}/.system_run.db"
os.environ.setdefault("OPENROUTER_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
if os.environ.get("OPENAI_BASE_URL"):
    os.environ["OPENROUTER_BASE_URL"] = os.environ["OPENAI_BASE_URL"]
os.environ["QDRANT_LOCAL_PATH"] = str(_RESULTS_DIR / "qdrant")
# Resume only the current self-host retrieval lane, regardless of stale .env
# flags left by a prior provider configuration.
os.environ["USE_LOCAL_EMBEDDINGS"] = "1"
os.environ["RETRIEVAL_SEMANTIC_RERANK"] = "1"
os.environ["RERANK_TOP_N"] = "15"

from uuid import UUID, uuid4  # noqa: E402

# Import AFTER env so settings pick up the run's DATABASE_URL
from app.database import bootstrap_sqlite  # noqa: E402
from eval.benchmarks.longmemeval_s import load_instances  # noqa: E402
from eval.retrieval_contract import retrieval_contract_matches  # noqa: E402
from eval.run_system_benchmark import (  # noqa: E402
    DATASET,
    _apply_graph_build_switch,
    answer_from_stack,
    build_stack_metadata,
    ingest_instance,
    judge_one,
    purge_instance_memories,
)


async def main_async(args) -> int:
    from openai import AsyncOpenAI

    from app.database import AsyncSessionLocal
    from app.models.user import User

    results_path = args.results
    payload = json.loads(results_path.read_text())
    recorded_stack = payload.get("stack")
    if not isinstance(recorded_stack, dict):
        print("refusing to resume: result has no retrieval contract", file=sys.stderr)
        return 2
    recorded_graph_builds = recorded_stack.get("graph_builds")
    if not isinstance(recorded_graph_builds, str) or recorded_graph_builds not in {
        "on",
        "off (bench ingest)",
    }:
        print("refusing to resume: result lacks a valid graph-build policy", file=sys.stderr)
        return 2
    _apply_graph_build_switch(recorded_graph_builds == "on")
    recorded_execution = recorded_stack.get("execution")
    if not isinstance(recorded_execution, dict):
        print("refusing to resume: result lacks execution metadata", file=sys.stderr)
        return 2
    recorded_policy = recorded_execution.get("context_policy")
    if not isinstance(recorded_policy, dict):
        print("refusing to resume: result lacks a complete context policy", file=sys.stderr)
        return 2
    session_level = recorded_policy.get("session_level")
    chunk_chars = recorded_policy.get("chunk_chars")
    fuse = recorded_policy.get("fuse")
    if (
        not isinstance(session_level, bool)
        or not isinstance(chunk_chars, int)
        or isinstance(chunk_chars, bool)
        or not isinstance(fuse, bool)
    ):
        print("refusing to resume: result lacks a complete context policy", file=sys.stderr)
        return 2
    recorded_policy = {
        "session_level": session_level,
        "chunk_chars": chunk_chars,
        "fuse": fuse,
    }

    for argument, key in (("session", "session_level"), ("chunk_chars", "chunk_chars"), ("fuse", "fuse")):
        recorded_value = recorded_policy[key]
        requested_value = getattr(args, argument)
        if requested_value is not None and requested_value != recorded_value:
            print("refusing to resume: requested context policy differs from the run", file=sys.stderr)
            return 2
        setattr(args, argument, recorded_value)

    recorded_top_k = recorded_stack.get("recall_top_k")
    if (
        not isinstance(recorded_top_k, int)
        or isinstance(recorded_top_k, bool)
        or recorded_top_k < 1
        or (args.top_k is not None and args.top_k != recorded_top_k)
    ):
        print("refusing to resume: requested top_k differs from the run", file=sys.stderr)
        return 2
    args.top_k = recorded_top_k

    current_stack = build_stack_metadata(
        top_k=args.top_k,
        session_level=args.session,
        chunk_chars=args.chunk_chars,
        fuse=args.fuse,
    )
    if not retrieval_contract_matches(recorded_stack, current_stack):
        print(
            "refusing to resume: recorded retrieval contract differs from the "
            "current local embedding/rerank lane; start a fresh run",
            file=sys.stderr,
        )
        return 2
    recorded_dataset_sha256 = recorded_stack.get("dataset_sha256")
    if (
        not isinstance(recorded_dataset_sha256, str)
        or recorded_dataset_sha256 != current_stack.get("dataset_sha256")
    ):
        print("refusing to resume: dataset fingerprint differs from the recorded run", file=sys.stderr)
        return 2

    sample = payload.get("sample")
    records = payload.get("per_question")
    sample_ids = sample.get("question_ids") if isinstance(sample, dict) else None
    record_ids = (
        [record.get("question_id") for record in records]
        if isinstance(records, list) and all(isinstance(record, dict) for record in records)
        else None
    )
    if (
        not isinstance(sample, dict)
        or not isinstance(sample.get("n"), int)
        or isinstance(sample.get("n"), bool)
        or not isinstance(sample_ids, list)
        or any(not isinstance(qid, str) or not qid for qid in sample_ids)
        or sample["n"] != len(sample_ids)
        or len(set(sample_ids)) != len(sample_ids)
        or record_ids is None
        or any(not isinstance(qid, str) or not qid for qid in record_ids)
        or len(set(record_ids)) != len(record_ids)
        or sample_ids != record_ids
        or not isinstance(recorded_stack.get("selected_question_ids"), list)
        or sample_ids != recorded_stack.get("selected_question_ids")
    ):
        print("refusing to resume: sample IDs do not match the records and recorded stack", file=sys.stderr)
        return 2

    all_instances = {i.question_id: i for i in load_instances(DATASET)}
    unknown_ids = set(sample_ids) - all_instances.keys()
    if unknown_ids:
        print(
            f"refusing to resume: sample question IDs absent from dataset: {sorted(unknown_ids)}",
            file=sys.stderr,
        )
        return 2
    if any(
        not isinstance(record.get("question_type"), str)
        or record["question_type"] != all_instances[record["question_id"]].question_type
        or not isinstance(record.get("correct"), bool)
        or not isinstance(record.get("memories_recalled"), int)
        or isinstance(record.get("memories_recalled"), bool)
        or record["memories_recalled"] < 0
        or "error" not in record
        or (record["error"] is not None and not isinstance(record["error"], str))
        for record in records
    ):
        print("refusing to resume: per-question fields are missing or inconsistent", file=sys.stderr)
        return 2

    resumed = payload.get("resumed")
    purge_failed_id = (
        resumed.get("purge_failed_question_id") if isinstance(resumed, dict) else None
    )
    purge_failed_user_raw = (
        resumed.get("purge_failed_user_id") if isinstance(resumed, dict) else None
    )
    purge_failed_user_id = None
    if purge_failed_id is not None:
        if (
            not isinstance(purge_failed_id, str)
            or purge_failed_id not in sample_ids
            or not isinstance(purge_failed_user_raw, str)
        ):
            print("refusing to resume: prior purge failure has no valid owner", file=sys.stderr)
            return 2
        try:
            purge_failed_user_id = UUID(purge_failed_user_raw)
        except ValueError:
            print("refusing to resume: prior purge failure has an invalid owner ID", file=sys.stderr)
            return 2
    elif purge_failed_user_raw is not None:
        print("refusing to resume: purge owner is present without a failed question", file=sys.stderr)
        return 2
    legacy_partial = "partial" not in payload and "_partial" in results_path.stem
    failed_idx = [
        i
        for i, r in enumerate(records)
        if (args.from_index is not None and i >= args.from_index)
        or r.get("memories_recalled") == 0
        or bool(r.get("error"))
        or r.get("question_id") == purge_failed_id
    ]
    if legacy_partial and records and len(records) - 1 not in failed_idx:
        failed_idx.append(len(records) - 1)
    partial_artifact = (
        payload.get("partial") is not False
        if "partial" in payload
        else legacy_partial
    )
    if not failed_idx and partial_artifact:
        print("refusing to accept an incomplete artifact as a successful no-op", file=sys.stderr)
        return 2
    if args.dry_run:
        print(
            "would re-run:",
            [records[i]["question_id"] for i in failed_idx],
        )
        return 0
    if not failed_idx:
        print("nothing to resume — no failed records")
        return 0

    await bootstrap_sqlite()
    if purge_failed_user_id is not None:
        assert isinstance(purge_failed_id, str)
        if await purge_instance_memories(purge_failed_user_id, purge_failed_id) is None:
            print(f"prior purge still failed for {purge_failed_id}; stopping resume", file=sys.stderr)
            return 2
    user_id = uuid4()
    async with AsyncSessionLocal() as db:
        db.add(
            User(
                id=user_id,
                email=f"resume-{user_id}@orivory.local",
                hashed_password="x",
                is_verified=True,
                is_active=True,
            )
        )
        await db.commit()

    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
        timeout=240.0,
    )

    run_dir = _RESULTS_DIR / "system_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"re-running {len(failed_idx)} failed questions with the same config")
    resumed_count = 0
    purge_failed = None
    for k, i in enumerate(failed_idx, 1):
        record = records[i]
        qid = record["question_id"]
        instance = all_instances[qid]
        # Re-ingest under the resume user using the verified local model.
        ingested = await ingest_instance(
            user_id, instance, run_dir, session_level=args.session,
            chunk_chars=args.chunk_chars,
        )
        response, recalled = await answer_from_stack(
            user_id, instance, args.top_k, fuse=args.fuse
        )
        correct = await judge_one(client, instance, response)
        records[i] = {
            "question_id": qid,
            "question_type": record["question_type"],
            "correct": correct,
            "response": response[:500],
            "memories_ingested": ingested,
            "memories_recalled": recalled,
            "seconds": record.get("seconds"),
            "error": None,
            "resumed": True,
        }
        resumed_count += 1
        print(f"  [{k}/{len(failed_idx)}] {qid}: {'correct' if correct else 'incorrect'} "
              f"(recalled={recalled})")
        if await purge_instance_memories(user_id, qid) is None:
            print(f"PURGE FAILED for {qid}; stopping resume", file=sys.stderr)
            purge_failed = qid
            break

    # Recompute aggregates
    scored = [r for r in records if not r.get("error")]
    errors = sum(bool(r.get("error")) for r in records)
    correct = sum(1 for r in scored if r["correct"])
    mean = round(correct / len(scored), 3) if scored else 0.0
    wilson = None
    if scored:
        n_s, p = len(scored), correct / len(scored)
        z = 1.96
        denom = 1 + z * z / n_s
        center = (p + z * z / (2 * n_s)) / denom
        half = z * math.sqrt(p * (1 - p) / n_s + z * z / (4 * n_s * n_s)) / denom
        wilson = [round(center - half, 3), round(center + half, 3)]

    by_type: dict[str, dict] = {}
    for r in scored:
        slot = by_type.setdefault(r["question_type"], {"n": 0, "correct": 0})
        slot["n"] += 1
        slot["correct"] += 1 if r["correct"] else 0

    payload["mean"] = mean
    payload["correct"] = correct
    payload["questions"] = len(scored)
    payload["errors"] = errors
    payload["wilson_95"] = wilson
    payload["by_type"] = by_type
    comparison = payload.get("comparison_to_baseline")
    if isinstance(comparison, dict):
        baseline_mean = comparison.get("baseline_mean")
        comparison["delta"] = (
            round(mean - baseline_mean, 3)
            if isinstance(baseline_mean, (int, float))
            and not isinstance(baseline_mean, bool)
            and math.isfinite(baseline_mean)
            and scored
            else None
        )
    complete = purge_failed is None and errors == 0
    payload["partial"] = not complete
    payload["resumed"] = {
        "questions": resumed_count,
        "complete": complete,
        "purge_failed_question_id": purge_failed,
        "purge_failed_user_id": str(user_id) if purge_failed is not None else None,
        "reason": (
            "re-ran selected records under the matching "
            "local retrieval contract"
        ),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    results_path.write_text(json.dumps(payload, indent=2))
    print(f"\nresumed mean: {mean:.3f} ({correct}/{len(scored)}) → {results_path}")
    print(f"wilson95: {payload['wilson_95']}")
    return 0 if complete else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--from-index", type=int, default=None,
                        help="re-run every record at/after this index (else: recalled=0)")
    session = parser.add_mutually_exclusive_group()
    session.add_argument("--session", dest="session", action="store_true")
    session.add_argument("--no-session", dest="session", action="store_false")
    parser.set_defaults(session=None)
    parser.add_argument("--chunk-chars", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    fuse = parser.add_mutually_exclusive_group()
    fuse.add_argument(
        "--fuse",
        dest="fuse",
        action="store_true",
        help="also recall with the rewritten query and union results",
    )
    fuse.add_argument(
        "--no-fuse",
        dest="fuse",
        action="store_false",
        help="skip the second rewritten-query recall",
    )
    parser.set_defaults(fuse=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
