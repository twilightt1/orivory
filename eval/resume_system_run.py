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
import uuid
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

from uuid import uuid4  # noqa: E402

# Import AFTER env so settings pick up the run's DATABASE_URL
from app.database import bootstrap_sqlite  # noqa: E402
from eval.benchmarks.longmemeval_s import load_instances  # noqa: E402
from eval.retrieval_contract import retrieval_contract_matches  # noqa: E402
from eval.run_system_benchmark import (  # noqa: E402
    DATASET,
    answer_from_stack,
    build_stack_metadata,
    ingest_instance,
    judge_one,
)


async def delete_instance_memories(db, user_id, question_id: str) -> int:
    """Delete every memory ingested for one question (by source_ref prefix)."""
    from sqlalchemy import delete

    from app.models.memory import Memory

    result = await db.execute(
        delete(Memory).where(
            Memory.user_id == user_id,
            Memory.source_ref.like(f"bench:{question_id}:%"),
        )
    )
    await db.commit()
    return result.rowcount or 0


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

    await bootstrap_sqlite()
    records = payload["per_question"]

    failed_idx = [
        i
        for i, r in enumerate(records)
        if (args.from_index is not None and i >= args.from_index)
        or r.get("memories_recalled") == 0
    ]
    if args.dry_run:
        print(
            "would re-run:",
            [records[i]["question_id"] for i in failed_idx],
        )
        return 0
    if not failed_idx:
        print("nothing to resume — no failed records")
        return 0

    all_instances = {i.question_id: i for i in load_instances(DATASET)}
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
    for k, i in enumerate(failed_idx, 1):
        record = records[i]
        qid = record["question_id"]
        instance = all_instances[qid]
        async with AsyncSessionLocal() as db:
            user_row = await db.get(User, user_id)
            await delete_instance_memories(db, user_id, qid)
            user_id_str = str(user_row.id)
        # Re-ingest under the resume user using the verified local model.
        user_uuid = uuid.UUID(user_id_str)
        ingested = await ingest_instance(
            user_uuid, instance, run_dir, session_level=args.session,
            chunk_chars=args.chunk_chars,
        )
        response, recalled = await answer_from_stack(
            user_uuid, instance, args.top_k, fuse=args.fuse
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
        print(f"  [{k}/{len(failed_idx)}] {qid}: {'correct' if correct else 'incorrect'} "
              f"(recalled={recalled})")

    # Recompute aggregates
    scored = [r for r in records if not r.get("error")]
    correct = sum(1 for r in scored if r["correct"])
    mean = round(correct / len(scored), 3) if scored else 0.0
    n_s, p = len(scored), correct / len(scored) if scored else 0.0
    z = 1.96
    denom = 1 + z * z / n_s
    center = (p + z * z / (2 * n_s)) / denom
    half = z * math.sqrt(p * (1 - p) / n_s + z * z / (4 * n_s * n_s)) / denom

    by_type: dict[str, dict] = {}
    for r in scored:
        slot = by_type.setdefault(r["question_type"], {"n": 0, "correct": 0})
        slot["n"] += 1
        slot["correct"] += 1 if r["correct"] else 0

    payload["mean"] = mean
    payload["correct"] = correct
    payload["questions"] = len(scored)
    payload["wilson_95"] = [round(center - half, 3), round(center + half, 3)]
    payload["by_type"] = by_type
    payload["resumed"] = {
        "questions": len(failed_idx),
        "reason": (
            "re-ran zero-recall records under the matching "
            "local retrieval contract"
        ),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    results_path.write_text(json.dumps(payload, indent=2))
    print(f"\nresumed mean: {mean:.3f} ({correct}/{len(scored)}) → {results_path}")
    print(f"wilson95: {payload['wilson_95']}")
    return 0


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
    fuse.add_argument("--fuse", dest="fuse", action="store_true")
    fuse.add_argument("--no-fuse", dest="fuse", action="store_false")
    parser.set_defaults(fuse=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
