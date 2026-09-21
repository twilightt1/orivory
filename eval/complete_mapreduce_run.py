#!/usr/bin/env python3
"""Complete an interrupted map-reduce run: run only missing questions.

Reads the partial outcomes recovered from the killed run's log
(/tmp/mr-partial.json), loads the same seeded sample (seed 20260906,
n=100), runs ONLY the question_ids not present in the partial set — same
config (session 4k chunks + overlap + decay floor + Jina rerank +
map-reduce answering) — and writes the full results JSON with provenance
including the split between the original run and this completion.

Usage: python3 eval/complete_mapreduce_run.py [--n 100]
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ENV_FILE = ROOT / ".env"
if not ENV_FILE.exists():
    ENV_FILE = ROOT.parent.parent / ".env"
for line in ENV_FILE.open():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())

os.environ["QDRANT_MODE"] = "local"
os.environ["JWT_SECRET_KEY"] = "benchmark-run-secret-key-not-for-prod"
_RESULTS_DIR = ROOT / "eval/benchmarks/results"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_RESULTS_DIR}/.system_run.db"
os.environ.setdefault("OPENROUTER_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
if os.environ.get("OPENAI_BASE_URL"):
    os.environ["OPENROUTER_BASE_URL"] = os.environ["OPENAI_BASE_URL"]
os.environ["QDRANT_LOCAL_PATH"] = str(_RESULTS_DIR / "qdrant")
os.environ["JINA_RERANKER_TOP_N"] = "15"
os.environ["RETRIEVAL_SEMANTIC_RERANK"] = "1"

from eval.benchmarks.llm_judge import JUDGE_PROMPT_VERSION  # noqa: E402
from eval.benchmarks.longmemeval_s import load_instances  # noqa: E402
from eval.run_system_benchmark import (  # noqa: E402
    DATASET,
    MODEL,
    ingest_instance,
    judge_one,
    stack_recall,
)

# FROZEN COPY of the map-reduce answering variant removed from
# run_system_benchmark.py (PR #20 NEGATIVE: 0.486 clean vs 0.570 single-pass).
# Kept here — and only here — so the recorded result stays reproducible.
# Do not extend, do not re-enable on the live path.
_MR_MAP_PROMPT = (
    "You are answering a question about a user's conversation history using "
    "ONE memory excerpt. Answer ONLY from that excerpt, in at most two "
    "sentences. If the excerpt does not help answer the question, reply "
    "exactly: I have no information about that."
)

_MR_FUSE_PROMPT = (
    "You are answering a question about a user's conversation history. "
    "Several partial findings were extracted from different memories; some "
    "may be irrelevant or say nothing was found. Synthesize them into ONE "
    "final answer (at most three sentences). Prefer concrete facts "
    "(dates, names, numbers) and reconcile conflicts by taking the most "
    "specific statement. If none of the findings answer the question, "
    "reply exactly: I have no information about that."
)


async def _answer_map_reduce(client, user_id, instance, top_k: int) -> tuple[str, int]:
    """Frozen two-phase answer: per-excerpt judgment, then fuse."""
    recalled = await stack_recall(user_id, instance.question, top_k)
    if not recalled:
        return "I have no information about that.", 0

    partials: list[str] = []

    async def _map_one(idx: int, r: dict) -> None:
        try:
            completion = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": _MR_MAP_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"MEMORY ({r['captured_at'] or 'undated'}):\n"
                            f"{r['content']}\n\nQUESTION: {instance.question}"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=120,
            )
            text = (completion.choices[0].message.content or "").strip()
            if text and text != "I have no information about that.":
                partials.append(f"[{idx + 1}] {text}")
        except Exception:
            pass  # a failed excerpt is simply absent from the reduce step

    await asyncio.gather(*(_map_one(i, r) for i, r in enumerate(recalled)))

    if not partials:
        return "I have no information about that.", len(recalled)

    findings = "\n".join(partials)
    completion = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _MR_FUSE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"PARTIAL FINDINGS:\n{findings}\n\nQUESTION: {instance.question}"
                ),
            },
        ],
        temperature=0.0,
        max_tokens=300,
    )
    return (completion.choices[0].message.content or "").strip(), len(recalled)

PARTIAL = Path(os.environ.get("MR_PARTIAL", "/tmp/mr-partial.json"))


async def main_async(args) -> int:
    from uuid import uuid4

    from openai import AsyncOpenAI

    from app.database import bootstrap_sqlite

    await bootstrap_sqlite()

    from app.database import AsyncSessionLocal
    from app.models.user import User

    user_id = uuid4()
    async with AsyncSessionLocal() as db:
        db.add(
            User(
                id=user_id,
                email=f"mr-complete-{user_id}@orivory.local",
                hashed_password="x",
                is_verified=True,
                is_active=True,
            )
        )
        await db.commit()

    all_instances = load_instances(DATASET)
    rng = random.Random(args.seed)
    idx = list(range(len(all_instances)))
    rng.shuffle(idx)
    selected = [all_instances[i] for i in sorted(idx[: args.n])]

    partial = {r["question_id"]: r for r in json.loads(PARTIAL.read_text())}
    missing = [inst for inst in selected if inst.question_id not in partial]
    print(f"partial records: {len(partial)} | missing: {len(missing)}")
    print(f"model: {MODEL} | judge: {JUDGE_PROMPT_VERSION} | top_k={args.top_k}")

    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
        timeout=240.0,
    )

    new_records: list[dict] = []
    t0 = time.time()
    for inst in missing:
        tt = time.time()
        try:
            ingested = await ingest_instance(
                user_id, inst, _RESULTS_DIR / "system_run",
                session_level=True, chunk_chars=4000,
            )
            response, recalled = await _answer_map_reduce(
                client, user_id, inst, args.top_k
            )
            correct = await judge_one(client, inst, response)
            error = None
        except Exception as exc:
            import traceback

            traceback.print_exc()
            response, recalled, ingested, correct, error = "", 0, 0, False, (
                f"{type(exc).__name__}: {exc}"
            )
        seconds = round(time.time() - tt, 1)
        status = "error" if error else ("correct" if correct else "incorrect")
        print(
            f"  {inst.question_id} [{inst.question_type}]: {status} "
            f"(ingested={ingested}, recalled={recalled}, {seconds}s)"
        )
        new_records.append(
            {
                "question_id": inst.question_id,
                "question_type": inst.question_type,
                "correct": correct,
                "response": response[:500],
                "memories_ingested": ingested,
                "memories_recalled": recalled,
                "seconds": seconds,
                "error": error,
            }
        )
    total = round(time.time() - t0, 1)

    all_records = [
        {
            "question_id": r["question_id"],
            "question_type": r["question_type"],
            "correct": r["correct"],
            "response": "",  # original run's responses not in the log
            "memories_ingested": r["memories_ingested"],
            "memories_recalled": r["memories_recalled"],
            "seconds": 0,
            "error": None,
            "from_original_run": True,
        }
        for r in partial.values()
        if r["question_id"] in {i.question_id for i in selected}
    ] + new_records

    scored = [r for r in all_records if not r.get("error")]
    correct_n = sum(1 for r in scored if r["correct"])
    mean = round(correct_n / len(scored), 3) if scored else 0.0
    n_s, p = len(scored), correct_n / len(scored) if scored else 0.0
    z = 1.96
    denom = 1 + z * z / n_s
    center = (p + z * z / (2 * n_s)) / denom
    half = z * (p * (1 - p) / n_s + z * z / (4 * n_s * n_s)) ** 0.5 / denom

    by_type: dict[str, dict] = {}
    for r in scored:
        slot = by_type.setdefault(r["question_type"], {"n": 0, "correct": 0})
        slot["n"] += 1
        slot["correct"] += 1 if r["correct"] else 0

    baseline = {}
    base_path = _RESULTS_DIR / "longmemeval_s_system_n100.json"
    if base_path.exists():
        baseline = json.loads(base_path.read_text())

    payload = {
        "benchmark": "longmemeval_s",
        "run_kind": "orivory_stack",
        "chunking": "session_level",
        "answer_mode": "map_reduce",
        "note": (
            "REAL dataset, session 4k chunks + overlap + decay floor + Jina "
            "semantic rerank + MAP-REDUCE answering (per-excerpt judgment "
            "then fuse). Completed in two segments: 58 questions in the "
            "original run (interrupted by Jina balance exhaustion), "
            f"{len(new_records)} completed by this script with the same "
            "config. Split recorded honestly — cross-segment comparisons "
            "carry that caveat."
        ),
        "dataset_path": "eval/benchmarks/data/longmemeval_s_cleaned.json",
        "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "dataset_instances": len(all_instances),
        "sample": {"n": len(selected), "seed": args.seed},
        "model": MODEL,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "stack": {
            "database": "sqlite (lite mode)",
            "vector_store": "qdrant local (in-process)",
            "embeddings": "jina-embeddings-v5-text-small",
            "retriever": "MemoryRetriever (semantic rerank + decay floor)",
            "recall_top_k": args.top_k,
            "answer_mode": "map_reduce",
        },
        "mean": mean,
        "questions": len(scored),
        "correct": correct_n,
        "errors": len([r for r in all_records if r.get("error")]),
        "wilson_95": [round(center - half, 3), round(center + half, 3)],
        "by_type": by_type,
        "segments": {
            "original_run_records": len([r for r in all_records if r.get("from_original_run")]),
            "completion_records": len(new_records),
        },
        "total_seconds": total,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "per_question": all_records,
    }
    if baseline.get("mean") is not None:
        payload["run_comparison"] = {
            "full_context_baseline_n20": 0.600,
            "orivory_per_turn_n20": 0.450,
            "orivory_session_n20_no_fix": 0.300,
            "orivory_session_n20_decay_floor": 0.700,
            "orivory_n100_no_rerank": 0.490,
            "orivory_n100_semantic_rerank_single_pass": 0.570,
            "orivory_n100_semantic_rerank_map_reduce": mean,
            "n100_wilson_95_no_rerank": [0.394, 0.587],
            "n100_wilson_95_single_pass": [0.472, 0.663],
            "n100_wilson_95_map_reduce": payload["wilson_95"],
        }
    out = _RESULTS_DIR / "longmemeval_s_system_n100_mapreduce.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nmap-reduce mean: {mean:.3f} ({correct_n}/{len(scored)}) → {out}")
    type_summary = {k: "{}/{}".format(v["correct"], v["n"]) for k, v in by_type.items()}
    print(f"by type: {type_summary}")
    print(f"wilson95: {payload['wilson_95']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
