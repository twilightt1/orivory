#!/usr/bin/env python3
"""Real sampled benchmark run: LongMemEval-S (n instances) with the LLM judge.

Honest protocol — what this IS and IS NOT:
- Dataset: the REAL LongMemEval-S (500 instances, sha256 recorded).
- The memory system under test is the FULL-CONTEXT READER BASELINE: the
  model reads the instance's entire haystack transcript and answers. No
  retrieval, no compression, no Orivory stack in the loop yet. This is the
  baseline the README's hygiene rules require before any system claim —
  results are recorded AS the baseline, never as Orivory's score.
- Every answer and every verdict is a REAL LLM call through the gateway in
  .env. Nothing is fabricated; per-question records carry both responses
  and verdicts.
- Sampling is DETERMINISTIC (seeded) so the run is reproducible.

Usage:
    python3 eval/run_real_sample.py --n 20 [--seed 20260906] [--concurrency 4]
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
if not ENV_FILE.exists():  # worktree: fall back to the main checkout's env
    ENV_FILE = ROOT.parent.parent / ".env"
for line in ENV_FILE.open():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())

from eval.benchmarks.llm_judge import JUDGE_PROMPT_VERSION, build_judge_messages  # noqa: E402
from eval.benchmarks.longmemeval_s import load_instances  # noqa: E402
from eval.run_system_benchmark import ANSWER_MAX_TOKENS, create_checked  # noqa: E402

DATASET = ROOT / "eval/benchmarks/data/longmemeval_s_cleaned.json"
MODEL = os.environ.get("BENCHMARK_JUDGE_MODEL", os.environ["LLM_MODEL"])

ANSWER_SYSTEM = (
    "You are answering questions about a user's conversation history. "
    "Answer ONLY from the transcript below, in at most two sentences. If the "
    "transcript does not contain the answer, reply exactly: "
    "I have no information about that."
)


def _transcript(instance) -> str:
    lines: list[str] = []
    for session in instance.sessions:
        if session.date:
            lines.append(f"--- session {session.session_id} ({session.date}) ---")
        for turn in session.turns:
            lines.append(f"{turn.role}: {turn.content}")
    return "\n".join(lines)


async def answer_one(client, instance) -> str:
    completion = await create_checked(
        client,
        model=MODEL,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"TRANSCRIPT:\n{_transcript(instance)}\n\n"
                    f"QUESTION: {instance.question}"
                ),
            },
        ],
        temperature=0.0,
        # Same reason as the system lane: 300 lets a reasoning model finish
        # the budget before emitting `content` and return content=None.
        max_tokens=ANSWER_MAX_TOKENS,
    )
    return (completion.choices[0].message.content or "").strip()


async def judge_one(client, instance, response: str) -> bool:
    completion = await create_checked(
        client,
        model=MODEL,
        messages=build_judge_messages(instance.question, instance.answer, response),
        temperature=0.0,
        max_tokens=8,
    )
    verdict = (completion.choices[0].message.content or "").strip().lower().strip("`.*!\n ")
    return verdict == "correct"


async def run_instance(client, instance, sem: asyncio.Semaphore) -> dict:
    async with sem:
        t0 = time.time()
        try:
            response = await answer_one(client, instance)
            correct = await judge_one(client, instance, response)
            error = None
        except Exception as exc:  # keep the run alive; record the failure
            response, correct, error = "", False, f"{type(exc).__name__}: {exc}"
        seconds = round(time.time() - t0, 1)
        status = "error" if error else ("correct" if correct else "incorrect")
        print(f"  {instance.question_id} [{instance.question_type}]: {status} ({seconds}s)")
        return {
            "question_id": instance.question_id,
            "question_type": instance.question_type,
            "correct": correct,
            "response": response[:500],
            "seconds": seconds,
            "error": error,
        }


async def run(instances, concurrency: int) -> list[dict]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
        timeout=240.0,
    )
    sem = asyncio.Semaphore(concurrency)
    return await asyncio.gather(*(run_instance(client, inst, sem) for inst in instances))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "eval/benchmarks/results/longmemeval_s_baseline.json",
    )
    args = parser.parse_args()

    all_instances = load_instances(DATASET)
    # Deterministic sample: shuffle with seed, take n — reproducible run.
    rng = random.Random(args.seed)
    indices = list(range(len(all_instances)))
    rng.shuffle(indices)
    selected = [all_instances[i] for i in sorted(indices[: args.n])]

    type_counts: dict[str, int] = {}
    for inst in selected:
        type_counts[inst.question_type] = type_counts.get(inst.question_type, 0) + 1

    print(f"dataset: {DATASET.name} ({len(all_instances)} instances)")
    print(f"sample: {len(selected)} (seed={args.seed}) types={type_counts}")
    print(f"model: {MODEL} | judge: {JUDGE_PROMPT_VERSION} | concurrency: {args.concurrency}")

    t0 = time.time()
    records = asyncio.run(run(selected, args.concurrency))
    total = round(time.time() - t0, 1)

    errors = [r for r in records if r["error"]]
    correct = sum(1 for r in records if r["correct"])
    scored = [r for r in records if not r["error"]]
    mean = round(correct / len(scored), 3) if scored else 0.0

    by_type: dict[str, dict] = {}
    for r in scored:
        slot = by_type.setdefault(r["question_type"], {"n": 0, "correct": 0})
        slot["n"] += 1
        slot["correct"] += 1 if r["correct"] else 0

    payload = {
        "benchmark": "longmemeval_s",
        "run_kind": "full_context_reader_baseline",
        "note": (
            "REAL dataset (LongMemEval-S), REAL judge verdicts. The memory "
            "system is the full-context reader baseline — NOT the Orivory "
            "retrieval stack. Do not cite as Orivory's score; this is the "
            "baseline the protocol requires first."
        ),
        "dataset_path": "eval/benchmarks/data/longmemeval_s_cleaned.json",
        "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "dataset_instances": len(all_instances),
        "sample": {"n": len(selected), "seed": args.seed, "types": type_counts},
        "model": MODEL,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "mean": mean,
        "questions": len(scored),
        "correct": correct,
        "errors": len(errors),
        "by_type": by_type,
        "total_seconds": total,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "per_question": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nbaseline mean: {mean:.3f} ({correct}/{len(scored)}, errors={len(errors)})")
    type_summary = {k: "{}/{}".format(v["correct"], v["n"]) for k, v in by_type.items()}
    print(f"by type: {type_summary}")
    print(f"total: {total}s → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
