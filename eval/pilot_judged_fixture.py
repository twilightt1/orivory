#!/usr/bin/env python3
"""Pilot benchmark run: fixture haystack → simulated memory recall → LLM judge.

Honest pilot protocol (per eval/benchmarks/README hygiene rules):
- The dataset is REAL schema (the CI fixture: 2 instances) and every judge
  verdict is a REAL LLM call through the gateway in .env — nothing here is
  fabricated.
- The MEMORY SYSTEM being scored is simulated: a haystack-reading baseline
  ("full-context reader") that answers from the raw sessions. This is a
  PLUMBING VALIDATION of the judge path, NOT a LongMemEval headline score —
  the README's baseline-completeness rule still applies (no full-context
  baseline claim is recorded beyond what this pilot measures).
- Results land in eval/benchmarks/results/ with judge prompt version +
  per-question verdicts, so any later real run can be compared honestly.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load .env (OPENAI_API_KEY / OPENAI_BASE_URL / LLM_MODEL). Absent is fine:
# fall through to the ambient environment rather than dying on a
# FileNotFoundError the caller cannot act on.
ENV_FILE = ROOT / ".env"
if not ENV_FILE.exists():  # worktree: fall back to the main checkout's env
    ENV_FILE = ROOT.parent.parent / ".env"
for line in ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []:
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())

from eval.benchmarks.llm_judge import JUDGE_PROMPT_VERSION, build_judge_messages  # noqa: E402

FIXTURE = ROOT / "eval/benchmarks/fixtures/longmemeval_s_fixture.json"
RESULTS = ROOT / "eval/benchmarks/results/pilot_judged_fixture.json"
MODEL = os.environ.get("LLM_MODEL") or "openai/gpt-4o-mini"


def answer_from_haystack(instance: dict) -> str:
    """Simulated memory system: read the haystack, answer the question.

    The pilot system is intentionally simple — a reader that gets the full
    haystack (the session transcripts) and answers. For the abstention
    instance the gold expects "no information"; the reader must resist
    inventing a car model when none was recorded.
    """
    sessions = instance["haystack_sessions"]
    transcript = "\n".join(
        f"{turn.get('role', 'user')}: {turn.get('content', '')}"
        for session in sessions
        for turn in session
    )
    # The "system" answers via the same LLM, reading the full transcript —
    # this is the full-context baseline, which is what a pilot can honestly
    # measure (no retrieval, no compression — raw reading comprehension).
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        timeout=150.0,
    )

    async def _call() -> str:
        completion = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are answering questions about a user's conversation "
                        "history. Answer ONLY from the transcript below. If the "
                        "transcript does not contain the answer, reply exactly: "
                        "I have no information about that."
                    ),
                },
                {"role": "user", "content": f"TRANSCRIPT:\n{transcript}\n\nQUESTION: {instance['question']}"},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        return (completion.choices[0].message.content or "").strip()

    return asyncio.run(_call())


async def judge(client, question: str, gold: str, response: str) -> str:
    completion = await client.chat.completions.create(
        model=MODEL,
        messages=build_judge_messages(question, gold, response),
        temperature=0.0,
        max_tokens=8,
    )
    verdict = (completion.choices[0].message.content or "").strip().lower().strip("`.*!\n ")
    # exact-token match — "correct" is a substring of "incorrect"
    return "correct" if verdict == "correct" else "incorrect"


def main() -> int:
    instances = json.loads(FIXTURE.read_text())
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        timeout=150.0,
    )
    per_question: list[dict] = []
    t0 = time.time()
    for instance in instances:
        answer_t = time.time()
        response = answer_from_haystack(instance)
        answer_seconds = time.time() - answer_t
        judge_t = time.time()
        verdict = asyncio.run(judge(client, instance["question"], instance["answer"], response))
        judge_seconds = time.time() - judge_t
        per_question.append(
            {
                "question_id": instance["question_id"],
                "question_type": instance["question_type"],
                "correct": verdict == "correct",
                "response": response[:400],
                "answer_seconds": round(answer_seconds, 1),
                "judge_seconds": round(judge_seconds, 1),
            }
        )
        print(f"{instance['question_id']}: {verdict}  ({judge_seconds:.1f}s judge)")

    correct = sum(1 for r in per_question if r["correct"])
    mean = correct / len(per_question) if per_question else 0.0
    payload = {
        "pilot": True,
        "note": (
            "Plumbing validation on the CI fixture (2 instances). The memory "
            "system is a full-context reader baseline, NOT the retrieval stack "
            "— do not cite as a LongMemEval score."
        ),
        "benchmark": "longmemeval_s",
        "dataset_path": "eval/benchmarks/fixtures/longmemeval_s_fixture.json",
        "dataset_sha256": __import__("hashlib").sha256(FIXTURE.read_bytes()).hexdigest(),
        "model": MODEL,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "mean": round(mean, 3),
        "questions": len(per_question),
        "correct": correct,
        "per_question": per_question,
        "total_seconds": round(time.time() - t0, 1),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(payload, indent=2))
    print(f"\npilot mean: {mean:.3f} ({correct}/{len(per_question)}) → {RESULTS}")
    print(f"total: {payload['total_seconds']}s | judge: {JUDGE_PROMPT_VERSION} | model: {MODEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
