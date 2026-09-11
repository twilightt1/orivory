"""LLM judge for LongMemEval answers (pinned protocol).

LongMemEval's official protocol judges ALL question types with an LLM, not
just short numeric answers (see
https://github.com/xiaowu0162/LongMemEval — src/evaluation/evaluate_qa.py).
The exact-match guard in ``longmemeval_s.judge_answer`` covers the small
deterministic subset; everything else needs this judge.

Judge prompt: pinned to the official LongMemEval judge prompt semantics —
the judge sees the question, the gold answer, and the model response, and
answers a strict yes/no "is the response equivalent to the gold answer".
The version constant below MUST be bumped whenever the prompt changes; run
reports record it (leaderboard hygiene rule 1, see eval/benchmarks/README).

Requires ``OPENAI_API_KEY`` (the same client the backend uses). When the
key is absent every judge call raises ``JudgeUnavailable`` — the runner
surfaces that as ``pending_interpretation`` rather than a fabricated score.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

#: Bump when the prompt semantics change — recorded in every results file.
JUDGE_PROMPT_VERSION = "longmemeval-official-v1"

_JUDGE_SYSTEM_PROMPT = """You are an answer-equivalence judge for a
long-term memory benchmark. Given a QUESTION, a GOLD ANSWER, and a MODEL
RESPONSE, decide whether the model response conveys the same core
information as the gold answer. Judge only factual equivalence for the
question asked; ignore phrasing, extra disclaimers, and politeness. If the
gold answer says information is unavailable, the response must equally
decline to answer for it to be correct.

Reply with exactly one token: `correct` or `incorrect`."""


class JudgeUnavailable(RuntimeError):
    """Raised when no LLM credentials are configured."""


def build_judge_messages(question: str, answer: str, response: str) -> list[dict[str, str]]:
    """The exact message list sent to the LLM (exported for tests/CI)."""
    return [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"QUESTION: {question}\n\n"
                f"GOLD ANSWER: {answer}\n\n"
                f"MODEL RESPONSE: {response}\n\n"
                "Is the model response correct? Reply with exactly one token: "
                "`correct` or `incorrect`."
            ),
        },
    ]


async def judge_answer(question: str, answer: str, response: str) -> bool:
    """Judge one answer via the LLM. True = equivalent to gold.

    Honors ``OPENAI_BASE_URL`` (local gateways / proxies) and
    ``BENCHMARK_JUDGE_MODEL`` (default ``gpt-4o-mini`` — the model is part
    of the run's honesty record, so it is read at call time and reported in
    every results file).
    """
    import os

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise JudgeUnavailable(
            "OPENAI_API_KEY is not set — LLM-judged scores cannot be produced "
            "(and will not be fabricated). Set the key to enable scoring."
        )

    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    messages = build_judge_messages(question, answer, response)
    completion = await client.chat.completions.create(
        model=os.environ.get("BENCHMARK_JUDGE_MODEL", "gpt-4o-mini"),
        messages=messages,
        temperature=0.0,
        max_tokens=8,
    )
    verdict = (completion.choices[0].message.content or "").strip().lower().strip("`.*!\n ")
    # stdlib logger: no structlog-style kwargs (they raise TypeError once the
    # effective level reaches INFO — i.e. in production and full-suite runs).
    log.info("benchmark judge verdict verdict=%s question=%.80s", verdict, question)
    # EXACT token match: "correct" is a substring of "incorrect", so a naive
    # substring test flips every rejection into a pass — caught by a real
    # gateway run (the fixture pilot scored 2/2 while wrong answers also
    # "passed"). Judge semantics: one token, correct OR incorrect.
    return verdict == "correct"
