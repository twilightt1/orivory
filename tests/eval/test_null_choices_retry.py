"""One retry on the gateway's choices=None blip, on both call paths.

The n=100 run lost question 4baee567 to a 200 response with choices=None;
the identical immediate request returned a normal completion. The retry must
fire on the blip and stay silent otherwise.
"""
from __future__ import annotations

import asyncio
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("LLM_MODEL", "test-model")


class _Gateway:
    """Returns choices=None for the first N calls, a real completion after."""

    def __init__(self, null_first: int):
        self.calls = 0
        self.null_first = null_first
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )

    async def _create(self, **_kwargs):
        self.calls += 1
        if self.calls <= self.null_first:
            return types.SimpleNamespace(choices=None)
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))
            ]
        )


def test_create_checked_retries_once_on_null_choices():
    from eval.run_system_benchmark import create_checked

    gateway = _Gateway(null_first=1)
    completion = asyncio.run(create_checked(gateway, model="m"))
    assert gateway.calls == 2
    assert completion.choices[0].message.content == "ok"


def test_create_checked_passes_through_a_persistent_null_without_retry():
    from eval.run_system_benchmark import create_checked

    gateway = _Gateway(null_first=2)
    completion = asyncio.run(create_checked(gateway, model="m"))
    assert gateway.calls == 2  # one retry, then the error record path owns it
    assert completion.choices is None


def test_shared_wrapper_retries_every_agent_on_the_client():
    """The retry belongs on the shared wrapper, not on one caller.

    `_ResilientCompletions` exists to fix all call sites at once; agents that
    bypass `complete()` (graph extraction, query rewriter, write-back, HyDE)
    would still see the raw null otherwise.
    """
    from app.agents import llm_client

    gateway = _Gateway(null_first=1)
    wrapped = llm_client.ResilientAsyncOpenAI(gateway)
    completion = asyncio.run(
        wrapped.chat.completions.create(model="m", messages=[])
    )
    assert gateway.calls == 2
    assert completion.choices[0].message.content == "ok"


def test_shared_wrapper_leaves_a_normal_completion_alone():
    from app.agents import llm_client

    gateway = _Gateway(null_first=0)
    wrapped = llm_client.ResilientAsyncOpenAI(gateway)
    asyncio.run(wrapped.chat.completions.create(model="m", messages=[]))
    assert gateway.calls == 1, "a healthy response must not be retried"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
