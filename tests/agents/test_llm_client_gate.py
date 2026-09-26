"""The provider gate belongs on the shared client, not on chosen callers.

Query rewrite and HyDE call the shared client directly and used to run
outside ``LLM_MAX_CONCURRENCY``; only ``complete()`` and graph extraction
took a permit. One seam must own the budget so a new call site cannot
burst the provider by accident.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents import llm_client  # noqa: E402


class _SlowGateway:
    """Counts peak concurrent creates so a missing gate shows up as a burst."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.inflight = 0
        self.peak = 0
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )

    async def _create(self, **_kwargs):
        with self._lock:
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)
        try:
            await asyncio.sleep(0.05)
        finally:
            with self._lock:
                self.inflight -= 1
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))]
        )


async def test_a_direct_client_call_still_takes_a_shared_permit(monkeypatch):
    """Any `client.chat.completions.create` goes through the app-wide gate."""
    limit = 2
    monkeypatch.setattr(llm_client.settings, "LLM_MAX_CONCURRENCY", limit)
    monkeypatch.setattr(llm_client, "_llm_semaphore", None, raising=False)

    gateway = _SlowGateway()
    wrapped = llm_client.ResilientAsyncOpenAI(gateway)

    await asyncio.gather(
        *(
            wrapped.chat.completions.create(model="m", messages=[])
            for _ in range(limit * 3)
        )
    )

    assert gateway.peak <= limit, (
        f"the shared client burst the provider: {gateway.peak} concurrent creates "
        f"with LLM_MAX_CONCURRENCY={limit}"
    )


async def test_complete_does_not_take_a_second_permit(monkeypatch):
    """`complete()` already holds the permit; the wrapper must not re-enter.

    At LLM_MAX_CONCURRENCY=1 a second acquisition deadlocks: the task that
    would release the permit is the one waiting for it.
    """
    monkeypatch.setattr(llm_client.settings, "LLM_MAX_CONCURRENCY", 1)
    monkeypatch.setattr(llm_client, "_llm_semaphore", None, raising=False)

    gateway = _SlowGateway()
    wrapped = llm_client.ResilientAsyncOpenAI(gateway)
    monkeypatch.setattr(llm_client, "get_llm_client", lambda: wrapped)

    response = await asyncio.wait_for(
        llm_client.complete(
            agent="grader",
            messages=[{"role": "user", "content": "grade this"}],
        ),
        timeout=5,
    )

    assert response.choices[0].message.content == "ok"
    assert gateway.peak == 1, f"complete() nested a permit: peak {gateway.peak}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
