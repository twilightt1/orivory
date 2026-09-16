"""Regression test: the shared LLM client must not go stale across loops.

Found in full-repo review (HIGH): `get_llm_client()` cached one
AsyncOpenAI at module scope, but `build_memory_graph_sync` (Celery) runs
each task via `asyncio.run()` on a FRESH loop. The cached httpx pool stays
bound to the first task's (now closed) loop, so every later task in that
worker raised and silently degraded to deterministic fallback extraction.
"""
from __future__ import annotations

import asyncio
import threading

import pytest

from app.agents import llm_client


@pytest.fixture(autouse=True)
def _fresh_client_cache(monkeypatch):
    # Dummy key: constructing the client is offline, only calls hit network.
    monkeypatch.setattr(llm_client.settings, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(llm_client, "_client", None, raising=False)
    monkeypatch.setattr(llm_client, "_client_loop", None, raising=False)
    yield
    monkeypatch.setattr(llm_client, "_client", None, raising=False)
    monkeypatch.setattr(llm_client, "_client_loop", None, raising=False)


async def _get():
    return llm_client.get_llm_client()


def test_same_loop_reuses_client():
    async def two_calls():
        return llm_client.get_llm_client() is llm_client.get_llm_client()

    assert asyncio.run(two_calls()) is True


def test_new_loop_rebuilds_client():
    """Two sequential asyncio.run() calls (the Celery pattern) must NOT
    share one httpx-bound client — the second run gets a fresh instance."""
    first = asyncio.run(_get())
    second = asyncio.run(_get())
    assert first is not second


def test_closed_loop_never_reused():
    first = asyncio.run(_get())
    first_loop = llm_client._client_loop
    assert first_loop is not None and first_loop.is_closed()
    second = asyncio.run(_get())
    assert second is not first
    assert llm_client._client_loop is not first_loop


# ── the shared gate: no event-loop affinity (P2/T9 fix round 1, I1) ─────────
#
# Graph extraction waits on the app-wide gate from the builder's worker-thread
# loop. An ``asyncio.Semaphore`` cannot serve that: a second loop that has to
# wait raises "bound to a different event loop", and a waiter parked on the
# bound loop is never woken by a release from another loop (``call_soon`` does
# not wake a sleeping loop) — the thread hangs, and so does the request it
# serves. The gate must therefore be loop-free, while staying ONE shared budget.


async def _hold_gate(gate: llm_client._SharedSemaphore, seconds: float) -> None:
    async with gate:
        await asyncio.sleep(seconds)


def test_the_gate_is_waitable_from_a_worker_thread_loop():
    """The app loop holds the only permit; two worker-thread loops wait for it."""
    gate = llm_client._SharedSemaphore(1)
    entered: list[str] = []
    lock = threading.Lock()

    def _thread_waiter(tag: str) -> None:
        async def _enter() -> None:
            async with gate:
                with lock:
                    entered.append(tag)

        asyncio.run(_enter())

    async def _scenario() -> None:
        holder = asyncio.create_task(_hold_gate(gate, 0.15))
        await asyncio.sleep(0.02)  # the app loop took the only permit
        threads = [
            # Daemon threads: a mutant that hangs a waiter must not hang pytest.
            threading.Thread(target=_thread_waiter, args=(f"t{index}",), daemon=True)
            for index in range(2)
        ]
        for thread in threads:
            thread.start()
        await holder  # releases the permit to the waiters
        for _ in range(40):  # let them wake — without blocking this loop
            if not any(thread.is_alive() for thread in threads):
                break
            await asyncio.sleep(0.05)
        assert all(not thread.is_alive() for thread in threads), (
            "a worker-thread wait on the shared gate never woke"
        )

    asyncio.run(_scenario())
    assert sorted(entered) == ["t0", "t1"], entered


def test_cancelling_a_gate_wait_does_not_bleed_the_permit():
    """A cancelled wait leaves its thread parked in ``acquire()``.

    That thread takes a permit as soon as one frees, so the gate has to hand it
    back — otherwise it bleeds one slot per cancelled wait until nothing passes.
    """
    gate = llm_client._SharedSemaphore(1)

    async def _scenario() -> None:
        holder = asyncio.create_task(_hold_gate(gate, 0.15))
        await asyncio.sleep(0.02)  # the permit is taken
        waiter = asyncio.create_task(_hold_gate(gate, 0.0))  # parks in a thread
        await asyncio.sleep(0.05)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        await holder  # the release goes to the orphaned thread
        try:
            async with asyncio.timeout(1.0):
                async with gate:  # reopened only if the permit came back
                    pass
        finally:
            # Safety valve for mutant runs: give any parked thread its permit
            # back so the loop can shut its executor down instead of hanging.
            gate._semaphore.release()

    asyncio.run(_scenario())
