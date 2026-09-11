"""Regression test: the shared LLM client must not go stale across loops.

Found in full-repo review (HIGH): `get_llm_client()` cached one
AsyncOpenAI at module scope, but `build_memory_graph_sync` (Celery) runs
each task via `asyncio.run()` on a FRESH loop. The cached httpx pool stays
bound to the first task's (now closed) loop, so every later task in that
worker raised and silently degraded to deterministic fallback extraction.
"""
from __future__ import annotations

import asyncio

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
