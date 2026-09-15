"""Scoring tests: the decay floor keeps semantic match dominant."""
from __future__ import annotations

import asyncio
import math
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.database import Base
from app.retrieval.embedder import EmbeddingDimensionMismatch
from app.retrieval.memory import retriever as retriever_module
from app.retrieval.memory.retriever import MemoryRetriever
from app.retrieval.memory.scoring import time_decay_score


def test_recent_memory_unchanged_by_floor():
    now = datetime.now(UTC)
    score, _ = time_decay_score(0.8, now - timedelta(days=1), now=now)
    expected = 0.8 * math.exp(-1 / 30)
    assert abs(score - expected) < 1e-9  # above the floor → pure decay


def test_old_memory_floors_instead_of_vanishing():
    now = datetime.now(UTC)
    score, reasons = time_decay_score(0.8, now - timedelta(days=1100), now=now)
    # unbounded decay would be ~1e-16; the floor keeps it at 0.1x
    assert score == 0.8 * 0.1
    assert any("decay" in r for r in reasons)


def test_floor_keeps_semantic_ordering():
    """The regression the benchmark caught: with unbounded decay, a 2023
    memory scored 1e-16 and ranking became noise (any fresh memory beat
    every old one, whatever the semantics). With the floor, the ordering
    between two equally-salient memories follows semantic match regardless
    of age."""
    now = datetime.now(UTC)
    strong_old, _ = time_decay_score(0.9, now - timedelta(days=1100), now=now)
    weak_old, _ = time_decay_score(0.2, now - timedelta(days=1100), now=now)
    assert strong_old > weak_old
    # the floor keeps old memories in the SAME order of magnitude as fresh
    # ones (0.09 vs 0.19 here) — before the fix it was 1e-16 vs 0.19, i.e.
    # total exclusion from top-k
    fresh_weak, _ = time_decay_score(0.2, now - timedelta(days=1), now=now)
    assert strong_old > fresh_weak * 0.1


async def _empty_context(*_args, **_kwargs):
    return []


async def _identity_rewrite(*_args, **_kwargs):
    return {
        "rewritten_query": "trace probe",
        "entities": [],
        "reasoning": None,
        "_fallback_used": False,
    }


async def _zero_embedding(*_args, **_kwargs):
    return [0.0]


async def _failed_embedding(*_args, **_kwargs):
    await asyncio.sleep(0.001)
    raise RuntimeError("embedding unavailable")


async def _mismatched_embedding(*_args, **_kwargs):
    raise EmbeddingDimensionMismatch("collection contract is 384-dim, query is 1536-dim")


async def _empty_search(*_args, **_kwargs):
    return []


@pytest.mark.asyncio
async def test_trace_has_new_stage_keys_and_total(monkeypatch, barrier_outbox):
    # ``barrier_outbox``: recall's R14 freshness barrier reads the outbox
    # through its own sessionmaker; a real (empty) one keeps the count read
    # succeeding — see tests/retrieval/conftest.py.
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with AsyncSession(engine) as db:
        monkeypatch.setattr(retriever_module, "fetch_personal_context", _empty_context)
        monkeypatch.setattr(retriever_module, "rewrite_query", _identity_rewrite)
        monkeypatch.setattr(retriever_module, "embed_query", _zero_embedding)
        monkeypatch.setattr(retriever_module, "search_memories", _empty_search)
        response = await MemoryRetriever(db, uuid.uuid4()).recall(
            "trace probe", top_k=5, include_personal_context=True
        )
    await engine.dispose()
    required = {
        "context", "queue_wait", "embed_compute", "lexical", "rerank",
        "eligibility", "score", "serialization", "total",
    }
    assert required <= response.trace.stage_ms.keys()
    assert response.trace.stage_ms["total"] >= max(
        value for key, value in response.trace.stage_ms.items() if key != "total"
    )


@pytest.mark.asyncio
async def test_embed_failure_returns_empty_response_with_trace_timings(monkeypatch, barrier_outbox):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with AsyncSession(engine) as db:
        monkeypatch.setattr(retriever_module, "fetch_personal_context", _empty_context)
        monkeypatch.setattr(retriever_module, "rewrite_query", _identity_rewrite)
        monkeypatch.setattr(retriever_module, "embed_query", _failed_embedding)
        response = await MemoryRetriever(db, uuid.uuid4()).recall(
            "trace probe", top_k=5, include_personal_context=True
        )
    await engine.dispose()

    required = {
        "context", "queue_wait", "embed_compute", "lexical", "rerank",
        "eligibility", "score", "serialization", "total",
    }
    legacy = {"rewrite_ms", "embed_ms", "search_ms", "hydrate_ms"}
    assert response.results == []
    assert required | legacy <= response.trace.stage_ms.keys()
    assert response.trace.stage_ms["total"] > 0
    assert response.trace.stage_ms["embed_compute"] > 0
    assert response.trace.stage_ms["embed_ms"] > 0
    assert response.trace.rewrite_skipped is False


@pytest.mark.asyncio
async def test_embed_mismatch_propagates_instead_of_empty_response(monkeypatch, tmp_path, barrier_outbox):
    """A contract mismatch is a readiness failure — never flattened to empty."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 't7_scoring.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with AsyncSession(engine) as db:
        monkeypatch.setattr(retriever_module, "fetch_personal_context", _empty_context)
        monkeypatch.setattr(retriever_module, "rewrite_query", _identity_rewrite)
        monkeypatch.setattr(retriever_module, "embed_query", _mismatched_embedding)
        with pytest.raises(EmbeddingDimensionMismatch):
            await MemoryRetriever(db, uuid.uuid4()).recall(
                "trace probe", top_k=5, include_personal_context=True
            )
    await engine.dispose()
