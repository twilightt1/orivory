"""Tests: semantic rerank wiring in MemoryRetriever (opt-in, graceful)."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from app.retrieval.memory import retriever as rmod
from app.retrieval.memory.retriever import MemoryRetriever


def _mem(user_id, mid, content, days_old):
    from app.models.memory import Memory

    return Memory(
        id=mid,
        user_id=user_id,
        title=content[:40],
        content=content,
        tags=[],
        salience=0.5,
        pinned=False,
        source_type="manual_note",
        recall_count=0,
        captured_at=datetime.now(UTC) - timedelta(days=days_old),
        indexed_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _candidate(mid, score):
    return {"memory_id": str(mid), "content": "c", "score": score}


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _FakeDB:
    def __init__(self, rows):
        self.rows = rows

    async def execute(self, _stmt):
        return _FakeResult(self.rows)


def _fake_vector_search(cands):
    async def _search(_embedding, *, user_id, top_k=10, where=None, namespace=None):
        return cands[:top_k]

    return _search


@pytest.fixture()
def recall_env(monkeypatch, barrier_outbox):
    """Wire a working recall path without any external service.

    The retriever takes user_id directly (no principal) — returns it for
    the tests to build the retriever with. ``barrier_outbox`` gives recall's
    R14 freshness barrier a real (empty) outbox to count: this suite fakes the
    retriever's DB, but the barrier reads its own — see tests/retrieval/conftest.py.
    """
    uid = uuid.uuid4()

    async def _fallback_rewrite(query, context=None, *, model=None):
        return {
            "rewritten_query": query,
            "entities": [],
            "reasoning": None,
            "_fallback_used": True,
        }

    async def _embed(_query):
        return [0.1, 0.2]

    monkeypatch.setattr(rmod, "rewrite_query", _fallback_rewrite)
    monkeypatch.setattr(rmod, "embed_query", _embed)
    monkeypatch.setattr(
        rmod, "fetch_personal_context", AsyncMock(return_value=[])
    )
    return uid


@pytest.mark.asyncio
async def test_rerank_reorders_before_top_k(recall_env, monkeypatch):
    """With semantic_rerank on, the cross-encoder's ordering decides which
    memories make top-k — even when raw cosine prefers the wrong ones."""
    uid = recall_env
    relevant = _mem(uid, uuid.uuid4(), "backpack ordered Jan 15 arrived 1/20", 1100)
    decoy = _mem(uid, uuid.uuid4(), "wireless mouse battery talk", 1)

    monkeypatch.setattr(
        rmod,
        "search_memories",
        _fake_vector_search(
            [_candidate(decoy.id, 0.66), _candidate(relevant.id, 0.59)]
        ),
    )

    async def _rerank(query, chunks, *, top_n=None):
        # cross-encoder judgment: the backpack chunk is the relevant one —
        # and it stamps rerank_score, which the scorer uses as semantic base
        for c in chunks:
            c["rerank_score"] = 0.9 if c["memory_id"] == str(relevant.id) else 0.2
        return sorted(chunks, key=lambda c: c["memory_id"] != str(relevant.id))

    monkeypatch.setattr("app.retrieval.reranker.rerank", _rerank)

    retr = MemoryRetriever(_FakeDB([relevant, decoy]), uid, semantic_rerank=True)
    resp = await retr.recall("when did my backpack arrive", top_k=1)
    ids = [str(m.id) for m in resp.results]
    assert ids == [str(relevant.id)], (
        "cross-encoder ordering must decide top-k over raw cosine"
    )


@pytest.mark.asyncio
async def test_rerank_off_keeps_vector_order(recall_env, monkeypatch):
    uid = recall_env
    decoy = _mem(uid, uuid.uuid4(), "mouse talk", 1)
    relevant = _mem(uid, uuid.uuid4(), "backpack facts", 1100)
    monkeypatch.setattr(
        rmod,
        "search_memories",
        _fake_vector_search(
            [_candidate(decoy.id, 0.66), _candidate(relevant.id, 0.59)]
        ),
    )
    retr = MemoryRetriever(_FakeDB([decoy, relevant]), uid, semantic_rerank=False)
    resp = await retr.recall("when did my backpack arrive", top_k=2)
    assert next(str(m.id) for m in resp.results) == str(decoy.id)


@pytest.mark.asyncio
async def test_rerank_failure_falls_back(recall_env, monkeypatch):
    """Reranker outage must never break recall — vector order continues."""
    uid = recall_env
    m1 = _mem(uid, uuid.uuid4(), "one", 5)
    m2 = _mem(uid, uuid.uuid4(), "two", 6)
    monkeypatch.setattr(
        rmod,
        "search_memories",
        _fake_vector_search([_candidate(m1.id, 0.7), _candidate(m2.id, 0.6)]),
    )

    async def _boom(query, chunks, *, top_n=None):
        raise RuntimeError("jina down")

    retr = MemoryRetriever(_FakeDB([m1, m2]), uid, semantic_rerank=True)
    monkeypatch.setattr("app.retrieval.reranker.rerank", _boom)
    resp = await retr.recall("query", top_k=2)
    assert len(resp.results) == 2


def test_default_flag_defers_to_settings(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "RETRIEVAL_SEMANTIC_RERANK", True)
    retr = MemoryRetriever(_FakeDB([]), uuid.uuid4())
    assert retr.semantic_rerank is True
    monkeypatch.setattr(settings, "RETRIEVAL_SEMANTIC_RERANK", False)
    retr2 = MemoryRetriever(_FakeDB([]), uuid.uuid4())
    assert retr2.semantic_rerank is False
