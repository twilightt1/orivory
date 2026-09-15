"""Task 7 — contract mismatch and vector outage are TYPED errors, not empty.

Pinned contract (P1a):
- memory ``search_memories`` raises ``VectorUnavailableError`` when the
  collection cannot be acquired **or when the count/query calls themselves
  fail** (Chroma dying after acquisition) — never ``[]``, so "cannot
  answer" is never served as a false no-match.
- ``recall`` lets the typed errors escape (generic failures still degrade to
  an empty response with a trace).
- The API answers 503 with a typed body instead of an unhandled 500 or a
  silent empty 200: ``{"error": "embedding_contract_mismatch"}`` and
  ``{"error": "vector_unavailable"}``.
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base, get_db
from app.main import app
from app.retrieval.embedder import EmbeddingDimensionMismatch
from app.retrieval.embedding_fingerprint import (
    canonical_fingerprint,
    current_fingerprint,
    fingerprint_generation,
)
from app.retrieval.memory import retriever as retriever_module
from app.retrieval.memory import vector_store
from app.retrieval.vector_retriever import VectorUnavailableError
from app.utils.dependencies import enforce_llm_quota, get_current_verified_user


async def _empty_context(*_args, **_kwargs):
    return []


async def _identity_rewrite(*_args, **_kwargs):
    return {
        "rewritten_query": "trace probe",
        "entities": [],
        "reasoning": None,
        "_fallback_used": False,
    }


async def _mismatched_embedding(*_args, **_kwargs):
    raise EmbeddingDimensionMismatch("collection contract is 384-dim, query is 1536-dim")


async def _zero_embedding(*_args, **_kwargs):
    return [0.0] * 8


@asynccontextmanager
async def _recall_client(tmp_path):
    """App client on a real temp-SQLite DB, with auth/quota/DB overridden."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 't7_contract.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def _db_override():
        async with sessions() as session:
            yield session

    app.dependency_overrides[get_current_verified_user] = lambda: SimpleNamespace(
        id=uuid.uuid4()
    )
    app.dependency_overrides[enforce_llm_quota] = lambda: None
    app.dependency_overrides[get_db] = _db_override
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
        await engine.dispose()


async def test_recall_endpoint_returns_503_on_contract_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(retriever_module, "fetch_personal_context", _empty_context)
    monkeypatch.setattr(retriever_module, "rewrite_query", _identity_rewrite)
    monkeypatch.setattr(retriever_module, "embed_query", _mismatched_embedding)

    async with _recall_client(tmp_path) as client:
        response = await client.post(
            "/api/v1/memories/recall", json={"query": "trace probe"}
        )

    assert response.status_code == 503
    assert response.json() == {"error": "embedding_contract_mismatch"}


async def test_recall_endpoint_returns_503_on_vector_unavailable(tmp_path, monkeypatch):
    async def unavailable(*_args, **_kwargs):
        raise VectorUnavailableError("ChromaDB unreachable")

    monkeypatch.setattr(retriever_module, "fetch_personal_context", _empty_context)
    monkeypatch.setattr(retriever_module, "rewrite_query", _identity_rewrite)
    monkeypatch.setattr(retriever_module, "embed_query", _zero_embedding)
    monkeypatch.setattr(retriever_module, "search_memories", unavailable)

    async with _recall_client(tmp_path) as client:
        response = await client.post(
            "/api/v1/memories/recall", json={"query": "trace probe"}
        )

    assert response.status_code == 503
    assert response.json() == {"error": "vector_unavailable"}


async def test_search_raises_when_collection_unavailable(monkeypatch):
    """Acquire failure is a typed outage (a readiness signal), not []."""
    async def unavailable():
        raise ConnectionError("Qdrant refused")

    monkeypatch.setattr(vector_store, "_open_collection", lambda _dim: unavailable())
    with pytest.raises(VectorUnavailableError):
        await vector_store.search_memories([0.1] * 8, user_id=str(uuid.uuid4()))


def _current_contract() -> dict:
    """Generation info that satisfies the real embedding-contract guard."""
    canonical = canonical_fingerprint(current_fingerprint())
    return {
        "info": {"dim": int(current_fingerprint()["dim"]), "distance": "Cosine"},
        "manifest_fingerprint": fingerprint_generation(canonical),
    }


def _fake_store(monkeypatch, *, count: object = 4, query_error: Exception | None = None):
    """A fake Qdrant face with a count and a query step, no real store."""
    from app.retrieval import vector_backend

    contract = _current_contract()

    class Client:
        async def count(self, _generation):
            if isinstance(count, Exception):
                raise count
            return SimpleNamespace(count=count)

        async def query_points(self, **_kwargs):
            if query_error is not None:
                raise query_error
            return SimpleNamespace(points=[])

    monkeypatch.setattr(
        vector_store,
        "_open_collection",
        lambda _dim: _async_result((Client(), "generation", contract["manifest_fingerprint"])),
    )
    monkeypatch.setattr(
        vector_backend,
        "collection_info_async",
        lambda *_args, **_kwargs: _async_result(contract["info"]),
    )
    return contract


async def _async_result(value):
    return value


async def test_search_raises_when_count_fails_after_acquisition(monkeypatch):
    """Qdrant dying between acquire and count must not read as no-match."""
    _fake_store(monkeypatch, count=ConnectionError("store died after acquisition"))
    with pytest.raises(VectorUnavailableError) as excinfo:
        await vector_store.search_memories([0.1] * 8, user_id="user-1")
    assert "count" in str(excinfo.value)


async def test_search_raises_when_query_fails_after_count(monkeypatch):
    """A query failure is typed even when count() and the contract guard pass."""
    contract = _fake_store(monkeypatch, query_error=ConnectionError("store died mid-query"))
    with pytest.raises(VectorUnavailableError) as excinfo:
        await vector_store.search_memories(
            [0.1] * contract["info"]["dim"], user_id="user-1"
        )
    assert "query" in str(excinfo.value)


async def test_search_keeps_contract_mismatch_typed(monkeypatch):
    """A stale generation contract stays EmbeddingDimensionMismatch, never the
    availability error the count/query guards raise."""
    contract = _fake_store(monkeypatch)
    contract["info"]["dim"] = int(contract["info"]["dim"]) + 1

    with pytest.raises(EmbeddingDimensionMismatch):
        await vector_store.search_memories([0.1] * 8, user_id="user-1")


async def test_search_mismatch_raised_by_count_is_not_retyped(monkeypatch):
    """Ordering pin: EmbeddingDimensionMismatch passes the count guard unchanged."""
    _fake_store(monkeypatch, count=EmbeddingDimensionMismatch("contract mismatch mid-count"))
    with pytest.raises(EmbeddingDimensionMismatch):
        await vector_store.search_memories([0.1] * 8, user_id="user-1")
