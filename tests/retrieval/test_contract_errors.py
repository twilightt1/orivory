"""Task 7 — contract mismatch and vector outage are TYPED errors, not empty.

Pinned contract (P1a):
- memory ``search_memories`` raises ``VectorUnavailableError`` when the
  collection cannot be acquired (Chroma down) — never ``[]``, so "cannot
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
        raise ConnectionError("Chroma refused")

    monkeypatch.setattr(vector_store, "_get_collection", unavailable)
    with pytest.raises(VectorUnavailableError):
        await vector_store.search_memories([0.1] * 8, user_id=str(uuid.uuid4()))
