"""Chroma-down degradation contract on the chat retrieval path.

Contract:
- vector_retriever.search raises VectorUnavailableError when Chroma itself
  is unreachable (client acquisition fails).
- Empty collection / no matching docs still return [] (genuinely no vectors).
- retrieval_agent sets state["vector_unavailable"]=True only when ALL
  vector calls failed AND no vector results came back; BM25 results are
  still used (Postgres-only degradation, never a silent total miss).
- The personal-memory path (memory.vector_store.search_memories) mirrors
  the same typed-outage contract; see TestMemorySearchAvailability.
"""
from types import SimpleNamespace

import pytest


class TestVectorUnavailableSignal:
    async def test_search_raises_when_chroma_unreachable(self, monkeypatch):
        from app.retrieval import vector_retriever

        async def boom():
            raise ConnectionError("refused")

        monkeypatch.setattr(vector_retriever, "_get_async_client", boom)
        with pytest.raises(vector_retriever.VectorUnavailableError):
            await vector_retriever.search("q", 5, "cid")

    async def test_search_empty_collection_still_returns_empty(self, monkeypatch):
        from app.retrieval import vector_retriever

        class FakeCollection:
            async def count(self):
                return 0

        class FakeClient:
            async def get_collection(self, name):
                return FakeCollection()

        async def fake_client():
            return FakeClient()

        monkeypatch.setattr(vector_retriever, "_get_async_client", fake_client)
        assert await vector_retriever.search("q", 5, "cid") == []


class TestMemorySearchAvailability:
    """The personal-memory path mirrors the document-chunk outage contract."""

    async def test_memory_search_raises_when_collection_acquisition_fails(self, monkeypatch):
        from app.retrieval import vector_retriever
        from app.retrieval.memory import vector_store

        async def boom(_dim):
            raise ConnectionError("refused")

        monkeypatch.setattr(vector_store, "_open_collection", boom)
        with pytest.raises(vector_retriever.VectorUnavailableError):
            await vector_store.search_memories([0.1] * 8, user_id="user-1")

    async def test_memory_search_empty_collection_still_returns_empty(self, monkeypatch):
        from app.retrieval.memory import vector_store

        class FakeClient:
            async def count(self, _generation):
                return SimpleNamespace(count=0)

        async def fake_collection(_dim):
            return FakeClient(), "generation", "f" * 64

        async def fake_info(*_args, **_kwargs):
            return {"dim": 8, "distance": "Cosine"}

        from app.retrieval import vector_backend

        monkeypatch.setattr(vector_store, "_open_collection", fake_collection)
        # The guard is called synchronously: this double must be sync too.
        monkeypatch.setattr(vector_store, "check_generation_contract", lambda *a, **k: None)
        monkeypatch.setattr(vector_backend, "collection_info_async", fake_info)
        assert await vector_store.search_memories([0.1] * 8, user_id="user-1") == []

