"""Chroma-down degradation contract on the chat retrieval path.

Contract:
- vector_retriever.search raises VectorUnavailableError when Chroma itself
  is unreachable (client acquisition fails).
- Empty collection / no matching docs still return [] (genuinely no vectors).
- retrieval_agent sets state["vector_unavailable"]=True only when ALL
  vector calls failed AND no vector results came back; BM25 results are
  still used (Postgres-only degradation, never a silent total miss).
"""
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


class TestVectorUnavailableFlag:
    def test_all_failed_no_results_is_unavailable(self):
        from app.agents.retrieval_agent import _vector_unavailable
        from app.retrieval.vector_retriever import VectorUnavailableError

        assert _vector_unavailable([VectorUnavailableError("x")], []) is True

    def test_empty_without_errors_is_genuine_no_results(self):
        from app.agents.retrieval_agent import _vector_unavailable

        assert _vector_unavailable([], []) is False

    def test_partial_success_is_available(self):
        from app.agents.retrieval_agent import _vector_unavailable
        from app.retrieval.vector_retriever import VectorUnavailableError

        chunk = {"content": "c", "score": 0.9}
        assert _vector_unavailable([VectorUnavailableError("x"), [chunk]], [chunk]) is False
