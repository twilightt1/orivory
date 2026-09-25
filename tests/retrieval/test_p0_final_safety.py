"""Final P0 regression tests for contract, authorization, and filter safety."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.retrieval import embedder
from app.retrieval.embedder import EmbeddingDimensionMismatch


class _SyncCollection:
    def __init__(self, metadata, count=0, modify_error=None):
        self.metadata = dict(metadata)
        self._count = count
        self.modify_error = modify_error
        self.modified = None

    def count(self):
        return self._count

    def modify(self, *, metadata):
        if self.modify_error:
            raise self.modify_error
        self.modified = metadata
        self.metadata = metadata


class _AsyncCollection:
    """Async Chroma-like collection (the chunk path's stamp guard)."""

    def __init__(self, metadata, count=0, modify_error=None):
        self.metadata = dict(metadata)
        self._count = count
        self.modify_error = modify_error

    async def count(self):
        return self._count

    async def modify(self, *, metadata):
        if self.modify_error:
            raise self.modify_error
        self.metadata = metadata


def test_populated_unstamped_collection_is_quarantined():
    collection = _SyncCollection({"hnsw:space": "cosine"}, count=1)

    with pytest.raises(EmbeddingDimensionMismatch, match="populated"):
        embedder.stamp_collection_dim(
            collection,
            384,
            backend="local-e5",
            fingerprint="contract-v1",
        )

    assert collection.modified is None


@pytest.mark.asyncio
async def test_async_stamp_failure_blocks_contract_write():
    collection = _AsyncCollection({}, count=0, modify_error=OSError("metadata unavailable"))

    with pytest.raises(EmbeddingDimensionMismatch, match="persist"):
        await embedder.astamp_collection_dim(
            collection,
            384,
            backend="local-e5",
            fingerprint="contract-v1",
        )


@pytest.mark.asyncio
async def test_async_stamp_accepts_matching_populated_collection():
    collection = _AsyncCollection(
        {
            "orivory_embed_backend": "local-e5",
            "orivory_embed_dim": 384,
            "orivory_embed_fingerprint": "contract-v1",
        },
        count=1,
    )

    await embedder.astamp_collection_dim(
        collection,
        384,
        backend="local-e5",
        fingerprint="contract-v1",
    )


def test_sync_stamp_failure_blocks_contract_write():
    collection = _SyncCollection({}, count=0, modify_error=OSError("metadata unavailable"))

    with pytest.raises(EmbeddingDimensionMismatch, match="persist"):
        embedder.stamp_collection_dim(
            collection,
            384,
            backend="local-e5",
            fingerprint="contract-v1",
        )


def test_stamp_does_not_rewrite_chroma_distance_metadata():
    class ChromaLike(_SyncCollection):
        def modify(self, *, metadata):
            assert not any(key.startswith("hnsw:") for key in metadata)
            super().modify(metadata=metadata)

    collection = ChromaLike({"hnsw:space": "cosine"}, count=0)
    embedder.stamp_collection_dim(
        collection,
        384,
        backend="local-e5",
        fingerprint="contract-v1",
    )
    assert collection.modified is not None
    assert collection.modified[embedder.EMBED_FINGERPRINT_META_KEY] == "contract-v1"


def test_silent_metadata_write_failure_blocks_vector_write():
    class SilentCollection(_SyncCollection):
        def modify(self, *, metadata):
            self.modified = metadata

    collection = SilentCollection({}, count=0)
    with pytest.raises(EmbeddingDimensionMismatch, match="verified"):
        embedder.stamp_collection_dim(
            collection,
            384,
            backend="local-e5",
            fingerprint="contract-v1",
        )


@pytest.mark.asyncio
async def test_write_back_propagates_contract_failure(monkeypatch):
    from app.retrieval.memory import write_back

    async def fail(_memory):
        raise EmbeddingDimensionMismatch("contract mismatch")

    monkeypatch.setattr("app.retrieval.memory.vector_store.upsert_memory", fail)
    with pytest.raises(EmbeddingDimensionMismatch, match="contract mismatch"):
        await write_back.safe_upsert_to_index(SimpleNamespace(id=uuid4(), user_id=uuid4()))


@pytest.mark.asyncio
async def test_async_search_quarantines_populated_unstamped_collection(monkeypatch):
    """The P0 law on the Qdrant side: a populated generation whose manifest
    row is gone (``manifest_fingerprint is None``) is quarantined, never
    served as an empty result."""
    from app.retrieval import vector_backend
    from app.retrieval.memory import vector_store

    class Client:
        async def count(self, _generation):
            return SimpleNamespace(count=1)

    monkeypatch.setattr(
        vector_store, "_open_collection", AsyncMock(return_value=(Client(), "generation", None))
    )
    monkeypatch.setattr(
        vector_backend,
        "collection_info_async",
        AsyncMock(return_value={"dim": 384, "distance": "Cosine"}),
    )
    with pytest.raises(EmbeddingDimensionMismatch, match="populated"):
        await vector_store.search_memories(
            [0.1] * 384,
            user_id="owner",
        )


@pytest.mark.asyncio
async def test_recall_propagates_contract_mismatch(monkeypatch, barrier_outbox):
    # ``barrier_outbox``: the R14 freshness barrier reads its own outbox on
    # every recall (tests/retrieval/conftest.py); this test's retriever DB is
    # a ``SimpleNamespace``, but the barrier's is real.
    from app.retrieval.memory import retriever as retriever_module
    from app.retrieval.memory.retriever import MemoryRetriever

    uid = uuid4()
    monkeypatch.setattr(retriever_module, "fetch_personal_context", AsyncMock(return_value=[]))

    async def embed(_query):
        return [0.1, 0.2]

    async def search(*_args, **_kwargs):
        raise EmbeddingDimensionMismatch("fresh reindex required")

    monkeypatch.setattr(retriever_module, "embed_query", embed)
    monkeypatch.setattr(retriever_module, "search_memories", search)

    with pytest.raises(EmbeddingDimensionMismatch, match="fresh reindex"):
        await MemoryRetriever(SimpleNamespace(), uid).recall(
            "query",
            include_personal_context=False,
        )


@pytest.mark.asyncio
async def test_remote_rerank_sees_only_current_sql_owned_content(monkeypatch, barrier_outbox):
    from datetime import UTC, datetime

    from app.models.memory import Memory
    from app.retrieval.memory import retriever as retriever_module
    from app.retrieval.memory.retriever import MemoryRetriever

    owner = uuid4()
    authorized = Memory(
        id=uuid4(),
        user_id=owner,
        parent_id=None,
        title="Current title",
        content="current SQL content",
        summary=None,
        tags=[],
        salience=0.5,
        pinned=False,
        recall_count=0,
        last_used_at=None,
        source_type="manual_note",
        source_ref=None,
        source_url=None,
        captured_at=datetime.now(UTC),
        indexed_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    authorized_two = Memory(
        id=uuid4(),
        user_id=owner,
        parent_id=None,
        title="Second title",
        content="second current SQL content",
        summary=None,
        tags=[],
        salience=0.5,
        pinned=False,
        recall_count=0,
        last_used_at=None,
        source_type="manual_note",
        source_ref=None,
        source_url=None,
        captured_at=datetime.now(UTC),
        indexed_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    foreign_id = uuid4()
    seen_chunks = []

    class Result:
        def scalars(self):
            return self

        def all(self):
            return [authorized, authorized_two]

    class DB:
        async def execute(self, _statement):
            return Result()

    async def rewrite(query, context=None):
        return {
            "rewritten_query": query,
            "entities": [],
            "reasoning": None,
            "_fallback_used": True,
        }

    async def embed(_query):
        return [0.1, 0.2]

    async def search(*_args, **_kwargs):
        return [
            {
                "memory_id": str(foreign_id),
                "content": "FOREIGN SECRET FROM VECTOR PAYLOAD",
                "score": 0.99,
            },
            {
                "memory_id": str(authorized.id),
                "content": "STALE VECTOR CONTENT",
                "score": 0.90,
            },
            {
                "memory_id": str(authorized_two.id),
                "content": "STALE SECOND VECTOR CONTENT",
                "score": 0.80,
            },
        ]

    async def rerank(_query, chunks, *, top_n=None):
        seen_chunks.extend(chunks)
        return chunks

    monkeypatch.setattr(retriever_module, "fetch_personal_context", AsyncMock(return_value=[]))
    monkeypatch.setattr(retriever_module, "rewrite_query", rewrite)
    monkeypatch.setattr(retriever_module, "embed_query", embed)
    monkeypatch.setattr(retriever_module, "search_memories", search)
    monkeypatch.setattr("app.retrieval.reranker.rerank", rerank)

    response = await MemoryRetriever(DB(), owner, semantic_rerank=True).recall(
        "query",
        top_k=2,
        include_personal_context=False,
    )

    assert {str(memory.id) for memory in response.results} == {
        str(authorized.id),
        str(authorized_two.id),
    }
    assert len(seen_chunks) == 2
    assert {chunk["content"] for chunk in seen_chunks} == {
        "Title: Current title\ncurrent SQL content",
        "Title: Second title\nsecond current SQL content",
    }
    assert "FOREIGN SECRET" not in repr(seen_chunks)


@pytest.mark.asyncio
async def test_tenant_filter_is_immutable_and_allowlisted(monkeypatch):
    """The tenant clause is always the first MUST of the store's own filter,
    and a caller's ``where`` can never replace it."""
    from qdrant_client import models as qm

    from app.retrieval import vector_backend
    from app.retrieval.memory import vector_store

    query_calls: list[dict] = []

    class Client:
        async def count(self, _generation):
            return SimpleNamespace(count=1)

        async def query_points(self, **kwargs):
            query_calls.append(kwargs)
            return SimpleNamespace(points=[])

    monkeypatch.setattr(
        vector_store, "_open_collection", AsyncMock(return_value=(Client(), "generation", "f" * 64))
    )
    monkeypatch.setattr(
        vector_backend,
        "collection_info_async",
        AsyncMock(return_value={"dim": 384, "distance": "Cosine"}),
    )
    monkeypatch.setattr(vector_store, "check_generation_contract", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="user_id"):
        await vector_store.search_memories(
            [0.1] * 384,
            user_id="owner",
            where={"user_id": {"$eq": "attacker"}},
        )

    await vector_store.search_memories(
        [0.1] * 384,
        user_id="owner",
        where={"source_type": {"$eq": "manual_note"}},
    )
    sent_filter = query_calls[-1]["query_filter"]
    assert sent_filter.must[0] == qm.FieldCondition(
        key="user_id", match=qm.MatchValue(value="owner")
    )
    assert sent_filter.must[1] == qm.FieldCondition(
        key="source_type", match=qm.MatchValue(value="manual_note")
    )


def test_only_missing_rejects_contract_mismatch(monkeypatch):
    """``only_missing`` reindex trusts presence only under a matching contract."""
    from app.retrieval import vector_backend
    from app.retrieval.memory import vector_store

    class Client:
        def count(self, _generation):
            return SimpleNamespace(count=1)

    monkeypatch.setattr(
        vector_store,
        "_open_collection_sync",
        lambda _dim: (Client(), "generation", "old-contract"),
    )
    monkeypatch.setattr(
        vector_backend,
        "collection_info",
        lambda *_args, **_kwargs: {"dim": 384, "distance": "Cosine"},
    )

    with pytest.raises(EmbeddingDimensionMismatch, match="different embedding contract"):
        vector_store.get_existing_memory_ids_sync([str(uuid4())])


def test_contract_generation_mismatch_is_not_treated_as_present():
    collection = _SyncCollection(
        {
            "orivory_embed_backend": "local-e5",
            "orivory_embed_dim": 384,
            "orivory_embed_fingerprint": "contract-v1",
            "orivory_embed_generation": "wrong-generation",
        },
        count=1,
    )

    with pytest.raises(EmbeddingDimensionMismatch, match="generation"):
        embedder.check_collection_dim(
            collection,
            384,
            backend="local-e5",
            fingerprint="contract-v1",
        )


def test_memory_payload_carries_embedding_provenance(monkeypatch):
    from datetime import UTC, datetime

    from app.retrieval.memory import vector_store

    fingerprint = {
        "model_id": "model",
        "model_revision": "revision-1",
        "dim": 384,
        "provider": "test",
    }
    monkeypatch.setattr(vector_store, "current_fingerprint", lambda: fingerprint)
    memory = SimpleNamespace(
        id=uuid4(),
        user_id=uuid4(),
        source_type="manual_note",
        captured_at=datetime.now(UTC),
        salience=0.5,
        pinned=False,
        tags=[],
        extra_metadata={},
        revision=7,
    )

    metadata = vector_store._memory_to_metadata(memory)

    assert metadata["orivory_embed_fingerprint"]
    assert metadata["orivory_embed_model_revision"] == "revision-1"
    assert metadata["orivory_embed_dim"] == 384
    assert metadata["orivory_memory_revision"] == 7
    assert "metadata" not in metadata


def test_fingerprint_represents_arctic_cls_and_configured_dimensions(monkeypatch):
    from app.config import settings
    from app.retrieval import embedding_fingerprint as fingerprint_module

    monkeypatch.setattr(settings, "USE_LOCAL_EMBEDDINGS", True)
    monkeypatch.setattr(settings, "LOCAL_EMBED_MODEL", "arctic")
    arctic = fingerprint_module.current_fingerprint()
    assert arctic == fingerprint_module.ARCTIC_CLS_FINGERPRINT
    assert arctic["model_id"] == "Snowflake/snowflake-arctic-embed-xs"
    assert arctic["provider"] == "onnxruntime-cpu"
    assert arctic["dim"] == 384
    assert arctic["pooling"] == "cls"
    assert arctic["artifact_digest"]

    monkeypatch.setattr(settings, "USE_LOCAL_EMBEDDINGS", False)
    monkeypatch.setattr(settings, "EMBED_DIMENSIONS", 768)
    assert fingerprint_module.current_fingerprint()["dim"] == 768
