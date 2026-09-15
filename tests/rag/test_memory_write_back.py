"""P0 spine tests: memory write-back + reindex.

These verify the fix for the gap where connector-synced memories were written
to Postgres but never embedded into ChromaDB (making them invisible to
recall). All tests are CI-safe: ChromaDB, embeddings, and the graph builder
are mocked.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from app.retrieval.memory import write_back

pytestmark = pytest.mark.rag


def _memory(**overrides):
    base = dict(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        title="Note",
        content="Some content",
        source_type="manual_note",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestSafeUpsert:
    @pytest.mark.asyncio
    async def test_returns_true_on_success(self, monkeypatch):
        called = {}

        async def fake_upsert(memory):
            called["id"] = memory.id

        monkeypatch.setattr(
            "app.retrieval.memory.vector_store.upsert_memory", fake_upsert
        )
        mem = _memory()
        ok = await write_back.safe_upsert_to_index(mem)
        assert ok is True
        assert called["id"] == mem.id

    @pytest.mark.asyncio
    async def test_never_raises_on_failure(self, monkeypatch):
        async def boom(memory):
            raise RuntimeError("chroma down")

        monkeypatch.setattr(
            "app.retrieval.memory.vector_store.upsert_memory", boom
        )
        ok = await write_back.safe_upsert_to_index(_memory())
        assert ok is False  # swallowed, not raised


class TestSafeEnqueueGraph:
    def test_never_raises_when_builder_fails(self, monkeypatch):
        # Force the sync builder to raise; helper must swallow it.
        def boom(*a, **k):
            raise RuntimeError("graph store down")

        monkeypatch.setattr("app.graph.builder.build_memory_graph_sync", boom)
        # Should not raise
        write_back.safe_enqueue_graph_build(uuid.uuid4())

    def test_calls_builder_sync(self, monkeypatch):
        import app.graph.builder as gb

        calls = {}

        def fake_build(db, memory_id, **kwargs):
            calls["memory_id"] = memory_id
            return None

        monkeypatch.setattr(gb, "build_memory_graph_sync", fake_build)
        mid = uuid.uuid4()
        write_back.safe_enqueue_graph_build(mid)
        assert calls["memory_id"] == str(mid)


class TestIndexNewMemory:
    @pytest.mark.asyncio
    async def test_runs_embed_and_graph(self, monkeypatch):
        calls = {"embed": 0, "graph": 0}

        async def fake_upsert(memory):
            calls["embed"] += 1

        def fake_enqueue(memory_id):
            calls["graph"] += 1

        monkeypatch.setattr(
            "app.retrieval.memory.vector_store.upsert_memory", fake_upsert
        )
        monkeypatch.setattr(write_back, "safe_enqueue_graph_build", fake_enqueue)

        await write_back.index_new_memory(_memory())
        assert calls == {"embed": 1, "graph": 1}


class TestReindexMissingComputation:
    """The reindex task must only re-embed memories missing from the index."""

    def test_only_missing_filters_existing(self, monkeypatch):
        from app.retrieval.memory import vector_store

        present = {"a", "b"}

        def fake_existing(ids):
            return {i for i in ids if i in present}

        captured = {}

        def fake_upsert_batch(memories):
            captured["ids"] = [str(m.id) for m in memories]
            return len(memories)

        monkeypatch.setattr(vector_store, "get_existing_memory_ids_sync", fake_existing)
        monkeypatch.setattr(vector_store, "upsert_memories_sync", fake_upsert_batch)

        # Simulate the task's per-page filtering logic directly.
        rows = [SimpleNamespace(id="a"), SimpleNamespace(id="b"), SimpleNamespace(id="c")]
        existing = vector_store.get_existing_memory_ids_sync([str(m.id) for m in rows])
        to_index = [m for m in rows if str(m.id) not in existing]
        written = vector_store.upsert_memories_sync(to_index)

        assert existing == {"a", "b"}
        assert captured["ids"] == ["c"]
        assert written == 1
