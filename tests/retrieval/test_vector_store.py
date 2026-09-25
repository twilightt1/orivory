"""
Unit tests for app/retrieval/memory/vector_store.py

The store's backend is Qdrant (P1b): the payload contract, the point shape and
the client calls are pinned here with fakes; the real-store behaviour
(recall parity, filters, the manifest guard) lives in
``tests/retrieval/test_qdrant_parity.py``.
"""
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

# Import the module to test
from app.retrieval.memory import vector_store


class _FakeClient:
    """Records the calls the store makes through one face."""

    def __init__(self, *, count: int = 1, records=()) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._count = count
        self._records = list(records)

    def count(self, collection_name):
        self.calls.append(("count", {"collection_name": collection_name}))
        return SimpleNamespace(count=self._count)

    def upsert(self, *, collection_name, points):
        self.calls.append(("upsert", {"collection_name": collection_name, "points": points}))

    def delete(self, *, collection_name, points_selector):
        self.calls.append(("delete", {"collection_name": collection_name,
                                      "points_selector": points_selector}))

    def retrieve(self, *, collection_name, ids, with_payload=True):
        self.calls.append(("retrieve", {"collection_name": collection_name, "ids": list(ids),
                                        "with_payload": with_payload}))
        return [record for record in self._records if record.id in set(ids)]

    def query_points(self, **kwargs):
        self.calls.append(("query_points", kwargs))
        return SimpleNamespace(points=[])


def _sync_face(monkeypatch, client, *, generation="Orivory_memories", fingerprint="f" * 64,
               count=1):
    """Bind the store's sync seam to ``client`` (guard already satisfied)."""
    monkeypatch.setattr(
        vector_store,
        "_checked_collection_sync",
        lambda _dim: (client, generation, count),
    )
    monkeypatch.setattr(
        vector_store,
        "_open_collection_sync",
        lambda _dim: (client, generation, fingerprint),
    )


class TestMemoryToDocument:
    """Tests for _memory_to_document helper."""

    def test_memory_with_title(self):
        """Should prepend title to content."""
        mock_memory = MagicMock()
        mock_memory.title = "Test Title"
        mock_memory.content = "Test content body"

        result = vector_store._memory_to_document(mock_memory)

        assert result == "Title: Test Title\nTest content body"

    def test_memory_without_title(self):
        """Should return only content when title is None."""
        mock_memory = MagicMock()
        mock_memory.title = None
        mock_memory.content = "Just content"

        result = vector_store._memory_to_document(mock_memory)

        assert result == "Just content"

    def test_memory_with_empty_title(self):
        """Should return only content when title is empty."""
        mock_memory = MagicMock()
        mock_memory.title = ""
        mock_memory.content = "Content only"

        result = vector_store._memory_to_document(mock_memory)

        assert result == "Content only"


class TestMemoryToMetadata:
    """Tests for the Qdrant payload builder."""

    def test_metadata_basic_fields(self):
        """Should include the memory family's contract fields."""
        memory_id = uuid4()
        user_id = uuid4()

        mock_memory = MagicMock()
        mock_memory.id = memory_id
        mock_memory.user_id = user_id
        mock_memory.source_type = "manual_note"
        mock_memory.captured_at = None
        mock_memory.salience = 0.75
        mock_memory.pinned = False
        mock_memory.tags = ["tag1", "tag2"]

        result = vector_store._memory_to_metadata(mock_memory)

        assert result["kind"] == "memory"
        assert result["user_id"] == str(user_id)
        assert result["memory_id"] == str(memory_id)
        assert result["source_type"] == "manual_note"
        assert result["salience"] == 0.75
        assert result["pinned"] is False
        assert result["tags"] == ["tag1", "tag2"]
        # A MagicMock has no cm_* markers: the lifecycle state is "current".
        assert result["visibility_state"] == "current"

    def test_metadata_with_captured_at(self):
        """Should format captured_at as ISO string."""
        from datetime import UTC, datetime

        mock_memory = MagicMock()
        mock_memory.id = uuid4()
        mock_memory.user_id = uuid4()
        mock_memory.source_type = "web_clip"
        mock_memory.captured_at = datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC)
        mock_memory.salience = 0.5
        mock_memory.pinned = True
        mock_memory.tags = []

        result = vector_store._memory_to_metadata(mock_memory)

        assert result["captured_at"] == "2025-01-15T10:30:00+00:00"

    def test_metadata_omits_absent_values(self):
        """No nulls in a Qdrant payload: an absent value is an absent key."""
        mock_memory = MagicMock()
        mock_memory.id = uuid4()
        mock_memory.user_id = uuid4()
        mock_memory.source_type = "rss"
        mock_memory.captured_at = None
        mock_memory.salience = 0.9  # Already float
        mock_memory.pinned = True
        mock_memory.tags = None

        result = vector_store._memory_to_metadata(mock_memory)

        assert isinstance(result["salience"], float)
        assert isinstance(result["pinned"], bool)
        assert "tags" not in result
        assert "captured_at" not in result

    def test_metadata_handles_none_tags(self):
        mock_memory = MagicMock()
        mock_memory.id = uuid4()
        mock_memory.user_id = uuid4()
        mock_memory.source_type = "gmail"
        mock_memory.captured_at = None
        mock_memory.salience = 0.5
        mock_memory.pinned = False
        mock_memory.tags = None

        result = vector_store._memory_to_metadata(mock_memory)

        assert "tags" not in result

    def test_metadata_labels_the_lifecycle_state(self):
        from app.retrieval.memory.correction import CM_DERIVED_DIRTY, CM_SUPERSEDED_BY

        mock_memory = MagicMock()
        mock_memory.id = uuid4()
        mock_memory.user_id = uuid4()
        mock_memory.extra_metadata = {CM_SUPERSEDED_BY: str(uuid4())}
        mock_memory.tags = []

        assert vector_store._memory_to_metadata(mock_memory)["visibility_state"] == "superseded"

        mock_memory.extra_metadata = {CM_DERIVED_DIRTY: True}
        assert vector_store._memory_to_metadata(mock_memory)["visibility_state"] == "dirty"


class TestCollectionName:
    """Tests for COLLECTION_NAME constant."""

    def test_collection_name_is_correct(self):
        """Collection name should be Orivory_memories."""
        assert vector_store.COLLECTION_NAME == "Orivory_memories"


class TestMemoryUpsert:
    """Tests for asynchronous and synchronous memory upserts."""

    def test_upsert_memory_sync_writes_one_point(self, monkeypatch):
        memory = MagicMock()
        memory.id = uuid4()
        memory.user_id = uuid4()
        memory.title = "Test"
        memory.content = "Content"
        memory.source_type = "manual"
        memory.captured_at = None
        memory.salience = 0.5
        memory.pinned = False
        memory.tags = []
        memory.revision = 2

        client = _FakeClient()
        _sync_face(monkeypatch, client, generation="generation-a")
        monkeypatch.setattr(
            vector_store, "embed_texts_sync", lambda _texts: [[0.1] * 8]
        )

        vector_store.upsert_memory_sync(memory)

        # The contract check lives inside the patched seam; the point itself is
        # what this test pins.
        assert [name for name, _ in client.calls] == ["upsert"]
        upsert = client.calls[0][1]
        assert upsert["collection_name"] == "generation-a"
        (point,) = upsert["points"]
        assert point.id == str(memory.id)
        assert point.vector == [0.1] * 8
        # The payload is the contract plus the embedded document text.
        assert point.payload["content"] == "Title: Test\nContent"
        assert point.payload["kind"] == "memory"
        assert point.payload["orivory_memory_revision"] == 2


    async def test_precomputed_embedding_is_written_without_reembedding(self, monkeypatch):
        memory = MagicMock()
        memory.id = uuid4()
        memory.user_id = uuid4()
        memory.title = "Test"
        memory.content = "Content"
        memory.source_type = "manual"
        memory.captured_at = None
        memory.salience = 0.5
        memory.pinned = False
        memory.tags = []
        memory.revision = 2

        client = _FakeClient()
        _async_face(monkeypatch, client)

        async def should_not_embed(_texts):
            raise AssertionError("precomputed vector should avoid another model call")

        monkeypatch.setattr(vector_store, "embed_texts", should_not_embed)
        await vector_store.upsert_memory(memory, embedding=[0.3] * 8)

        upsert = next(kwargs for name, kwargs in client.calls if name == "upsert")
        (point,) = upsert["points"]
        assert point.vector == [0.3] * 8
        assert point.payload["content"] == "Title: Test\nContent"

    def test_upsert_memories_sync_batches_into_one_point_per_memory(self, monkeypatch):
        memories = [MagicMock(id=uuid4(), user_id=uuid4(), title=None, content=f"body {i}",
                              source_type="manual", captured_at=None, salience=0.5,
                              pinned=False, tags=[], revision=1) for i in range(3)]
        client = _FakeClient()
        _sync_face(monkeypatch, client)
        monkeypatch.setattr(vector_store, "embed_texts_sync", lambda texts: [[0.2] * 8 for _ in texts])

        assert vector_store.upsert_memories_sync(memories) == 3
        assert vector_store.upsert_memories_sync([]) == 0

        upsert = next(kwargs for name, kwargs in client.calls if name == "upsert")
        assert len(upsert["points"]) == 3
        assert [point.payload["content"] for point in upsert["points"]] == [
            "body 0", "body 1", "body 2",
        ]


class TestDeleteMemoriesSync:
    """Tests for delete_memories_sync function."""

    def test_delete_memories_sync_deletes_by_point_ids(self, monkeypatch):
        memory_ids = [str(uuid4()) for _ in range(3)]
        client = _FakeClient()
        _sync_face(monkeypatch, client)

        vector_store.delete_memories_sync(memory_ids)

        (name, kwargs), (rname, rkwargs) = client.calls
        assert name == "delete"
        assert list(kwargs["points_selector"].points) == memory_ids
        # M1: the batch is read back — and the readback asks about the ids it
        # just deleted, never a subset.
        assert rname == "retrieve"
        assert rkwargs["ids"] == memory_ids
        assert rkwargs["with_payload"] is False

    def test_delete_memories_sync_warns_about_a_survivor(self, monkeypatch, caplog):
        """M1: a missed purge is logged BY ID — the caller has no intent to retry."""
        memory_ids = [str(uuid4()) for _ in range(3)]
        client = _FakeClient(records=[SimpleNamespace(id=memory_ids[1])])
        _sync_face(monkeypatch, client)

        with caplog.at_level(logging.WARNING, logger="app.retrieval.memory.vector_store"):
            vector_store.delete_memories_sync(memory_ids)

        (record,) = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert record.getMessage() == "Memory delete not confirmed (sync)"
        assert record.memory_ids == [memory_ids[1]]
        assert record.still_present == 1

    def test_delete_memories_sync_empty_list(self, monkeypatch):
        client = _FakeClient()
        _sync_face(monkeypatch, client)

        vector_store.delete_memories_sync([])

        # Should not call the store for an empty list.
        assert client.calls == []

    def test_delete_memories_sync_never_raises(self, monkeypatch):
        def _boom(_dim):
            raise ConnectionError("qdrant down")

        monkeypatch.setattr(vector_store, "_open_collection_sync", _boom)

        vector_store.delete_memories_sync([str(uuid4())])  # best-effort by contract


class TestGetExistingMemoryIdsSync:
    """Tests for get_existing_memory_ids_sync function."""

    def test_get_existing_memory_ids_returns_set(self, monkeypatch):
        memory_ids = [str(uuid4()) for _ in range(3)]
        records = [SimpleNamespace(id=memory_ids[0]), SimpleNamespace(id=memory_ids[1])]
        client = _FakeClient(records=records)
        _sync_face(monkeypatch, client)

        result = vector_store.get_existing_memory_ids_sync(memory_ids)

        assert result == {memory_ids[0], memory_ids[1]}
        retrieve = next(kwargs for name, kwargs in client.calls if name == "retrieve")
        assert retrieve["ids"] == memory_ids
        assert retrieve["with_payload"] is False

    def test_get_existing_memory_ids_handles_missing(self, monkeypatch):
        client = _FakeClient(records=[])
        _sync_face(monkeypatch, client)

        assert vector_store.get_existing_memory_ids_sync([str(uuid4())]) == set()

    def test_get_existing_memory_ids_empty_list_short_circuits(self, monkeypatch):
        def _never(_dim):
            raise AssertionError("no store call for an empty id list")

        monkeypatch.setattr(vector_store, "_checked_collection_sync", _never)

        assert vector_store.get_existing_memory_ids_sync([]) == set()


class TestSearchMemories:
    """The item shape and the degradation contract of the read path."""

    async def test_search_maps_points_to_items(self, monkeypatch):
        point_id = str(uuid4())
        client = _FakeClient(count=2)
        client.query_points = lambda **kwargs: SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="point-b",
                    score=0.5,
                    payload={"memory_id": point_id, "content": "Title: T\nbody", "kind": "memory"},
                )
            ]
        )
        _async_face(monkeypatch, client)

        hits = await vector_store.search_memories([0.1] * 8, user_id="owner")

        assert hits == [
            {
                "memory_id": point_id,
                "content": "Title: T\nbody",
                "score": 0.5,
                "metadata": {"memory_id": point_id, "kind": "memory"},
                "rank": 0,
                "source": "vector",
            }
        ]

    async def test_search_returns_empty_for_an_empty_generation(self, monkeypatch):
        client = _FakeClient(count=0)
        _async_face(monkeypatch, client)

        assert await vector_store.search_memories([0.1] * 8, user_id="owner") == []
        assert [name for name, _ in client.calls] == ["count"]


def _async_face(monkeypatch, client) -> None:
    """Bind the store's async seam to a sync fake.

    ``_SyncAsAsync`` is the same adapter the local face uses in production, so
    the fake records exactly the calls the real client would receive. The
    contract guard has its own tests (``test_qdrant_parity``): here it is a
    no-op so each test pins one thing.
    """
    from app.retrieval import vector_backend
    from app.retrieval.vector_backend import _SyncAsAsync

    async def _open(_dim):
        return _SyncAsAsync(client), "generation-a", "f" * 64

    async def _info(*_args, **_kwargs):
        return {"dim": 8, "distance": "Cosine"}

    monkeypatch.setattr(vector_store, "_open_collection", _open)
    monkeypatch.setattr(vector_backend, "collection_info_async", _info)
    monkeypatch.setattr(vector_store, "check_generation_contract", lambda *a, **k: None)
