"""
Unit tests for app/retrieval/memory/vector_store.py

Tests the ChromaDB-backed memory vector store functions.
"""
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

# Import the module to test
from app.retrieval.memory import vector_store


class TestRetryDecorator:
    """Tests for the _with_retry decorator."""

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_first_try(self):
        """Function should succeed on first try without retry."""
        call_count = 0

        @vector_store._with_retry(retries=3)
        async def flaky_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await flaky_function()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_succeeds_after_failures(self):
        """Function should succeed after transient failures."""
        call_count = 0

        @vector_store._with_retry(retries=3, base_delay=0.01)
        async def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Could not connect")
            return "success"

        result = await flaky_function()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_fails_after_max_retries(self):
        """Should raise after exhausting retries."""
        @vector_store._with_retry(retries=2, base_delay=0.01)
        async def always_fails():
            raise ConnectionError("Could not connect")

        with pytest.raises(ConnectionError):
            await always_fails()

    @pytest.mark.asyncio
    async def test_retry_raises_non_transient_errors(self):
        """Should raise immediately for non-transient errors."""
        @vector_store._with_retry(retries=3)
        async def bad_error():
            raise ValueError("Not a transient error")

        with pytest.raises(ValueError):
            await bad_error()

    def test_retry_sync_succeeds_on_first_try(self):
        """Sync function should succeed on first try."""
        call_count = 0

        @vector_store._with_retry(retries=3)
        def sync_flaky_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = sync_flaky_function()
        assert result == "success"
        assert call_count == 1

    def test_retry_sync_fails_after_max_retries(self):
        """Sync function should raise after exhausting retries."""
        @vector_store._with_retry(retries=2, base_delay=0.01)
        def sync_always_fails():
            raise ConnectionError("Could not connect")

        with pytest.raises(ConnectionError):
            sync_always_fails()


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
    """Tests for _memory_to_metadata helper."""

    def test_metadata_basic_fields(self):
        """Should include required metadata fields."""
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

        assert result["user_id"] == str(user_id)
        assert result["memory_id"] == str(memory_id)
        assert result["source_type"] == "manual_note"
        assert result["salience"] == 0.75
        assert result["pinned"] is False
        assert result["tags"] == ["tag1", "tag2"]

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

    def test_metadata_casts_types(self):
        """Should cast salience to float and pinned to bool."""
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
        # Empty/None tags are OMITTED (ChromaDB rejects empty-list metadata
        # values) — see _memory_to_metadata docstring.
        assert "tags" not in result

    def test_metadata_handles_none_tags(self):
        """None tags are omitted (ChromaDB rejects empty lists)."""
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


class TestCollectionName:
    """Tests for COLLECTION_NAME constant."""

    def test_collection_name_is_correct(self):
        """Collection name should be Orivory_memories."""
        assert vector_store.COLLECTION_NAME == "Orivory_memories"


class TestGetSyncClient:
    """Tests for _get_sync_client function."""

    def test_get_sync_client_returns_client(self):
        """_get_sync_client should return a ChromaDB sync client."""
        with patch.object(vector_store, "_sync_client", None):
            with patch("chromadb.HttpClient") as mock_client_class:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client

                client = vector_store._get_sync_client()

                assert client is mock_client
                mock_client_class.assert_called_once()

    def test_get_sync_client_caches_client(self):
        """_get_sync_client should cache the client after first call."""
        with patch.object(vector_store, "_sync_client", None):
            with patch("chromadb.HttpClient") as mock_client_class:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client

                client1 = vector_store._get_sync_client()
                client2 = vector_store._get_sync_client()

                assert client1 is client2
                assert mock_client_class.call_count == 1


class TestUpsertMemorySync:
    """Tests for upsert_memory_sync function."""

    def test_upsert_memory_sync_calls_collection(self):
        """Should call collection upsert with correct parameters."""
        mock_memory = MagicMock()
        mock_memory.id = uuid4()
        mock_memory.user_id = uuid4()
        mock_memory.title = "Test"
        mock_memory.content = "Content"
        mock_memory.source_type = "manual"
        mock_memory.captured_at = None
        mock_memory.salience = 0.5
        mock_memory.pinned = False
        mock_memory.tags = []

        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch.object(vector_store, "_get_sync_client", return_value=mock_client), \
             patch.object(vector_store, "embed_texts_sync", return_value=[[0.1] * 1536]):
            vector_store.upsert_memory_sync(mock_memory)

        mock_client.get_or_create_collection.assert_called_once()
        mock_collection.upsert.assert_called_once()


class TestDeleteMemoriesSync:
    """Tests for delete_memories_sync function."""

    def test_delete_memories_sync_calls_collection(self):
        """Should call collection delete with correct parameters."""
        memory_ids = [str(uuid4()) for _ in range(3)]

        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch.object(vector_store, "_get_sync_client", return_value=mock_client):
            vector_store.delete_memories_sync(memory_ids)

        mock_collection.delete.assert_called_once()
        call_kwargs = mock_collection.delete.call_args.kwargs
        assert "ids" in call_kwargs
        assert call_kwargs["ids"] == memory_ids

    def test_delete_memories_sync_empty_list(self):
        """Should handle empty list gracefully."""
        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch.object(vector_store, "_get_sync_client", return_value=mock_client):
            vector_store.delete_memories_sync([])

        # Should not call delete for empty list
        mock_collection.delete.assert_not_called()


class TestGetExistingMemoryIdsSync:
    """Tests for get_existing_memory_ids_sync function."""

    def test_get_existing_memory_ids_returns_set(self):
        """Should return a set of existing memory IDs."""
        memory_ids = [str(uuid4()) for _ in range(3)]

        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": [memory_ids[0], memory_ids[1]]
        }
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch.object(vector_store, "_get_sync_client", return_value=mock_client):
            result = vector_store.get_existing_memory_ids_sync(memory_ids)

        assert isinstance(result, set)
        assert len(result) == 2
        assert memory_ids[0] in result
        assert memory_ids[1] in result

    def test_get_existing_memory_ids_handles_missing(self):
        """Should handle case where no memories exist."""
        memory_ids = [str(uuid4())]

        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch.object(vector_store, "_get_sync_client", return_value=mock_client):
            result = vector_store.get_existing_memory_ids_sync(memory_ids)

        assert result == set()
