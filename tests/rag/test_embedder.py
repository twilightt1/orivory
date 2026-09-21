from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config import settings
from app.retrieval import embedder
from app.retrieval.embedder import embed_query, embed_texts, embed_texts_sync


@pytest.fixture(autouse=True)
def _dummy_provider_keys(monkeypatch):
    """Client construction requires keys but never touches the network:
    every test below replaces the transport (embeddings/post) with fakes."""
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(settings, "JINA_API_KEY", "test-key")
    # Local backend takes precedence when on (the zero-key default); these
    # tests target the keyed backends, so pin it off.
    monkeypatch.setattr(settings, "USE_LOCAL_EMBEDDINGS", False)


class _FakeAsyncEmbeddings:
    def __init__(self):
        self.calls: list[dict] = []

    async def create(self, model, input, encoding_format, timeout):
        self.calls.append(
            {
                "model": model,
                "input": input,
                "encoding_format": encoding_format,
                "timeout": timeout,
            }
        )
        data = [
            SimpleNamespace(embedding=[0.1, 0.2, 0.3]),
            SimpleNamespace(embedding=[0.4, 0.5, 0.6]),
        ][: len(input)]
        return SimpleNamespace(data=data)


class _FakeSyncEmbeddings:
    def __init__(self):
        self.calls: list[dict] = []

    def create(self, model, input, encoding_format, timeout):
        self.calls.append(
            {
                "model": model,
                "input": input,
                "encoding_format": encoding_format,
                "timeout": timeout,
            }
        )
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])


@pytest.mark.asyncio
async def test_embed_texts_async(monkeypatch):
    fake_embeddings = _FakeAsyncEmbeddings()
    monkeypatch.setattr(embedder.async_client, "embeddings", fake_embeddings)
    monkeypatch.setattr(settings, "USE_JINA_EMBEDDINGS", False)

    texts = ["hello", "world"]
    embeddings = await embed_texts(texts)

    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[1] == [0.4, 0.5, 0.6]
    assert fake_embeddings.calls[0]["input"] == texts
    assert fake_embeddings.calls[0]["encoding_format"] == "float"
    assert fake_embeddings.calls[0]["timeout"] == 30.0


@pytest.mark.asyncio
async def test_embed_query_async(monkeypatch):
    fake_embeddings = _FakeAsyncEmbeddings()
    monkeypatch.setattr(embedder.async_client, "embeddings", fake_embeddings)
    monkeypatch.setattr(settings, "USE_JINA_EMBEDDINGS", False)

    query = "search term"
    embedding = await embed_query(query)

    assert isinstance(embedding, list)
    assert embedding == [0.1, 0.2, 0.3]
    assert fake_embeddings.calls[0]["input"] == [query]


def test_embed_texts_sync(monkeypatch):
    fake_embeddings = _FakeSyncEmbeddings()
    monkeypatch.setattr(embedder.sync_client, "embeddings", fake_embeddings)
    monkeypatch.setattr(settings, "USE_JINA_EMBEDDINGS", False)

    texts = ["hello"]
    embeddings = embed_texts_sync(texts)

    assert len(embeddings) == 1
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert fake_embeddings.calls[0]["input"] == texts
    assert fake_embeddings.calls[0]["encoding_format"] == "float"


@pytest.mark.asyncio
async def test_embed_texts_jina(monkeypatch):
    """Test that Jina embeddings are used when configured."""
    # Mock the httpx async client
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]},
        ]
    }

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(return_value=mock_response)

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: mock_client)

    texts = ["hello", "world"]
    embeddings = await embedder._embed_with_jina(texts)

    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert embeddings[1] == [0.4, 0.5, 0.6]


def test_embed_texts_jina_sync(monkeypatch):
    """Test that Jina embeddings work synchronously."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3]},
        ]
    }

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=None)
    mock_client.post = MagicMock(return_value=mock_response)

    monkeypatch.setattr("httpx.Client", lambda **kwargs: mock_client)

    texts = ["hello"]
    embeddings = embedder._embed_sync_with_jina(texts)

    assert len(embeddings) == 1
    assert embeddings[0] == [0.1, 0.2, 0.3]
