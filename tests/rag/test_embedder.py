from types import SimpleNamespace

import pytest

from app.config import settings
from app.retrieval import embedder
from app.retrieval.embedder import embed_query, embed_texts, embed_texts_sync


@pytest.fixture(autouse=True)
def _dummy_provider_keys(monkeypatch):
    """Client construction requires an API key but never touches the network:
    tests below replace embeddings calls with fakes; local is disabled because
    these tests cover only the legacy OpenAI-compatible fallback."""
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")
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
async def test_embed_texts_stays_local_when_openai_key_exists(monkeypatch):
    monkeypatch.setattr(settings, "USE_LOCAL_EMBEDDINGS", True)
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        embedder,
        "_embed_with_local",
        lambda texts, *, query=False: [[0.1, 0.2, 0.3] for _ in texts],
    )
    monkeypatch.setattr(
        embedder,
        "_get_async_client",
        lambda: pytest.fail("local default must not construct a remote client"),
    )

    assert await embed_texts(["hello"]) == [[0.1, 0.2, 0.3]]


@pytest.mark.asyncio
async def test_embed_texts_async(monkeypatch):
    fake_embeddings = _FakeAsyncEmbeddings()
    monkeypatch.setattr(embedder.async_client, "embeddings", fake_embeddings)

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

    query = "search term"
    embedding = await embed_query(query)

    assert isinstance(embedding, list)
    assert embedding == [0.1, 0.2, 0.3]
    assert fake_embeddings.calls[0]["input"] == [query]


def test_embed_texts_sync(monkeypatch):
    fake_embeddings = _FakeSyncEmbeddings()
    monkeypatch.setattr(embedder.sync_client, "embeddings", fake_embeddings)

    texts = ["hello"]
    embeddings = embed_texts_sync(texts)

    assert len(embeddings) == 1
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert fake_embeddings.calls[0]["input"] == texts
    assert fake_embeddings.calls[0]["encoding_format"] == "float"
