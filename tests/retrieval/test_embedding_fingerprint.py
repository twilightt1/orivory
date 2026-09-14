import pytest

from app.config import settings
from app.retrieval import embedding_fingerprint as fp

FINGERPRINT_KEYS = {
    "model_id",
    "pooling",
    "query_prefix",
    "passage_prefix",
    "max_tokens",
    "normalize",
    "dim",
    "doc_format",
}


def test_current_fingerprint_preserves_arctic_legacy_contract(monkeypatch):
    monkeypatch.setattr(settings, "USE_LOCAL_EMBEDDINGS", True)
    monkeypatch.setattr(settings, "LOCAL_EMBED_MODEL", "arctic")

    fingerprint = fp.current_fingerprint()

    assert set(fingerprint) == FINGERPRINT_KEYS
    assert fingerprint == fp.LEGACY_MEAN_FINGERPRINT


@pytest.mark.parametrize(
    ("settings_overrides", "expected"),
    [
        (
            {
                "USE_LOCAL_EMBEDDINGS": True,
                "LOCAL_EMBED_MODEL": "e5",
                "USE_JINA_EMBEDDINGS": True,
                "JINA_API_KEY": "jina-test-key",
            },
            {
                "model_id": "Xenova/multilingual-e5-small",
                "pooling": "mean",
                "query_prefix": "query: ",
                "passage_prefix": "passage: ",
                "max_tokens": 512,
                "normalize": True,
                "dim": 384,
                "doc_format": "title-content-v1",
            },
        ),
        (
            {
                "USE_LOCAL_EMBEDDINGS": False,
                "USE_JINA_EMBEDDINGS": True,
                "JINA_API_KEY": "jina-test-key",
                "JINA_EMBED_MODEL": "jina-embeddings-v4-text-small",
            },
            {
                "model_id": "jina-embeddings-v4-text-small",
                "pooling": "api",
                "query_prefix": "",
                "passage_prefix": "",
                "max_tokens": 512,
                "normalize": True,
                "dim": 1024,
                "doc_format": "title-content-v1",
            },
        ),
        (
            {
                "USE_LOCAL_EMBEDDINGS": False,
                "USE_JINA_EMBEDDINGS": False,
                "EMBED_MODEL": "text-embedding-3-large",
            },
            {
                "model_id": "text-embedding-3-large",
                "pooling": "api",
                "query_prefix": "",
                "passage_prefix": "",
                "max_tokens": 512,
                "normalize": True,
                "dim": 1536,
                "doc_format": "title-content-v1",
            },
        ),
    ],
)
def test_current_fingerprint_uses_active_embedding_contract(
    monkeypatch, settings_overrides, expected
):
    for name, value in settings_overrides.items():
        monkeypatch.setattr(settings, name, value)

    fingerprint = fp.current_fingerprint()

    assert set(fingerprint) == FINGERPRINT_KEYS
    assert fingerprint == expected


def test_cache_key_changes_with_pooling():
    a = dict(fp.LEGACY_MEAN_FINGERPRINT)
    b = dict(a, pooling="cls")
    assert fp.cache_key(a, "query", "hello") != fp.cache_key(b, "query", "hello")
    assert fp.cache_key(a, "query", "hello") == fp.cache_key(a, "query", "hello")
    assert fp.cache_key(a, "query", "hello") != fp.cache_key(a, "passage", "hello")
