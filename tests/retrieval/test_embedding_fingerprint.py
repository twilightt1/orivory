import pytest

from app.config import settings
from app.retrieval import embedding_fingerprint as fp

FINGERPRINT_KEYS = {
    "model_id",
    "model_revision",
    "revision",
    "artifact_digest",
    "tokenizer_digest",
    "graph_outputs",
    "pooling",
    "query_prefix",
    "passage_prefix",
    "max_tokens",
    "truncation",
    "padding",
    "normalize",
    "dim",
    "precision",
    "provider",
    "doc_format",
}


def test_current_fingerprint_uses_the_arctic_cls_contract(monkeypatch):
    monkeypatch.setattr(settings, "USE_LOCAL_EMBEDDINGS", True)
    monkeypatch.setattr(settings, "LOCAL_EMBED_MODEL", "arctic")

    fingerprint = fp.current_fingerprint()

    assert set(fingerprint) == FINGERPRINT_KEYS
    assert fingerprint == fp.ARCTIC_CLS_FINGERPRINT
    assert fingerprint["pooling"] == "cls"
    assert fingerprint["model_revision"] == fp.ARCTIC_MODEL_REVISION
    # The mean-pooled contract stays importable for the ablation/rollback tool.
    assert fp.LEGACY_MEAN_FINGERPRINT["pooling"] == "legacy-mean"
    assert fingerprint != fp.LEGACY_MEAN_FINGERPRINT


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
                "model_revision": fp.E5_MODEL_REVISION,
                "revision": fp.E5_MODEL_REVISION,
                "artifact_digest": fp.E5_ARTIFACT_SHA256,
                "tokenizer_digest": fp.E5_TOKENIZER_SHA256,
                "graph_outputs": ["last_hidden_state"],
                "pooling": "mean",
                "query_prefix": "query: ",
                "passage_prefix": "passage: ",
                "max_tokens": 512,
                "truncation": "head",
                "padding": "batch-longest-zero",
                "normalize": True,
                "dim": 384,
                "precision": "float32",
                "provider": "onnxruntime-cpu",
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
                "model_revision": None,
                "revision": None,
                "artifact_digest": None,
                "tokenizer_digest": None,
                "graph_outputs": ["embedding"],
                "pooling": "api",
                "query_prefix": "",
                "passage_prefix": "",
                "max_tokens": None,
                "truncation": "provider-defined",
                "padding": "provider-defined",
                "normalize": None,
                "dim": 1024,
                "precision": "float32",
                "provider": "jina-api",
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
                "model_revision": None,
                "revision": None,
                "artifact_digest": None,
                "tokenizer_digest": None,
                "graph_outputs": ["embedding"],
                "pooling": "api",
                "query_prefix": "",
                "passage_prefix": "",
                "max_tokens": None,
                "truncation": "provider-defined",
                "padding": "provider-defined",
                "normalize": None,
                "dim": 1536,
                "precision": "float32",
                "provider": "openai-compatible-api",
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


def test_legacy_mean_fingerprint_stays_deterministic():
    """Same-dim contract swaps must still produce distinct cache keys."""
    assert fp.fingerprint_generation(fp.LEGACY_MEAN_FINGERPRINT) != fp.fingerprint_generation(
        fp.ARCTIC_CLS_FINGERPRINT
    )


def test_current_fingerprint_uses_configured_dimensions(monkeypatch):
    monkeypatch.setattr(settings, "USE_LOCAL_EMBEDDINGS", False)
    monkeypatch.setattr(settings, "USE_JINA_EMBEDDINGS", True)
    monkeypatch.setattr(settings, "JINA_API_KEY", "jina-test-key")
    monkeypatch.setattr(settings, "JINA_EMBED_DIMENSIONS", 768)
    assert fp.current_fingerprint()["dim"] == 768

    monkeypatch.setattr(settings, "USE_JINA_EMBEDDINGS", False)
    monkeypatch.setattr(settings, "EMBED_DIMENSIONS", 3072)
    assert fp.current_fingerprint()["dim"] == 3072


def test_cache_key_changes_with_pooling():
    a = dict(fp.LEGACY_MEAN_FINGERPRINT)
    b = dict(a, pooling="cls")
    assert fp.cache_key(a, "query", "hello") != fp.cache_key(b, "query", "hello")
    assert fp.cache_key(a, "query", "hello") == fp.cache_key(a, "query", "hello")
    assert fp.cache_key(a, "query", "hello") != fp.cache_key(a, "passage", "hello")
