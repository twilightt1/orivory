"""Regression tests: embedding dimension guard (found in full-repo review, HIGH).

`embed_texts` dispatches between 384-dim local, 1024-dim Jina and 1536-dim
OpenAI with only a code comment as guard ("do NOT mix backends in one
store"). Flipping USE_JINA_EMBEDDINGS / USE_LOCAL_EMBEDDINGS after data
exists wrote mismatched vectors into the same Chroma collection silently.
The guard stamps backend+dim into collection metadata on first write and
fails loud on any later mismatch.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.retrieval import embedder
from app.retrieval.embedder import (
    EmbeddingDimensionMismatch,
    active_backend_name,
    check_collection_dim,
)


def _collection(meta):
    return SimpleNamespace(metadata=dict(meta))


def test_active_backend_name_dispatch(monkeypatch):
    monkeypatch.setattr(embedder.settings, "USE_LOCAL_EMBEDDINGS", True)
    monkeypatch.setattr(embedder.settings, "LOCAL_EMBED_MODEL", "e5")
    assert active_backend_name() == "local-e5"

    monkeypatch.setattr(embedder.settings, "LOCAL_EMBED_MODEL", "minilm")
    assert active_backend_name() == "local"

    monkeypatch.setattr(embedder.settings, "USE_LOCAL_EMBEDDINGS", False)
    monkeypatch.setattr(embedder.settings, "USE_JINA_EMBEDDINGS", True)
    monkeypatch.setattr(embedder.settings, "JINA_API_KEY", "k")
    assert active_backend_name() == "jina"

    monkeypatch.setattr(embedder.settings, "JINA_API_KEY", "")
    assert active_backend_name() == "openai"


def test_unstamped_collection_returns_stamp():
    coll = _collection({"hnsw:space": "cosine"})
    stamp = check_collection_dim(coll, 1024, backend="jina")
    assert stamp is not None
    assert stamp["orivory_embed_dim"] == 1024
    assert stamp["orivory_embed_backend"] == "jina"
    assert stamp["hnsw:space"] == "cosine"  # existing keys preserved


def test_matching_stamp_passes_silently():
    coll = _collection({"orivory_embed_backend": "jina", "orivory_embed_dim": 1024})
    assert check_collection_dim(coll, 1024, backend="jina") is None


def test_dimension_switch_raises_loudly():
    coll = _collection({"orivory_embed_backend": "jina", "orivory_embed_dim": 1024})
    with pytest.raises(EmbeddingDimensionMismatch, match=r"1024.*1536|1536.*1024"):
        check_collection_dim(coll, 1536, backend="openai")


def test_backend_switch_same_dim_raises():
    coll = _collection({"orivory_embed_backend": "jina", "orivory_embed_dim": 1024})
    with pytest.raises(EmbeddingDimensionMismatch, match="backend"):
        check_collection_dim(coll, 1024, backend="local")


def test_unreadable_metadata_skips_silently():
    """Unit-test doubles (MagicMock) expose non-dict metadata — the guard
    must never break mocked paths, it just cannot verify them."""
    assert check_collection_dim(SimpleNamespace(), 1536, backend="openai") is None

    class _Weird:
        @property
        def metadata(self):
            raise RuntimeError("nope")

    assert check_collection_dim(_Weird(), 1536, backend="openai") is None
