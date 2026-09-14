"""Tests for the local multilingual-e5 embedding backend (no model download)."""

from __future__ import annotations

import numpy as np
import pytest

from app.retrieval import e5_local
from app.retrieval.embedder import EmbeddingDimensionMismatch, check_collection_dim


class _Encoding:
    def __init__(self, ids, mask):
        self.ids = ids
        self.attention_mask = mask


class _FakeTokenizer:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def encode_batch(self, texts):
        self.texts.extend(texts)
        return [_Encoding([1, 2, 3], [1, 1, 1]) for _ in texts]


class _StubSession:
    def __init__(self, dim: int = 8) -> None:
        from collections import namedtuple

        self.dim = dim
        self.calls = 0
        _In = namedtuple("_In", ["name"])
        self._inputs = [_In("input_ids"), _In("attention_mask")]

    def get_inputs(self):
        return self._inputs

    def run(self, _output_names, inputs):
        self.calls += 1
        n = inputs["input_ids"].shape[0]
        seq = inputs["input_ids"].shape[1]
        return [np.ones((n, seq, self.dim), dtype=np.float32)]


@pytest.fixture()
def stubbed(monkeypatch):
    tok, sess = _FakeTokenizer(), _StubSession()
    monkeypatch.setattr(e5_local, "_tokenizer", lambda: tok)
    monkeypatch.setattr(e5_local, "_session", lambda: sess)
    return tok, sess


def test_query_prefix_applied(stubbed):
    tok, _ = stubbed
    e5_local.embed_queries(["hello"])
    assert tok.texts == ["query: hello"]


def test_passage_prefix_applied(stubbed):
    tok, _ = stubbed
    e5_local.embed_passages(["hello"])
    assert tok.texts == ["passage: hello"]


def test_arctic_query_prefix_only(stubbed, monkeypatch):
    tok, _ = stubbed
    monkeypatch.setattr(e5_local, "_asession", lambda: e5_local._session())
    monkeypatch.setattr(e5_local, "_atokenizer", lambda: e5_local._tokenizer())
    e5_local.arctic_embed_queries(["hello"])
    assert tok.texts[-1].startswith("Represent this sentence")
    e5_local.arctic_embed_passages(["hello"])
    assert tok.texts[-1] == "hello"


def test_embeddings_l2_normalized(stubbed):
    _, sess = stubbed
    vecs = e5_local.embed_queries(["a", "b"])
    assert sess.calls == 1
    assert len(vecs) == 2
    for v in vecs:
        assert len(v) == 8
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_long_batches_split(stubbed, monkeypatch):
    _, sess = stubbed
    monkeypatch.setattr(e5_local, "_BATCH", 2)
    vecs = e5_local.embed_queries(["a", "b", "c"])
    assert sess.calls == 2
    assert len(vecs) == 3


def test_backend_guard_distinguishes_e5_from_minilm():
    stamped = {
        "orivory_embed_backend": "local",
        "orivory_embed_dim": 384,
        "orivory_embed_fingerprint": "local-test-contract",
    }

    class _C:
        metadata = stamped

    with pytest.raises(EmbeddingDimensionMismatch):
        check_collection_dim(
            _C(),
            384,
            backend="local-e5",
            fingerprint="local-test-contract",
        )
    assert (
        check_collection_dim(
            _C(),
            384,
            backend="local",
            fingerprint="local-test-contract",
        )
        is None
    )
