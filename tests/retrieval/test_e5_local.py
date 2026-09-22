"""Tests for the local multilingual-e5 embedding backend (no model download)."""

from __future__ import annotations

import hashlib
import os
import sys
import threading
import time
from email.message import Message
from pathlib import Path

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


# ── the downloader: unique temp, bounded wait, cleanup (findings 88) ────────
#
# No network: every URL below is a ``file://`` URL onto tmp_path, so these
# tests exercise the real download path (urlopen → chunks → digest → publish)
# without leaving the machine.


class _SlowResponse:
    """A real response, read in small slow blocks (two threads must overlap)."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def info(self) -> Message:
        return Message()  # no Content-Length: urlretrieve reads until EOF

    def read(self, n: int = -1):
        time.sleep(0.004)
        return self._inner.read(min(n, 4096) if n and n > 0 else n)

    def close(self) -> None:
        self._inner.close()

    def __enter__(self) -> _SlowResponse:
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


@pytest.fixture()
def source(tmp_path) -> tuple[str, bytes, str]:
    """A local stand-in for the model blob: (file URL, bytes, sha256)."""
    payload = bytes(range(256)) * 256  # 64 KiB
    path = tmp_path / "blob.bin"
    path.write_bytes(payload)
    return path.as_uri(), payload, hashlib.sha256(payload).hexdigest()


def test_a_download_publishes_the_verified_file_and_bounds_its_wait(tmp_path, monkeypatch, source):
    url, payload, digest = source
    dest = tmp_path / "cache" / "model.bin"
    dest.parent.mkdir()

    timeouts: list[float | None] = []
    real_urlopen = e5_local.urllib.request.urlopen

    def _spy(target, *args, **kwargs):
        timeouts.append(kwargs.get("timeout"))
        return real_urlopen(target, *args, **kwargs)

    monkeypatch.setattr(e5_local.urllib.request, "urlopen", _spy)

    e5_local._download(url, dest, digest)

    assert dest.read_bytes() == payload
    assert list(dest.parent.glob("*.part")) == [], "a verified download leaves no temp file"
    assert timeouts and all(t is not None and t > 0 for t in timeouts), (
        "the download must carry a network timeout, not block forever")


def test_an_interrupted_download_leaves_no_partial_file(tmp_path, monkeypatch, source):
    """A stalled/reset connection must not leave a poisoned ``.part`` behind."""
    url, _payload, _digest = source
    dest = tmp_path / "model.bin"

    class _Broken:
        def info(self) -> Message:
            return Message()

        def read(self, n: int = -1):
            raise OSError("connection reset mid-download")

        def close(self) -> None:
            pass

        def __enter__(self) -> _Broken:
            return self

        def __exit__(self, *exc) -> bool:
            return False

    monkeypatch.setattr(e5_local.urllib.request, "urlopen", lambda *a, **k: _Broken())

    with pytest.raises(OSError, match="mid-download"):
        e5_local._download(url, dest, _digest)

    assert not dest.exists()
    assert list(tmp_path.glob("*.part")) == [], "the interrupted attempt left debris"


def test_a_digest_mismatch_leaves_no_partial_file(tmp_path, monkeypatch, source):
    """The pre-existing check, kept honest: nothing is published, nothing kept."""
    url, _payload, _digest = source
    dest = tmp_path / "model.bin"

    with pytest.raises(ValueError, match="failed SHA256 verification"):
        e5_local._download(url, dest, "0" * 64)

    assert not dest.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_two_concurrent_downloads_for_one_destination_never_collide(
    tmp_path, monkeypatch, source
):
    """Two cold starts (warmup + a request retry) must not share an inode."""
    url, payload, digest = source
    dest = tmp_path / "cache" / "model.bin"
    dest.parent.mkdir()

    published: list[str] = []
    real_replace = os.replace

    def _replace(src, dst):
        published.append(Path(src).name)
        real_replace(src, dst)

    monkeypatch.setattr(e5_local.os, "replace", _replace)
    real_urlopen = e5_local.urllib.request.urlopen
    monkeypatch.setattr(
        e5_local.urllib.request,
        "urlopen",
        lambda target, *a, **k: _SlowResponse(real_urlopen(target, *a, **k)),
    )

    start = threading.Barrier(2, timeout=10)
    errors: list[BaseException] = []

    def _run() -> None:
        start.wait()
        try:
            e5_local._download(url, dest, digest)
        except BaseException as exc:  # a collision surfaces as a failed verify/rename
            errors.append(exc)

    threads = [threading.Thread(target=_run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == [], f"concurrent downloads collided: {errors!r}"
    assert dest.read_bytes() == payload
    assert len(published) == 2, "each download must publish its own verified temp file"
    assert len(set(published)) == 2, f"concurrent downloads shared a temp path: {published!r}"
    assert list(dest.parent.glob("*.part")) == []


# ── one native object per process, even from a cold race (finding 87) ───────
#
# One test per locked singleton: `_session`, `_tokenizer`, `_asession`,
# `_atokenizer`. Removing `with _init_lock` from any one builder must kill its
# own test (and only that one), or the lock is not actually pinned.


def _fake_onnxruntime() -> tuple[object, list[float]]:
    """A stand-in onnxruntime module whose session construction is observable."""
    constructed: list[float] = []

    class SessionOptions:
        intra_op_num_threads = 0

    class InferenceSession:
        def __init__(self, *_args, **_kwargs) -> None:
            time.sleep(0.05)  # a native session build: the race window
            constructed.append(time.time())

    return type("ort", (), {"SessionOptions": SessionOptions,
                            "InferenceSession": InferenceSession}), constructed


def _race(call) -> list[object]:
    """Run ``call`` in two threads released together; return both results."""
    start = threading.Barrier(2, timeout=10)
    results: list[object] = []

    def _run() -> None:
        start.wait()
        results.append(call())

    threads = [threading.Thread(target=_run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    return results


def test_two_cold_starts_construct_one_inference_session(monkeypatch, tmp_path):
    """Unsynchronized check-then-set built two native sessions (probe: 2)."""
    fake_ort, constructed = _fake_onnxruntime()
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(e5_local, "ensure_files", lambda: (tmp_path / "m.onnx", tmp_path / "t.json"))
    monkeypatch.setattr(e5_local, "_sess", None)

    results = _race(e5_local._session)

    assert len(constructed) == 1, f"built {len(constructed)} sessions"
    assert results[0] is results[1]


def test_two_cold_starts_construct_one_e5_tokenizer(monkeypatch, tmp_path):
    """`_tokenizer` is the same check-then-set: it must build ONE tokenizer."""
    loaded: list[int] = []

    class Tokenizer:
        @staticmethod
        def from_file(_path):
            time.sleep(0.05)
            loaded.append(1)
            return object()

    monkeypatch.setitem(sys.modules, "tokenizers", type("t", (), {"Tokenizer": Tokenizer}))
    monkeypatch.setattr(
        e5_local, "ensure_files", lambda: (tmp_path / "m.onnx", tmp_path / "t.json")
    )
    monkeypatch.setattr(e5_local, "_tok", None)

    results = _race(e5_local._tokenizer)

    assert len(loaded) == 1, f"loaded the tokenizer {len(loaded)} times"
    assert results[0] is results[1]


def test_two_cold_starts_construct_one_arctic_inference_session(monkeypatch, tmp_path):
    """`_asession` too — a cold race must not build two arctic sessions."""
    fake_ort, constructed = _fake_onnxruntime()
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(
        e5_local, "ensure_arctic_files", lambda: (tmp_path / "a.onnx", tmp_path / "t.json")
    )
    monkeypatch.setattr(e5_local, "_asess", None)

    results = _race(e5_local._asession)

    assert len(constructed) == 1, f"built {len(constructed)} arctic sessions"
    assert results[0] is results[1]


def test_two_cold_starts_construct_one_arctic_tokenizer(monkeypatch, tmp_path):
    loaded: list[int] = []

    class Tokenizer:
        @staticmethod
        def from_file(_path):
            time.sleep(0.05)
            loaded.append(1)
            return object()

    monkeypatch.setitem(sys.modules, "tokenizers", type("t", (), {"Tokenizer": Tokenizer}))
    monkeypatch.setattr(
        e5_local, "ensure_arctic_files", lambda: (tmp_path / "m.onnx", tmp_path / "t.json")
    )
    monkeypatch.setattr(e5_local, "_atok", None)

    results = _race(e5_local._atokenizer)

    assert len(loaded) == 1, f"loaded the tokenizer {len(loaded)} times"
    assert results[0] is results[1]
