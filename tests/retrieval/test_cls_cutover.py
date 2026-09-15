"""Task 4: Arctic XS CLS pooling cutover + MiniLM/chromadb branch removal.

Production arctic embeddings are CLS-pooled (``last_hidden_state[:, 0, :]``,
L2-normalized); e5 keeps mean pooling. Pooling is a parameter of the ONE
shared encoder, so the two contracts cannot drift apart silently.

Everything here runs on a stub session — no ONNX artifacts required (the
model-dependent parity check lives in ``test_xs_parity.py`` behind its skip
guard).
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import pytest
from pydantic import ValidationError

from app.config import Settings, settings
from app.retrieval import e5_local, embedder
from app.retrieval import embedding_fingerprint as fp

DIM = 2


class _Encoding:
    def __init__(self, ids, mask, length: int | None = None):
        n = length if length is not None else len(ids)
        self.ids = [1] * n
        self.attention_mask = [1] * n


class _FakeTokenizer:
    def __init__(self, length: int = 3) -> None:
        self.length = length
        self.texts: list[str] = []

    def encode_batch(self, texts):
        self.texts.extend(texts)
        return [_Encoding([], [], length=self.length) for _ in texts]


class _StubSession:
    """Token t of every row is ``[t + 1, 1 / (t + 1)]``.

    CLS (token 0) and the masked mean therefore point in measurably
    different directions, and both are L2-normalized by the encoder.
    """

    def __init__(self) -> None:
        _In = namedtuple("_In", ["name"])
        self._inputs = [_In("input_ids"), _In("attention_mask")]

    def get_inputs(self):
        return self._inputs

    def run(self, _output_names, inputs):
        n, seq = inputs["input_ids"].shape
        last = np.zeros((n, seq, DIM), dtype=np.float32)
        for t in range(seq):
            last[:, t, 0] = t + 1
            last[:, t, 1] = 1.0 / (t + 1)
        return [last]


@pytest.fixture()
def stubs(monkeypatch):
    tok, sess = _FakeTokenizer(), _StubSession()
    monkeypatch.setattr(e5_local, "_asession", lambda: sess)
    monkeypatch.setattr(e5_local, "_atokenizer", lambda: tok)
    monkeypatch.setattr(e5_local, "_session", lambda: sess)
    monkeypatch.setattr(e5_local, "_tokenizer", lambda: tok)
    return tok, sess


def _encode(texts, pooling):
    return e5_local._encode_with(texts, lambda: _StubSession(), lambda: _FakeTokenizer(), pooling=pooling)


# ── shared encoder: pooling is a parameter, mean stays the default ──────────


def test_pooling_defaults_to_mean_for_the_e5_path(stubs):
    default = e5_local._encode_with(["a"], e5_local._session, e5_local._tokenizer)
    explicit = _encode(["a"], "mean")
    assert np.allclose(default, explicit)


def test_cls_and_mean_differ_but_share_shape_and_unit_norm(stubs):
    mean = np.asarray(_encode(["a", "b"], "mean"))
    cls = np.asarray(_encode(["a", "b"], "cls"))

    assert mean.shape == cls.shape == (2, DIM)
    for vecs in (mean, cls):
        assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-6)
    assert not np.allclose(mean, cls)
    # CLS is literally token 0, normalized.
    assert np.allclose(cls[0], np.array([1.0, 1.0]) / np.sqrt(2.0))


def test_unknown_pooling_fails_loudly(stubs):
    with pytest.raises(ValueError, match="pooling"):
        _encode(["a"], "clsx")


class _FlatSession:
    """A 2-D export (already pooled — no token axis)."""

    def __init__(self) -> None:
        _In = namedtuple("_In", ["name"])
        self._inputs = [_In("input_ids"), _In("attention_mask")]

    def get_inputs(self):
        return self._inputs

    def run(self, _output_names, inputs):
        return [np.zeros((inputs["input_ids"].shape[0], DIM), dtype=np.float32)]


def test_cls_pooling_requires_token_embeddings():
    """A 2-D export must raise, not mis-slice ``last[:, 0, :]`` into silence."""
    with pytest.raises(ValueError, match="CLS pooling requires"):
        e5_local._encode_with(
            ["a"], lambda: _FlatSession(), lambda: _FakeTokenizer(), pooling="cls"
        )


def test_arctic_call_sites_use_cls_pooling(stubs):
    """The arctic entry points must request CLS — not inherit the e5 mean."""
    cls = np.asarray(_encode(["hello"], "cls"))
    mean = np.asarray(_encode([e5_local.ARCTIC_QUERY_PREFIX + "hello"], "mean"))

    assert np.allclose(np.asarray(e5_local.arctic_embed_queries(["hello"])), cls)
    assert np.allclose(np.asarray(e5_local.arctic_embed_passages(["hello"])), cls)
    assert not np.allclose(np.asarray(e5_local.arctic_embed_passages(["hello"])), mean)


def test_tokenization_padding_and_truncation_are_unchanged(stubs):
    tok, _ = stubs
    e5_local.arctic_embed_queries(["hello"])
    assert tok.texts == [e5_local.ARCTIC_QUERY_PREFIX + "hello"]
    e5_local.arctic_embed_passages(["hello"])
    assert tok.texts[-1] == "hello"


# ── fingerprint: arctic CLS is the current contract, legacy mean kept ───────


def test_current_fingerprint_arctic_is_cls(monkeypatch):
    monkeypatch.setattr(settings, "USE_LOCAL_EMBEDDINGS", True)
    monkeypatch.setattr(settings, "LOCAL_EMBED_MODEL", "arctic")

    fingerprint = fp.current_fingerprint()

    assert fingerprint == fp.ARCTIC_CLS_FINGERPRINT
    assert fingerprint["pooling"] == "cls"
    assert fingerprint["dim"] == 384
    assert fingerprint["provider"] == "onnxruntime-cpu"


def test_cls_fingerprint_differs_from_legacy_mean_only_in_pooling():
    assert set(fp.ARCTIC_CLS_FINGERPRINT) == set(fp.LEGACY_MEAN_FINGERPRINT)
    differing = {
        key
        for key in fp.LEGACY_MEAN_FINGERPRINT
        if fp.LEGACY_MEAN_FINGERPRINT[key] != fp.ARCTIC_CLS_FINGERPRINT[key]
    }
    assert differing == {"pooling"}
    assert fp.LEGACY_MEAN_FINGERPRINT["pooling"] == "legacy-mean"


def test_legacy_mean_fingerprint_stays_usable_for_the_ablation():
    """The mean contract must still hash and key caches (ablation/rollback)."""
    legacy = fp.LEGACY_MEAN_FINGERPRINT
    assert fp.canonical_fingerprint(legacy)
    assert fp.fingerprint_generation(legacy) != fp.fingerprint_generation(fp.ARCTIC_CLS_FINGERPRINT)
    assert fp.cache_key(legacy, "passage", "x") != fp.cache_key(
        fp.ARCTIC_CLS_FINGERPRINT, "passage", "x"
    )


def test_e5_fingerprint_is_unchanged(monkeypatch):
    monkeypatch.setattr(settings, "USE_LOCAL_EMBEDDINGS", True)
    monkeypatch.setattr(settings, "LOCAL_EMBED_MODEL", "e5")

    fingerprint = fp.current_fingerprint()

    assert fingerprint["pooling"] == "mean"
    assert fingerprint["model_id"] == "Xenova/multilingual-e5-small"
    assert fingerprint["provider"] == "onnxruntime-cpu"


# ── generation naming helper (R19) ──────────────────────────────────────────


def test_generation_name_names_both_kinds_from_the_active_fingerprint(monkeypatch):
    monkeypatch.setattr(settings, "USE_LOCAL_EMBEDDINGS", True)
    monkeypatch.setattr(settings, "LOCAL_EMBED_MODEL", "arctic")
    fp8 = fp.fingerprint_generation(fp.current_fingerprint())[:8]

    assert fp.generation_name("memory") == f"orivory_memories__{fp8}"
    assert fp.generation_name("chunk") == f"orivory_chunks__{fp8}"


def test_generation_name_accepts_an_explicit_fingerprint(monkeypatch):
    """Ablation/rollback name the legacy mean generation, not the active one."""
    monkeypatch.setattr(settings, "USE_LOCAL_EMBEDDINGS", True)
    monkeypatch.setattr(settings, "LOCAL_EMBED_MODEL", "arctic")

    legacy8 = fp.fingerprint_generation(fp.LEGACY_MEAN_FINGERPRINT)[:8]
    active8 = fp.fingerprint_generation(fp.current_fingerprint())[:8]

    assert legacy8 != active8
    assert fp.generation_name("memory", fp.LEGACY_MEAN_FINGERPRINT) == f"orivory_memories__{legacy8}"


def test_generation_name_rejects_unknown_kind():
    with pytest.raises(ValueError, match="kind"):
        fp.generation_name("document")


@pytest.mark.parametrize(
    "explicit",
    [
        pytest.param({}, id="empty-dict"),
        pytest.param("", id="empty-string"),
        pytest.param([1], id="non-mapping"),
    ],
)
def test_generation_name_refuses_an_explicit_empty_or_non_mapping_fingerprint(explicit):
    """T6 rollback / T8 ablation pass explicit fingerprints — a falsy one must
    never be silently upgraded to the ACTIVE generation's name."""
    with pytest.raises(ValueError, match="embedding fingerprint"):
        fp.generation_name("memory", explicit)


# ── MiniLM removal: config load fails loud (R22) ────────────────────────────


@pytest.mark.parametrize("value", ["minilm", "miniln", "onnx-minilm", "ARCTIC", ""])
def test_local_embed_model_rejects_removed_and_unknown_values_at_load(value):
    with pytest.raises(ValidationError, match="LOCAL_EMBED_MODEL"):
        Settings(_env_file=None, LOCAL_EMBED_MODEL=value)


def test_local_embed_model_accepts_arctic_and_e5():
    assert Settings(_env_file=None).LOCAL_EMBED_MODEL == "arctic"
    assert Settings(_env_file=None, LOCAL_EMBED_MODEL="e5").LOCAL_EMBED_MODEL == "e5"
    assert Settings(_env_file=None, LOCAL_EMBED_MODEL="arctic").LOCAL_EMBED_MODEL == "arctic"


def test_local_embedding_path_has_no_chromadb_branch():
    assert not hasattr(embedder, "_local_embed_fn")


def test_local_dispatch_only_knows_arctic_and_e5(monkeypatch, stubs):
    monkeypatch.setattr(settings, "USE_LOCAL_EMBEDDINGS", True)
    calls: list[tuple[str, bool, list[str]]] = []

    def record(name):
        def _fake(texts):
            calls.append((name, "query" in name, texts))
            return [[0.5]]

        return _fake

    for name in ("arctic_embed_queries", "arctic_embed_passages", "embed_queries", "embed_passages"):
        monkeypatch.setattr(e5_local, name, record(name))

    monkeypatch.setattr(settings, "LOCAL_EMBED_MODEL", "arctic")
    assert embedder._embed_with_local(["q"], query=True) == [[0.5]]
    assert embedder._embed_with_local(["p"]) == [[0.5]]
    monkeypatch.setattr(settings, "LOCAL_EMBED_MODEL", "e5")
    assert embedder._embed_with_local(["q"], query=True) == [[0.5]]
    assert embedder._embed_with_local(["p"]) == [[0.5]]

    assert [name for name, _, _ in calls] == [
        "arctic_embed_queries",
        "arctic_embed_passages",
        "embed_queries",
        "embed_passages",
    ]


def test_local_dispatch_fails_loud_on_an_unknown_model(monkeypatch):
    """Config load refuses these; a hand-patched setting must not embed as arctic."""
    monkeypatch.setattr(settings, "USE_LOCAL_EMBEDDINGS", True)
    monkeypatch.setattr(settings, "LOCAL_EMBED_MODEL", "minilm")

    with pytest.raises(ValueError, match="LOCAL_EMBED_MODEL"):
        embedder._embed_with_local(["q"], query=True)
