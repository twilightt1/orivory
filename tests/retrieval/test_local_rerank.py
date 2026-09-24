"""Local (ONNX) rerank lane: windowing, outcome parity, failure typing.

The lane is the only one, so these pins ARE the rerank contract: ordering,
dedup, the ``RERANK_TOP_N`` cap, typed failures, and the "dense order
continues" fallback. The model itself enters only through
``local_reranker.score_pairs``, so every contract pin here runs against a stub
— the one test that drives the real ONNX session is cache-guarded (house
artifact guard: never download 341 MB in CI).
"""
from __future__ import annotations

import numpy as np
import pytest

from app.config import settings
from app.retrieval import local_reranker
from app.retrieval import reranker as reranker_module
from app.retrieval.reranker import RerankInvalidResponse, RerankUnavailable

# ── fakes ───────────────────────────────────────────────────────────────────


class _Enc:
    def __init__(self, ids):
        self.ids = ids
        self.attention_mask = [1] * len(ids)


class _FakeTok:
    """One token per whitespace word — enough to prove slicing + round-trip."""

    def __init__(self):
        self._ids: dict[str, int] = {}
        self.decode_calls = 0

    def _id(self, word: str) -> int:
        return self._ids.setdefault(word, len(self._ids))

    def encode(self, text: str, pair: str | None = None):
        words = text.split() if pair is None else [*text.split(), *pair.split()]
        return _Enc([self._id(w) for w in words])

    def decode(self, ids):
        self.decode_calls += 1
        rev = {v: k for k, v in self._ids.items()}
        return " ".join(rev[i] for i in ids)


class _In:
    def __init__(self, name):
        self.name = name


class _FakeSess:
    """Returns one logit per call, counting calls: proves the window max-pool."""

    def __init__(self, values):
        self._values = list(values)
        self.calls = 0
        self.input_ids = []

    def get_inputs(self):
        return [_In("input_ids"), _In("attention_mask"), _In("token_type_ids")]

    def run(self, _outputs, feed):
        self.input_ids.append(feed["input_ids"].copy())
        value = self._values[min(self.calls, len(self._values) - 1)]
        self.calls += 1
        return [np.asarray([[value]], dtype=np.float32)]


def _chunk(mid, content, score=0.5):
    return {"memory_id": mid, "content": content, "score": score}


# ── windowing ───────────────────────────────────────────────────────────────


def test_windows_short_text_is_returned_untouched():
    tok = _FakeTok()
    text = "a b c d"
    # No decode round-trip for a text that fits: identity, not a re-join.
    assert local_reranker._windows(tok, text, room=10) == [text]


def test_windows_cover_the_tail_of_a_long_text():
    tok = _FakeTok()
    words = [f"w{i}" for i in range(10)]
    windows = local_reranker._windows(tok, " ".join(words), room=4)
    assert len(windows) == 3, windows
    assert all(len(w.split()) <= 4 for w in windows)
    # The last word must be scored in SOME window — silent truncation here is
    # exactly what the local graph does on its own and what this must prevent.
    assert any("w9" in w for w in windows)


def test_windows_stop_at_the_cap():
    tok = _FakeTok()
    text = " ".join(f"w{i}" for i in range(100))
    assert len(local_reranker._windows(tok, text, room=4)) == local_reranker._MAX_WINDOWS
    assert tok.decode_calls == local_reranker._MAX_WINDOWS


def test_long_query_is_bounded_before_document_windowing(monkeypatch):
    monkeypatch.setattr(local_reranker, "_MAX_TOKENS", 16)
    tok = _FakeTok()
    sess = _FakeSess([0.5])
    query = " ".join(f"q{i}" for i in range(20))

    local_reranker._score_pair(sess, tok, query, "doc-a doc-b")

    input_ids = sess.input_ids[0][0].tolist()
    assert len(input_ids) <= 16
    assert tok._id("doc-a") in input_ids, "long queries must not truncate every document token"
    assert tok._id("q19") not in input_ids, "the query is capped before pair encoding"


def test_pair_score_is_the_max_over_windows(monkeypatch):
    tok = _FakeTok()
    monkeypatch.setattr(local_reranker, "_MAX_TOKENS", 70)
    sess = _FakeSess([0.1, 0.9, 0.4, 0.2, 0.2])
    doc = " ".join(f"w{i}" for i in range(200))  # 200 tokens, 4 windows at 64
    score = local_reranker._score_pair(sess, tok, "q", doc)
    assert sess.calls == local_reranker._MAX_WINDOWS, "one forward pass per window, capped"
    assert score == pytest.approx(1.0 / (1.0 + np.exp(-0.9))), "max-pool over windows, not the last/first one"


def test_negative_logit_maps_to_positive_relevance():
    tok = _FakeTok()
    sess = _FakeSess([-2.0])

    score = local_reranker._score_pair(sess, tok, "q", "doc")

    assert score == pytest.approx(1.0 / (1.0 + np.exp(2.0)))
    assert 0.0 < score < 1.0


# ── contract parity through the shared entry point ──────────────────────────


@pytest.mark.asyncio
async def test_the_lane_orders_and_stamps(monkeypatch):
    async def _scores(_query, docs):
        return [0.1, 0.9, 0.4][: len(docs)]

    monkeypatch.setattr(local_reranker, "score_pairs", _scores)
    rows = await reranker_module.rerank("q", [_chunk("a", "c1"), _chunk("b", "c2"), _chunk("c", "c3")])
    assert [(r["memory_id"], r["rerank_score"]) for r in rows] == [
        ("b", 0.9), ("c", 0.4), ("a", 0.1)
    ]


@pytest.mark.asyncio
async def test_the_lane_caps_at_top_n(monkeypatch):
    async def _scores(_query, docs):
        return [0.1, 0.9, 0.4, 0.3][: len(docs)]

    monkeypatch.setattr(local_reranker, "score_pairs", _scores)
    monkeypatch.setattr(settings, "RERANK_TOP_N", 2)
    rows = await reranker_module.rerank("q", [_chunk(str(i), "c") for i in range(4)])
    assert len(rows) == 2, "RERANK_TOP_N caps the lane's own answer"


@pytest.mark.asyncio
async def test_the_lane_keeps_the_best_chunk_per_memory(monkeypatch):
    """Two chunks of one memory: the rows arrive in pool order, so the dedup
    must land on the highest-scored chunk, not on whichever sat first."""

    async def _scores(_query, docs):
        return [0.2, 0.8][: len(docs)]

    monkeypatch.setattr(local_reranker, "score_pairs", _scores)
    rows = await reranker_module.rerank(
        "q", [_chunk("m1", "weak chunk"), _chunk("m1", "strong chunk")]
    )
    assert len(rows) == 1
    assert rows[0]["content"] == "strong chunk"


@pytest.mark.asyncio
async def test_a_scoring_failure_is_typed_unavailable(monkeypatch):
    async def _boom(_query, _docs):
        raise OSError("model file gone")

    monkeypatch.setattr(local_reranker, "score_pairs", _boom)
    with pytest.raises(RerankUnavailable):
        await reranker_module.rerank("q", [_chunk("a", "c")])


@pytest.mark.asyncio
async def test_an_empty_pool_short_circuits(monkeypatch):
    async def _scores(_query, _docs):  # pragma: no cover - must not be reached
        raise AssertionError("no chunks handed in → no session work")

    monkeypatch.setattr(local_reranker, "score_pairs", _scores)
    assert await reranker_module.rerank("q", []) == []


@pytest.mark.asyncio
async def test_scoring_nothing_at_all_is_typed_invalid_response(monkeypatch):
    """R14(p2) outlives the transport it was written for: a scorer that answers
    with no row at all is a counted invalid response, never a silent fallback."""

    async def _no_rows(_query, _docs):
        return []

    monkeypatch.setattr(local_reranker, "score_pairs", _no_rows)
    with pytest.raises(RerankInvalidResponse):
        await reranker_module.rerank("q", [_chunk("a", "c")])


# ── the real model (cache-guarded house artifact) ───────────────────────────

_RELEVANT = "Orivory lưu ký ức trong SQLite kèm Qdrant nhúng, không cần dịch vụ ngoài."
_IRRELEVANT = "Giá cà phê robusta tăng nhẹ so với tuần trước."


@pytest.mark.asyncio
@pytest.mark.skipif(
    not local_reranker.files_cached(),
    reason="gte reranker onnx cache missing — run local, do not download in CI",
)
async def test_real_model_ranks_the_relevant_chunk_first():
    scores = await local_reranker.score_pairs(
        "Orivory lưu ký ức ở đâu?", [_IRRELEVANT, _RELEVANT]
    )
    assert scores[1] > scores[0], scores
