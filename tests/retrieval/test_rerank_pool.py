"""Rerank contract (P2/T2): explicit pool, merge (never shrink), bounded refill, typed failures.

Pins the four rulings:

- R4(p2): the dense fetch is ``max(top_k, ceil(top_k * RETRIEVAL_RERANK_POOL_MULTIPLIER))``
  (default 2.0) and the per-call ``top_n`` is the REQUEST's own top_k;
  ``JINA_RERANKER_TOP_N`` is only the transport CAP over it.
- R10(p2): rerank SUCCESS merges (``ranked + remainder``, deduped by memory id) —
  ``len(results) == min(top_k, eligible)`` and the count never depends on the
  reranker. RED before this task: success REPLACED the pool with Jina's rows,
  so ``top_k=10`` under the default cap returned 5.
- R11(p2): ``RerankUnavailable`` (transport/status/timeout) and
  ``RerankInvalidResponse`` (unusable body) both degrade to dense order with
  ``retrieval.rerank_failed`` counted and ``stage_ms["rerank"]`` written; a
  malformed ROW is skipped, never fatal.
- R12(p2): at most ONE refill search (hard cap ``top_k * 4``), only when
  filtering left fewer than top_k AND the first fetch filled its whole pool;
  the trace records it — rerank flag or not, it is not rerank-specific.
- R13(p2): ``JINA_RERANKER_TOP_N`` (shipped default 20) stays a CAP and covers
  the default ``top_k=10`` x pool-multiplier window. The count pins below hold
  the cap at the pre-R13 5 on purpose, so they keep proving the served count
  does not depend on the reranker's answer size.
- R14(p2): an EMPTY ``results`` array is a ``RerankInvalidResponse`` — counted,
  dense order served. A transport that answers nothing must not hide as
  "nothing to rerank".

Zero is data: a ``0.0`` relevance_score stays a score (read by key presence,
never truthiness) — the already-correct behaviour stays pinned.
"""
from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import httpx
import pytest

from app.config import settings
from app.observability.fallbacks import fallback_counts, reset_fallback_counts
from app.retrieval import reranker as reranker_module
from app.retrieval.memory import retriever as rmod
from app.retrieval.memory.retriever import MemoryRetriever
from app.retrieval.reranker import RerankInvalidResponse, RerankUnavailable
from app.schemas.Orivory import (
    RECALL_TRACE_COUNTER_KEYS,
    RECALL_TRACE_STAGE_KEYS,
    RECALL_TRACE_ZERO_KEYS,
    RecallTrace,
)

# ── fakes ───────────────────────────────────────────────────────────────────


def _mem(uid, *, content="c", days_old=1, dirty=False, superseded=False):
    from app.models.memory import Memory

    meta: dict = {}
    if superseded:
        meta["cm_superseded_by"] = str(uuid.uuid4())
    if dirty:
        meta["cm_derived_from"] = [str(uuid.uuid4())]
        meta["cm_derived_dirty"] = True
    return Memory(
        id=uuid.uuid4(),
        user_id=uid,
        title=content[:40],
        content=content,
        tags=[],
        salience=0.5,
        pinned=False,
        source_type="manual_note",
        recall_count=0,
        captured_at=datetime.now(UTC) - timedelta(days=days_old),
        indexed_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        extra_metadata=meta,
    )


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _FakeDB:
    """The REAL ``_hydrate`` runs against this: the statement is ignored and every
    row is returned, so a candidate id that is NOT among ``rows`` is exactly the
    "still in the index, gone from SQL" case the eligibility filter drops."""

    def __init__(self, rows):
        self.rows = list(rows)

    async def execute(self, _statement):
        return _FakeResult(self.rows)


class _Store:
    """A dense store: real slicing, real order, every limit recorded."""

    def __init__(self, rows):  # [(memory_id, score)], best first
        self.rows = list(rows)
        self.calls: list[int] = []

    async def __call__(self, _embedding, *, user_id, top_k=10, where=None,
                       namespace=None):
        self.calls.append(top_k)
        return [
            {"memory_id": str(memory_id), "content": "vector copy", "score": score}
            for memory_id, score in self.rows[:top_k]
        ]


def _cap_rerank(seen_top_n=None, seen_chunks=None):
    """The transport's own shape: at most ``min(top_n, JINA_RERANKER_TOP_N)``
    rows come back, best first — never the whole pool."""

    async def _rerank(_query, chunks, *, top_n=None):
        if seen_top_n is not None:
            seen_top_n.append(top_n)
        if seen_chunks is not None:
            seen_chunks.extend(chunks)
        cap = settings.JINA_RERANKER_TOP_N if top_n is None else int(top_n)
        cap = min(cap, settings.JINA_RERANKER_TOP_N)
        return [
            dict(chunk, rerank_score=1.0 - index / 100.0)
            for index, chunk in enumerate(chunks[:cap])
        ]

    return _rerank


@pytest.fixture()
def recall_env(monkeypatch, barrier_outbox):
    """The recall path with every out-of-process seam faked except the reranker
    (each test wires its own; ``barrier_outbox`` gives the freshness barrier its
    real outbox — see tests/retrieval/conftest.py)."""
    uid = uuid.uuid4()

    async def _rewrite(query, context=None):
        return {
            "rewritten_query": query,
            "entities": [],
            "reasoning": None,
            "_fallback_used": False,
        }

    async def _embed(_query):
        return [0.1, 0.2]

    monkeypatch.setattr(rmod, "rewrite_query", _rewrite)
    monkeypatch.setattr(rmod, "embed_query", _embed)
    monkeypatch.setattr(rmod, "fetch_personal_context", AsyncMock(return_value=[]))
    return uid


@pytest.fixture(autouse=True)
def _pin_the_paid_transport(monkeypatch):
    """This module exercises the Jina HTTP lane — pin it.

    ``RERANK_BACKEND=auto`` resolves to the bundled local cross-encoder, so
    every test here that mocks ``get_jina_client`` must ask for the paid lane
    by name; the auto/local resolution is pinned in test_local_rerank.py.
    """
    monkeypatch.setattr(settings, "RERANK_BACKEND", "jina")


# ── R10/R4: count invariant over top_k x pool shape ─────────────────────────


@pytest.mark.parametrize("top_k", [1, 5, 10, 15])
@pytest.mark.parametrize("shape", ["sufficient", "stale_heavy", "empty"])
async def test_returned_count_is_min_top_k_eligible_never_the_rerankers(
    recall_env, monkeypatch, top_k, shape
):
    """R10/R4. Sufficient pool: the reranker returns at most the CAP (pinned to
    the pre-R13 5 here — the count invariant must keep biting) yet the caller
    asked for top_k; the merge is what makes the count independent of it.

    The expectation is derived from the CONTRACT — the limit the call SHOULD
    have used (R4 pool, then R12's refill) — never from ``store.calls[-1]``,
    which would happily track an under-fetch and stay green.
    """
    uid = recall_env
    monkeypatch.setattr(settings, "JINA_RERANKER_TOP_N", 5)  # the pre-R13 cap

    clean = [_mem(uid, content=f"clean{i}") for i in range(40)]
    stale = [_mem(uid, content=f"dirty{i}", dirty=True) for i in range(16)]
    superseded = [_mem(uid, content=f"old{i}", superseded=True) for i in range(8)]
    gone = [uuid.uuid4() for _ in range(8)]  # in the index, no SQL row

    if shape == "sufficient":
        page = clean
    elif shape == "stale_heavy":
        page = [*stale, *clean]  # the first fetch page is mostly stale
    else:
        page = []
    rows = [(m.id, 0.90 - i / 1000) for i, m in enumerate(page)]
    memory_rows = [*clean, *stale, *superseded]
    acceptable = {str(m.id) for m in clean} if shape != "empty" else set()

    # The contract's own expectation (R4 + R12), not the store's last call:
    # the first fetch asks for `max(top_k, ceil(top_k * multiplier))`, and a
    # page that filled that limit while filtering left fewer than top_k earns
    # exactly ONE refill at the `top_k * 4` hard cap.
    pool = max(top_k, math.ceil(top_k * settings.RETRIEVAL_RERANK_POOL_MULTIPLIER))
    first_page = rows[:pool]
    first_eligible = sum(1 for memory_id, _ in first_page if str(memory_id) in acceptable)
    refill = first_eligible < top_k and len(first_page) == pool
    expected_calls = [pool, top_k * 4] if refill else [pool]
    fetched = rows[: expected_calls[-1]]
    eligible = sum(1 for memory_id, _score in fetched if str(memory_id) in acceptable)

    store = _Store(rows)
    monkeypatch.setattr(rmod, "search_memories", store)
    monkeypatch.setattr(reranker_module, "rerank", _cap_rerank())

    retriever = MemoryRetriever(_FakeDB(memory_rows), uid, semantic_rerank=True)
    response = await retriever.recall("backpack", top_k=top_k, include_personal_context=False)

    assert store.calls == expected_calls, "R4's pool, then R12's refill — nothing else"
    assert len(response.results) == min(top_k, eligible)
    assert response.trace.num_results == len(response.results)
    assert {str(result.id) for result in response.results} <= acceptable
    assert not (set(str(g) for g in gone) & {str(result.id) for result in response.results})

    # T3: the same facts, as counters — the pre-filter fetch, the rows the
    # refill really ADDED (a stale-only refill page adds zero, and that zero is
    # a measured fact, not a fabricated one), the pool that entered scoring and
    # the served count. A leg that never ran has NO key.
    counts = response.trace.counts
    assert counts["dense"] == len(first_page)
    assert counts["eligible"] == eligible
    assert counts["returned"] == len(response.results)
    if refill:
        assert counts["refill"] == eligible - first_eligible
    else:
        assert "refill" not in counts
    if shape != "empty" and eligible > 1:  # the rerank leg needs >1 candidate
        assert counts["reranked"] == min(eligible, min(top_k, 5))  # cap pinned above
    else:
        assert "reranked" not in counts


async def test_pool_comes_from_the_settings_multiplier_and_top_n_from_the_request(
    recall_env, monkeypatch
):
    """R4: ``ceil(top_k * RETRIEVAL_RERANK_POOL_MULTIPLIER)`` (signed default
    2.0) replaces the old hardcoded factor 3; the reranker is asked for the
    REQUEST's top_k, capped by the deployment setting."""
    uid = recall_env
    memories = [_mem(uid, content=f"m{i}") for i in range(50)]
    store = _Store([(m.id, 0.9 - i / 1000) for i, m in enumerate(memories)])
    seen_top_n: list = []
    monkeypatch.setattr(rmod, "search_memories", store)
    monkeypatch.setattr(reranker_module, "rerank", _cap_rerank(seen_top_n))

    retriever = MemoryRetriever(_FakeDB(memories), uid, semantic_rerank=True)
    await retriever.recall("backpack", top_k=10, include_personal_context=False)
    assert settings.RETRIEVAL_RERANK_POOL_MULTIPLIER == 2.0  # signed default
    assert store.calls == [math.ceil(10 * 2.0)]  # 20 — not top_k * 3
    assert seen_top_n == [10]  # the request's top_k, not the global cap

    # The setting is read at construction (the `semantic_rerank` seam): a new
    # retriever picks the new value up, an existing one keeps its own.
    monkeypatch.setattr(settings, "RETRIEVAL_RERANK_POOL_MULTIPLIER", 3.0)
    store.calls.clear()
    retriever = MemoryRetriever(_FakeDB(memories), uid, semantic_rerank=True)
    await retriever.recall("backpack", top_k=4, include_personal_context=False)
    assert store.calls == [math.ceil(4 * 3.0)]  # the setting drives the pool

    store.calls.clear()
    retriever = MemoryRetriever(_FakeDB(memories), uid, semantic_rerank=True,
                                pool_multiplier=1.5)
    await retriever.recall("backpack", top_k=4, include_personal_context=False)
    assert store.calls == [6]  # the ctor seam wins over the setting

    monkeypatch.setattr(settings, "RETRIEVAL_RERANK_POOL_MULTIPLIER", 0.1)
    store.calls.clear()
    retriever = MemoryRetriever(_FakeDB(memories), uid, semantic_rerank=True)
    await retriever.recall("backpack", top_k=10, include_personal_context=False)
    assert store.calls == [10]  # a multiplier below 1 must not under-fetch top_k


async def test_empty_pool_is_an_empty_result_and_no_rerank_call(recall_env, monkeypatch):
    uid = recall_env
    store = _Store([])
    seen_top_n: list = []
    monkeypatch.setattr(rmod, "search_memories", store)
    monkeypatch.setattr(reranker_module, "rerank", _cap_rerank(seen_top_n))

    retriever = MemoryRetriever(_FakeDB([]), uid, semantic_rerank=True)
    response = await retriever.recall("backpack", top_k=10, include_personal_context=False)
    assert response.results == [] and response.trace.num_results == 0
    assert seen_top_n == []  # nothing to rerank — no transport call


# ── R10: merge keeps dense rows in dense order ──────────────────────────────


async def test_rerank_success_keeps_the_unranked_remainder_in_dense_order(
    recall_env, monkeypatch
):
    """The reranked head decides the top of the answer; the rows the reranker
    did not return keep their dense order behind it (they are still scored)."""
    uid = recall_env
    monkeypatch.setattr(settings, "JINA_RERANKER_TOP_N", 2)
    memories = [_mem(uid, content=f"m{i}") for i in range(6)]
    store = _Store([(m.id, 0.9 - i / 100) for i, m in enumerate(memories)])
    monkeypatch.setattr(rmod, "search_memories", store)

    async def _rerank(_query, chunks, *, top_n=None):
        # The cross-encoder promotes the LAST dense row, and answers with one
        # row only — the classic "fewer rows than it was handed" success.
        promoted = dict(chunks[-1], rerank_score=0.99)
        return [promoted]

    monkeypatch.setattr(reranker_module, "rerank", _rerank)
    retriever = MemoryRetriever(_FakeDB(memories), uid, semantic_rerank=True)
    response = await retriever.recall("backpack", top_k=6, include_personal_context=False)

    ids = [str(result.id) for result in response.results]
    assert len(ids) == 6  # every eligible row survives the merge
    assert ids[0] == str(memories[-1].id)  # rerank's order wins at the top
    # No row is duplicated by the merge (ranked + remainder, deduped by id).
    assert len(set(ids)) == 6


async def test_zero_rerank_score_is_data_not_absence(recall_env, monkeypatch):
    """A ``0.0`` relevance is a real score: read by key presence, never
    truthiness — otherwise the dense score would silently replace it."""
    uid = recall_env
    zero = _mem(uid, content="zero relevance")
    other = _mem(uid, content="positive relevance")
    store = _Store([(zero.id, 0.99), (other.id, 0.10)])
    monkeypatch.setattr(rmod, "search_memories", store)

    async def _rerank(_query, chunks, *, top_n=None):
        graded = []
        for chunk in chunks:
            stamped = dict(chunk)
            stamped["rerank_score"] = 0.0 if chunk["memory_id"] == str(zero.id) else 0.7
            graded.append(stamped)
        return graded

    monkeypatch.setattr(reranker_module, "rerank", _rerank)
    retriever = MemoryRetriever(_FakeDB([zero, other]), uid, semantic_rerank=True)
    response = await retriever.recall("backpack", top_k=2, include_personal_context=False)

    by_id = {str(result.id): result for result in response.results}
    assert by_id[str(zero.id)].match_reasons[0] == "rerank:0.00"
    assert by_id[str(other.id)].match_reasons[0] == "rerank:0.70"


# ── R12: bounded refill ─────────────────────────────────────────────────────


async def test_one_bounded_refill_when_filtering_empties_a_full_pool(recall_env, monkeypatch):
    """First fetch fills the pool (20 of 20) and filtering leaves 4 < top_k:
    exactly ONE extra search at the ``top_k * 4`` cap, and the trace says so."""
    uid = recall_env
    good = [_mem(uid, content=f"good{i}") for i in range(4)]
    stale = [_mem(uid, content=f"dirty{i}", dirty=True) for i in range(16)]
    later_good = [_mem(uid, content=f"later{i}") for i in range(20)]
    later_stale = [_mem(uid, content=f"later-dirty{i}", dirty=True) for i in range(6)]
    page = [*good, *stale, *later_good, *later_stale]
    rows = [(m.id, 0.9 - i / 1000) for i, m in enumerate(page)]

    store = _Store(rows)
    monkeypatch.setattr(rmod, "search_memories", store)
    monkeypatch.setattr(reranker_module, "rerank", _cap_rerank())

    retriever = MemoryRetriever(_FakeDB(page), uid, semantic_rerank=True)
    response = await retriever.recall("backpack", top_k=10, include_personal_context=False)

    assert store.calls == [20, 40], "one refill, at the top_k * 4 hard cap"
    assert "refill" in response.trace.stage_ms
    assert response.trace.stage_ms["refill"] > 0.0
    ids = {str(result.id) for result in response.results}
    assert len(ids) == 10
    # The refill page is filtered and SQL-authorized exactly like the first:
    # its stale rows are dropped, its good rows are served.
    assert ids <= {str(m.id) for m in [*good, *later_good]}
    assert ids & {str(m.id) for m in later_good}
    # T3: the counter is what the refill ADDED to the pool (post-filter), not
    # the page it fetched (36 new rows: 16 stale + 20 usable).
    assert response.trace.counts["refill"] == 20
    assert response.trace.counts["dense"] == 20
    assert response.trace.counts["eligible"] == 24
    assert response.trace.counts["returned"] == 10


async def test_no_refill_when_the_store_was_already_exhausted(recall_env, monkeypatch):
    """The first fetch returned FEWER rows than the pool: the store has nothing
    more, so a larger limit would only repeat the same page."""
    uid = recall_env
    memories = [_mem(uid, content=f"m{i}") for i in range(6)]
    store = _Store([(m.id, 0.9 - i / 100) for i, m in enumerate(memories)])
    monkeypatch.setattr(rmod, "search_memories", store)
    monkeypatch.setattr(reranker_module, "rerank", _cap_rerank())

    retriever = MemoryRetriever(_FakeDB(memories), uid, semantic_rerank=True)
    response = await retriever.recall("backpack", top_k=10, include_personal_context=False)

    assert store.calls == [20], "no second search — the page was short"
    assert "refill" not in response.trace.stage_ms
    assert "refill" not in response.trace.counts
    assert len(response.results) == 6 == min(10, 6)


async def test_refill_runs_with_rerank_off_too(recall_env, monkeypatch):
    """R12's refill is not rerank-specific: the visibility filter starves a
    plain dense recall of top_k the same way, and the same one bounded refill
    happens — with no transport call and no fallback counted."""
    uid = recall_env
    good = [_mem(uid, content=f"good{i}") for i in range(4)]
    stale = [_mem(uid, content=f"dirty{i}", dirty=True) for i in range(16)]
    later_good = [_mem(uid, content=f"later{i}") for i in range(20)]
    page = [*good, *stale, *later_good]
    store = _Store([(m.id, 0.9 - i / 1000) for i, m in enumerate(page)])

    async def _never(*_args, **_kwargs):
        raise AssertionError("rerank must not be called with semantic_rerank=False")

    monkeypatch.setattr(rmod, "search_memories", store)
    monkeypatch.setattr(reranker_module, "rerank", _never)
    reset_fallback_counts()

    retriever = MemoryRetriever(_FakeDB(page), uid, semantic_rerank=False)
    response = await retriever.recall("backpack", top_k=10, include_personal_context=False)

    assert store.calls == [20, 40], "the refill is not gated on the rerank flag"
    assert response.trace.stage_ms["refill"] > 0.0
    ids = {str(result.id) for result in response.results}
    assert len(ids) == 10
    assert ids <= {str(m.id) for m in [*good, *later_good]}
    assert "retrieval.rerank_failed" not in fallback_counts()
    # Rerank off: the leg never ran, so its counter never appears (T3/C3).
    assert response.trace.counts["refill"] == 20
    assert "reranked" not in response.trace.counts


# ── R11: typed failures, dense fallback, counted ────────────────────────────


@pytest.mark.parametrize(
    "failure",
    [
        RerankUnavailable("connect timeout"),
        RerankInvalidResponse("results is not a list"),
        RuntimeError("something nobody typed"),
    ],
)
async def test_rerank_failure_degrades_to_dense_order_and_is_counted(
    recall_env, monkeypatch, failure
):
    """Every failure shape — typed or not — leaves dense order in place, with
    the already-registered fallback counted and the duration recorded."""
    uid = recall_env
    memories = [_mem(uid, content=f"m{i}") for i in range(12)]
    store = _Store([(m.id, 0.9 - i / 100) for i, m in enumerate(memories)])
    monkeypatch.setattr(rmod, "search_memories", store)

    async def _boom(_query, chunks, *, top_n=None):
        raise failure

    monkeypatch.setattr(reranker_module, "rerank", _boom)
    reset_fallback_counts()

    retriever = MemoryRetriever(_FakeDB(memories), uid, semantic_rerank=True)
    response = await retriever.recall("backpack", top_k=10, include_personal_context=False)

    assert [str(result.id) for result in response.results] == [
        str(m.id) for m in memories[:10]
    ], "vector order continues on a reranker failure"
    assert fallback_counts()["retrieval.rerank_failed"] == 1
    assert response.trace.stage_ms["rerank"] > 0.0  # recorded in `finally`
    # T3: a fallback serves no reranked head, so no `reranked` count — the
    # fallback counter and the stage duration are the record of what happened.
    assert "reranked" not in response.trace.counts
    assert response.trace.counts["returned"] == 10


async def test_empty_results_body_degrades_to_dense_order_and_is_counted(
    recall_env, monkeypatch
):
    """R14: a 2xx whose ``results`` is EMPTY answers nothing while the pool was
    non-empty. Through the REAL transport under recall that is an invalid
    response: dense order is served, the counter fires exactly once and nothing
    escapes — it must not read as "nothing to rerank" (the uncounted path)."""
    uid = recall_env
    memories = [_mem(uid, content=f"m{i}") for i in range(12)]
    store = _Store([(m.id, 0.9 - i / 100) for i, m in enumerate(memories)])
    monkeypatch.setattr(rmod, "search_memories", store)

    def handler(_request):
        return httpx.Response(200, json={"results": []})

    monkeypatch.setattr(
        reranker_module, "get_jina_client", lambda: _client(httpx.MockTransport(handler))
    )
    reset_fallback_counts()

    retriever = MemoryRetriever(_FakeDB(memories), uid, semantic_rerank=True)
    response = await retriever.recall("backpack", top_k=10, include_personal_context=False)

    assert [str(result.id) for result in response.results] == [
        str(m.id) for m in memories[:10]
    ], "vector order continues when the transport answers with an empty pool"
    assert fallback_counts()["retrieval.rerank_failed"] == 1
    assert response.trace.stage_ms["rerank"] > 0.0
    assert "reranked" not in response.trace.counts  # R14: no head came back


# ── R11: the transport's own classification ─────────────────────────────────


def _client(handler: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=handler)


_DOCS = [
    {"memory_id": "a", "content": "doc a"},
    {"memory_id": "b", "content": "doc b"},
    {"memory_id": "c", "content": "doc c"},
]


async def test_transport_errors_are_rerank_unavailable(monkeypatch):
    def handler(_request):
        raise httpx.ConnectTimeout("took too long")

    monkeypatch.setattr(
        reranker_module, "get_jina_client", lambda: _client(httpx.MockTransport(handler))
    )
    with pytest.raises(RerankUnavailable):
        await reranker_module.rerank("q", _DOCS, top_n=3)


async def test_non_2xx_status_is_rerank_unavailable_not_a_partial_pool(monkeypatch):
    def handler(_request):
        return httpx.Response(503, json={"detail": "service unavailable"})

    monkeypatch.setattr(
        reranker_module, "get_jina_client", lambda: _client(httpx.MockTransport(handler))
    )
    with pytest.raises(RerankUnavailable):
        await reranker_module.rerank("q", _DOCS, top_n=3)


@pytest.mark.parametrize(
    "body",
    [
        {"content": b"<html>gateway error</html>", "headers": {"content-type": "text/html"}},
        {"json": {"results": "not-a-list"}},
        {"json": {}},
        {"json": {"results": []}},  # R14: an empty answer is a failure, not silence
    ],
)
async def test_unusable_bodies_are_rerank_invalid_response(monkeypatch, body):
    def handler(_request):
        return httpx.Response(200, **body)

    monkeypatch.setattr(
        reranker_module, "get_jina_client", lambda: _client(httpx.MockTransport(handler))
    )
    with pytest.raises(RerankInvalidResponse):
        await reranker_module.rerank("q", _DOCS, top_n=3)


async def test_malformed_rows_are_skipped_never_fatal(monkeypatch):
    """One bad row must not throw the whole pool away — including the index
    that is out of range, which used to raise IndexError inside the loop."""
    def handler(_request):
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 0, "relevance_score": 0.4},
                    {"index": 99, "relevance_score": 0.9},  # out of range
                    {"index": 1},  # no relevance_score
                    "not-a-row",  # not a dict
                    {"index": True, "relevance_score": 1.0},  # bool is not an index
                    {"index": 2, "relevance_score": 0.0},  # a REAL zero
                ]
            },
        )

    monkeypatch.setattr(
        reranker_module, "get_jina_client", lambda: _client(httpx.MockTransport(handler))
    )
    ranked = await reranker_module.rerank("q", _DOCS, top_n=3)
    assert [(row["memory_id"], row["rerank_score"]) for row in ranked] == [
        ("a", 0.4),
        ("c", 0.0),
    ]


async def test_duplicate_rows_return_a_memory_once(monkeypatch):
    """M1: two rows carrying the same ``index`` score the SAME memory. Keeping
    both would return it twice and let the repeat occupy a head slot that a
    distinct row should have had — the first (best) row for a memory wins."""
    def handler(_request):
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 1, "relevance_score": 0.8},  # the same memory again
                    {"index": 0, "relevance_score": 0.5},
                    {"index": 2, "relevance_score": 0.0},
                    {"index": 2},  # a repeat with no score is not a second row
                ]
            },
        )

    monkeypatch.setattr(
        reranker_module, "get_jina_client", lambda: _client(httpx.MockTransport(handler))
    )
    ranked = await reranker_module.rerank("q", _DOCS, top_n=3)
    assert [(row["memory_id"], row["rerank_score"]) for row in ranked] == [
        ("b", 0.9),
        ("a", 0.5),
        ("c", 0.0),
    ]


async def test_the_per_call_timeout_bounds_the_request(monkeypatch):
    """R11/M5: the POST carries ``JINA_RERANKER_TIMEOUT_SECONDS`` as its own
    timeout — a hung transport must not hold the recall path for the module
    client's 30 s default."""
    seen: list = []

    def handler(request):
        seen.append(request.extensions.get("timeout"))
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.5}]})

    monkeypatch.setattr(
        reranker_module, "get_jina_client", lambda: _client(httpx.MockTransport(handler))
    )
    monkeypatch.setattr(settings, "JINA_RERANKER_TIMEOUT_SECONDS", 3.5)

    await reranker_module.rerank("q", _DOCS, top_n=3)

    # httpx normalizes the scalar into the per-phase timeout extension.
    assert seen == [{"connect": 3.5, "read": 3.5, "write": 3.5, "pool": 3.5}]


async def test_top_n_is_per_call_and_the_setting_only_caps_it(monkeypatch):
    sent: list[int] = []

    def handler(request):
        import json

        sent.append(json.loads(request.content)["top_n"])
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.5}]})

    monkeypatch.setattr(
        reranker_module, "get_jina_client", lambda: _client(httpx.MockTransport(handler))
    )
    monkeypatch.setattr(settings, "JINA_RERANKER_TOP_N", 7)
    documents = [{"memory_id": str(i), "content": f"c{i}"} for i in range(12)]

    await reranker_module.rerank("q", documents, top_n=10)  # capped by the setting
    await reranker_module.rerank("q", documents, top_n=3)  # the request may be smaller
    await reranker_module.rerank("q", documents)  # no per-call value → the cap
    assert sent == [7, 3, 7]


# ── T3: the trace contract — declared stage keys + real candidate counters ──


def test_the_declared_trace_keys_cover_every_stage_the_path_writes():
    """C2: `refill` (T2's new stage) and the four legacy `_ms` keys are part of
    ``RECALL_TRACE_STAGE_KEYS`` — declared, so the contract covers what the
    retriever writes instead of leaking unversioned extras.

    `refill` is the ONE declared key that is not pre-initialised: T2's contract
    is "present ⟺ the refill ran", and a pre-filled 0.0 would claim it ran in
    zero time. Every other declared key starts at 0.0 (P0 design: the reserved
    keys stay diffable, `lexical` included, until its leg exists in T5).
    """
    assert "refill" in RECALL_TRACE_STAGE_KEYS
    assert "refill" not in RECALL_TRACE_ZERO_KEYS
    assert {"rewrite_ms", "embed_ms", "search_ms", "hydrate_ms"} <= set(RECALL_TRACE_STAGE_KEYS)
    trace = RecallTrace(
        rewritten_query="q", entities=[], latency_ms=0.0, num_candidates=0,
        num_results=0, used_personal_context=False, llm_fallback=False,
    )
    assert set(trace.stage_ms) == set(RECALL_TRACE_ZERO_KEYS)
    assert trace.counts == {}  # counters: absent stays absent, never a zero wall


async def test_the_counters_name_every_leg_that_ran_and_none_that_did_not(
    recall_env, monkeypatch
):
    """C3: `counts` is made of REAL numbers — a key exists iff its leg ran.

    Empty pool: the dense search really ran and counted 0; the legs that never
    ran (lexical/fused — no hybrid index until T5; refill; reranked) stay absent
    rather than appearing as fabricated zeros. With rows: the same keys appear
    with the counts the pipeline factually produced, and nothing else.
    """
    uid = recall_env
    store = _Store([])
    monkeypatch.setattr(rmod, "search_memories", store)
    monkeypatch.setattr(reranker_module, "rerank", _cap_rerank())

    retriever = MemoryRetriever(_FakeDB([]), uid, semantic_rerank=True)
    empty = await retriever.recall("backpack", top_k=10, include_personal_context=False)
    assert empty.trace.counts == {
        "dense": 0, "eligible": 0, "hydrated": 0, "returned": 0,
    }
    assert set(empty.trace.counts) <= set(RECALL_TRACE_COUNTER_KEYS)
    assert "lexical" not in empty.trace.counts and "fused" not in empty.trace.counts

    memories = [_mem(uid, content=f"m{i}") for i in range(12)]
    store.rows = [(m.id, 0.9 - i / 100) for i, m in enumerate(memories)]
    retriever = MemoryRetriever(_FakeDB(memories), uid, semantic_rerank=True)
    full = await retriever.recall("backpack", top_k=5, include_personal_context=False)

    assert full.trace.counts == {
        "dense": 10,      # the R4 pool: ceil(5 * 2.0)
        "eligible": 10,   # every row survived the filter, none refilled
        "reranked": 5,    # min(top_k, cap) rows carry a rerank score
        "hydrated": 12,   # the fake DB hands over every row it holds
        "returned": 5,
    }
    assert set(full.trace.counts) <= set(RECALL_TRACE_COUNTER_KEYS)
