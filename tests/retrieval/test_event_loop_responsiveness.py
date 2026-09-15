"""P2/T1 — the event loop stays responsive while the real embedder works.

The stores are the P1b gate's own (a private SQLite file + a private embedded
Qdrant folder, ``_bind_engines`` imported rather than reinvented) and the
embedder is NOT faked: every vector here comes out of the real arctic ONNX
session built at ``~/.cache/orivory/e5``. That is the claim — a synchronous ONNX
call awaited inline from async code must never be what the event loop is waiting
on. Two shapes are covered, because they stall it differently:

* the memory paths (the freshness barrier's drain, the write-through, the query
  embed) embed ONE document per call: a 50-intent batch is ~50 inline calls;
* the ingestion path (``document_service.upload_document`` → the sync pipeline)
  embeds ALL of a document's children in ONE call — the recon's 674-693 ms
  unbroken stall, on an async endpoint.

The measurement is a 2 ms heartbeat (:class:`LoopTicker`) recording how much
later than requested each tick actually ran, while a real recall, a real ingest
(50 documents, the import shape) and a real document upload run concurrently.
Ruling R6(p2) fixes the contract: ``max < 30 ms`` and ``p99 < 15 ms``, with the
model warmed BEFORE the ticker starts (a cold ``InferenceSession`` is boot's
cost, not a request's).

The second test pins the other half of "bounded": the barrier's
``asyncio.wait_for`` must really PREEMPT an offloaded drain. Before P2 the drain
ran inline, so a slow embedding held the barrier — and the whole loop — until
the model returned: the budget was a promise the code could not keep.
"""
from __future__ import annotations

import asyncio
import contextlib
import math
import statistics
import threading
import time
import uuid
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import select

from app import database
from app.config import settings
from app.models.conversation import Conversation
from app.models.document import Document
from app.models.index_outbox import IndexOutbox
from app.models.memory import Memory
from app.retrieval import e5_local, vector_backend
from app.retrieval.embedder import warmup_embedder
from app.retrieval.embedding_fingerprint import generation_name
from app.retrieval.memory import freshness, outbox
from app.retrieval.memory.outbox import IndexFreshnessTimeout
from app.retrieval.memory.write_back import index_new_memory
from tests.retrieval.test_p1b_gate import _bind_engines, _memory, _user

# Ruling R6(p2): the contract, not a suggestion. A flake is fixed by making the
# harness deterministic (warm-up, document count, iteration count) — never by
# relaxing these.
MAX_LAG_MS = 30.0
P99_LAG_MS = 15.0
TICK_MS = 2.0
DRAIN_DOCS = 50  # the default drain batch: the batch the barrier drains on recall
INGEST_DOCS = 50

QUERY = "what did i see on the walk at dusk"

_FILLER = (
    "the orchard walk at dusk was quiet and the air smelled of rain, the lanterns "
    "coming on along the path past the old mill and the river bend where the heron "
    "stands, and i wrote down what the day had been so the memory of it survives "
)
_DOCUMENT_BODY = (_FILLER * 80)[:18000]  # ~52 children at CHILD_SIZE=400


def _text(tag: str, index: int) -> str:
    """One ~864-char memory note — the recon's default-drain document shape."""
    prefix = f"{tag} #{index}: "
    return prefix + (_FILLER * 5)[: 864 - len(prefix)]


class LoopTicker:
    """A 2 ms heartbeat; ``lags`` is how late each tick actually ran, in ms.

    Re-anchored every beat, so a stall shows up as ONE spike of its true
    duration instead of a cascade of catch-up zeros. Public for T8's gate, which
    imports this ticker rather than rolling its own (preflight ruling).
    """

    def __init__(self, interval: float = TICK_MS / 1000.0) -> None:
        self.interval = interval
        self.lags: list[float] = []
        self._task: asyncio.Task | None = None

    async def _beat(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            expected = loop.time() + self.interval
            await asyncio.sleep(self.interval)
            self.lags.append(max(loop.time() - expected, 0.0) * 1000.0)

    async def __aenter__(self) -> LoopTicker:
        self._task = asyncio.create_task(self._beat())
        return self

    async def __aexit__(self, *_exc) -> None:
        assert self._task is not None
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task

    def summary(self) -> str:
        ordered = sorted(self.lags)
        p50 = statistics.median(ordered)
        p99 = ordered[max(math.ceil(0.99 * len(ordered)) - 1, 0)]
        return (
            f"event-loop lag (n={len(ordered)}, {TICK_MS:g} ms heartbeat): "
            f"p50={p50:.2f}ms p99={p99:.2f}ms max={ordered[-1]:.2f}ms"
        )


# ── the real store, and the real embedder (nothing faked) ───────────────────


@pytest_asyncio.fixture
async def store(tmp_path, monkeypatch):
    """Private SQLite + private embedded Qdrant + ``USE_LOCAL_EMBEDDINGS`` arctic."""
    folder = tmp_path / "qdrant"
    folder.mkdir()
    monkeypatch.setattr(settings, "QDRANT_MODE", "local")
    monkeypatch.setattr(settings, "QDRANT_LOCAL_PATH", str(folder))
    monkeypatch.setattr(settings, "USE_LOCAL_EMBEDDINGS", True)
    monkeypatch.setattr(settings, "LOCAL_EMBED_MODEL", "arctic")
    # The knowledge-graph build is an LLM call, never part of this claim.
    monkeypatch.setattr("app.graph.builder.build_memory_graph_sync", lambda *a, **k: None)

    url = f"sqlite+aiosqlite:///{tmp_path / 'responsive.db'}"
    monkeypatch.setattr(settings, "DATABASE_URL", url)
    engine, sync_engine, sessions = _bind_engines(url, monkeypatch)
    await database.bootstrap_sqlite()
    async with sessions() as db:
        user = _user(db, "loop@gate.invalid")
        db.add(user)
        await db.commit()
        owner = user.id
    try:
        yield SimpleNamespace(sessions=sessions, owner=owner, qdrant_dir=folder, url=url)
    finally:
        await vector_backend.close_clients()
        await engine.dispose()
        sync_engine.dispose()


async def _seed_pending(store, contents: list[str]) -> list[uuid.UUID]:
    """Committed rows + their durable upsert intents, no vectors.

    Exactly the state the drain exists for (a write-through that did not run):
    the recall's barrier is what drains this batch, on the request path.
    """
    ids: list[uuid.UUID] = []
    async with store.sessions() as db:
        for content in contents:
            memory = _memory(store.owner, content)
            outbox.bump_revision(memory)
            db.add(memory)
            await outbox.enqueue_upsert(db, memory)
            ids.append(memory.id)
        await db.commit()
    return ids


async def _ingest(store, contents: list[str]) -> int:
    """The import shape: row + intent in one commit, then the write-through.

    Mirrors ``app.services.import_service`` — the caller whose in-band indexing
    must not hold the loop either.
    """
    rows: list[Memory] = []
    async with store.sessions() as db:
        for content in contents:
            memory = _memory(store.owner, content)
            outbox.bump_revision(memory)
            db.add(memory)
            await outbox.enqueue_upsert(db, memory)
            rows.append(memory)
        await db.commit()
        for memory in rows:
            await db.refresh(memory)
    landed = 0
    for memory in rows:
        if await index_new_memory(memory):
            landed += 1
    return landed


class _Upload:
    """The ``UploadFile`` surface ``upload_document`` reads, nothing more."""

    filename = "loop.txt"
    content_type = "text/plain"

    async def read(self) -> bytes:
        return b"loop gate bytes"


def _stub_out_of_process(monkeypatch) -> None:
    """MinIO, Redis and BM25 are out of process in production — stubbed here.

    Nothing about SQLite, the embedded Qdrant, the chunker or the embedder is.
    """

    async def _none(*_a, **_k):
        return None

    monkeypatch.setattr("app.storage.put_object", _none)
    monkeypatch.setattr("app.storage.get_object_sync", lambda *a, **k: b"loop gate bytes")
    monkeypatch.setattr("app.utils.chunker.extract_text", lambda *a, **k: _DOCUMENT_BODY)
    monkeypatch.setattr("app.retrieval.parent_store.store_parents_sync", lambda *a, **k: None)
    monkeypatch.setattr(
        "app.retrieval.bm25_retriever.bm25_retriever.publish_build_sync", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "app.retrieval.bm25_retriever.bm25_retriever.publish_rebuild_async", _none
    )
    monkeypatch.setattr("app.retrieval.retrieval_cache.invalidate_query_cache", _none)
    monkeypatch.setattr(
        "app.retrieval.retrieval_cache.invalidate_query_cache_sync", lambda *a, **k: None
    )


async def _upload(store) -> Document:
    """The REAL upload endpoint's service call: ingest a document, in-band."""
    from app.services.document_service import upload_document

    async with store.sessions() as db:
        conversation = Conversation(id=uuid.uuid4(), user_id=store.owner, document_count=0)
        db.add(conversation)
        await db.commit()
        return await upload_document(db, conversation, _Upload())  # type: ignore[arg-type]


async def _recall(store):
    """The REAL recall path: real barrier, real embed_query, real store.

    Only the LLM query rewrite (an out-of-process call) is stubbed.
    """
    from app.retrieval.memory import retriever as retriever_module
    from app.retrieval.memory.retriever import MemoryRetriever

    async def _rewrite(query, context=None, **_kwargs):
        return {"rewritten_query": query, "entities": [], "reasoning": None,
                "_fallback_used": False}

    original = retriever_module.rewrite_query
    retriever_module.rewrite_query = _rewrite
    try:
        async with store.sessions() as db:
            return await MemoryRetriever(db, store.owner, semantic_rerank=False).recall(QUERY)
    finally:
        retriever_module.rewrite_query = original


async def _intent_statuses(store) -> set[str]:
    async with store.sessions() as db:
        rows = (await db.execute(select(IndexOutbox))).scalars().all()
    return {row.status for row in rows}


def _point_count(kind: str) -> int:
    return int(vector_backend.get_sync_client().count(generation_name(kind)).count)


# ── 1. the heartbeat: recall + ingest + upload concurrently, model warm ─────


async def test_recall_ingest_and_upload_together_never_stall_the_loop(store, monkeypatch):
    """The barrier drains 50 intents on the recall path while 50 documents
    ingest AND a document with ~52 children is chunk-embedded — the loop beats.

    This is the load the signed p95 has to survive, not a synthetic one: every
    leg is a real production path over the real stores and the real model.
    """
    # Test-local budget, NOT a relaxation of the signed contract: this harness
    # runs three legs at once through the 2-worker embed executor, so the drain
    # queues behind the other legs' embeds (measured at intra-op=0: all 50
    # intents land at ~3.1 s — see the T1 report's fix round 1). The signed
    # 2.0 s RYW budget is a recall-alone measurement (1.15 s here, the P3 gate's
    # own shape) and is untouched. 10 s buys ~3x headroom on a slower box.
    monkeypatch.setattr(settings, "RECALL_FRESHNESS_BUDGET_SECONDS", 10.0)
    _stub_out_of_process(monkeypatch)
    # Warm BEFORE the ticker starts: the session build is boot's cost, not this
    # measurement's (the lifespan does exactly this — app/main.py).
    await warmup_embedder()
    await _seed_pending(store, [_text("backlog", i) for i in range(DRAIN_DOCS)])

    async with LoopTicker() as ticker:
        response, landed, document = await asyncio.gather(
            _recall(store),
            _ingest(store, [_text("live", i) for i in range(INGEST_DOCS)]),
            _upload(store),
        )

    print(ticker.summary())
    # The work really happened — the drain on the request path, the
    # write-through, and the whole ingestion pipeline.
    assert landed == INGEST_DOCS
    assert len(response.results) > 0
    async with store.sessions() as db:
        uploaded = await db.get(Document, document.id)
    assert uploaded.status == "ready", uploaded.error_msg
    assert _point_count("chunk") >= 50  # the document's children are indexed
    assert _point_count("memory") >= DRAIN_DOCS + INGEST_DOCS
    assert await _intent_statuses(store) == {"done"}

    assert len(ticker.lags) >= 50, "too few heartbeats to judge"
    assert max(ticker.lags) < MAX_LAG_MS, ticker.summary()
    ordered = sorted(ticker.lags)
    p99 = ordered[max(math.ceil(0.99 * len(ordered)) - 1, 0)]
    assert p99 < P99_LAG_MS, ticker.summary()


async def test_warmup_builds_the_session_before_the_first_embed(store, monkeypatch):
    """``warmup_embedder`` is what keeps the cold session off the first drain."""
    monkeypatch.setattr(e5_local, "_asess", None)
    await warmup_embedder()
    assert e5_local._asess is not None


# ── 2. the barrier really preempts an offloaded drain ───────────────────────


async def test_the_barrier_wait_for_bounds_an_offloaded_drain(store, monkeypatch):
    """A parked embedding must not hold the recall's barrier past its budget.

    The embed itself is parked in the executor thread: with the drain off the
    loop, ``asyncio.wait_for`` can cancel the drain's await and the barrier ends
    where the budget says it does — while the loop keeps beating.
    """
    budget = 0.3
    monkeypatch.setattr(settings, "RECALL_FRESHNESS_BUDGET_SECONDS", budget)
    await _seed_pending(store, [_text("parked", 0)])

    release, started, finished = threading.Event(), threading.Event(), threading.Event()
    real_embed = e5_local.arctic_embed_passages

    def parked_embed(texts):
        started.set()
        release.wait(5)  # the model is slow; nothing else may care
        try:
            return real_embed(texts)
        finally:
            finished.set()

    monkeypatch.setattr(e5_local, "arctic_embed_passages", parked_embed)

    async with LoopTicker() as ticker:
        began = time.perf_counter()
        with pytest.raises(IndexFreshnessTimeout):
            await freshness.await_freshness(user_id=str(store.owner), timeout=budget)
        elapsed = time.perf_counter() - began

    assert started.is_set(), "the drain never reached the embedding"
    print(f"barrier bounded a parked drain at {elapsed:.3f}s (budget {budget}s)")
    assert elapsed < budget + 0.6, f"the wait outlived its budget: {elapsed:.2f}s"
    assert max(ticker.lags) < MAX_LAG_MS, ticker.summary()

    release.set()  # unpark before teardown: no thread outlives the store
    assert await asyncio.to_thread(finished.wait, 10)


# ── 3. the signed RYW budget at the recall-ALONE shape (T1 review F1 → C1) ──


async def test_a_50_intent_backlog_drains_inside_the_signed_read_your_writes_budget(store):
    """C1/F1: recall ALONE drains the default 50-intent backlog inside 2.0 s.

    Ruling R9(p2) keeps the signed 2.0 s meaning for exactly this shape — a
    recall with no competing ingest. Nothing here softens the budget: the
    shipped setting is asserted and then measured, with the REAL embedder,
    because the drain embeds all 50 documents on the request path (that is
    what makes this number a real one). The concurrent three-leg shape above
    keeps its own, test-local budget; this is the release contract.
    """
    assert settings.RECALL_FRESHNESS_BUDGET_SECONDS == 2.0  # the signed budget
    await warmup_embedder()
    # Warm the drain path itself (first claim, first write) so the measured
    # run is the backlog's cost, not one-time session/coordination plumbing.
    await _seed_pending(store, [_text("warmup", 0)])
    assert len((await _recall(store)).results) > 0

    await _seed_pending(store, [_text("backlog", index) for index in range(DRAIN_DOCS)])

    began = time.perf_counter()
    response = await _recall(store)  # the barrier drains the backlog in-band
    elapsed = time.perf_counter() - began

    print(
        f"RYW recall-alone (n={DRAIN_DOCS} intents, real embedder): {elapsed:.3f}s "
        f"of the signed {settings.RECALL_FRESHNESS_BUDGET_SECONDS}s budget"
    )
    assert elapsed < settings.RECALL_FRESHNESS_BUDGET_SECONDS, (
        f"the recall-alone drain regressed past the signed budget: {elapsed:.3f}s"
    )
    assert await _intent_statuses(store) == {"done"}  # the backlog really landed
    assert len(response.results) > 0  # served, not an empty 200
    assert response.trace.stage_ms["queue_wait"] > 0.0  # a measured wait, not a no-op
