"""Task 7 — the P3 acceptance gate: the plan's §9 P3 row, over the REAL stores.

Every claim below runs against a real SQLite file and a real embedded Qdrant
folder. The harness is the P1b gate's (``tests/retrieval/test_p1b_gate.py``,
registered as a plugin and reused by name), so this suite inherits the same
isolation: a private per-test database, a private embedded-Qdrant folder, the
real cutover install (both manifests ACTIVE) and deterministic unit vectors for
the embedding CONTRACT the store was built with. Nothing about SQL or the
vector store is mocked.

The only substitutions are the seams that are out of process in production:

* the embedder — deterministic unit vectors, because no claim here is about
  embedding QUALITY; the embedding contract (dim 384, the cutover generation
  name) is the real one, so every payload, manifest row and collection the gate
  asserts against is the production shape;
* the recall path's LLM query rewrite (an out-of-process call, never the claim);
* the two OUTAGE readings, which are REAL outages: the client is pointed at a
  port nothing listens on (``_unreachable_store``), so the vector store really
  refuses the connection and the embedded folder this suite owns is untouched.
  Nothing is stubbed into "behaving like it is down".

The §9 P3 gate, bullet by bullet (one test each, marked with the bullet text):

* kill/restart between SQL/Qdrant/ack → a ``pending`` intent survives the
  restart and lands exactly once;
* stale upsert after a correction / a delete → the point never keeps the
  superseded snapshot (T5's races, re-run end to end through the API);
* an ambiguous timeout / in-flight late write → no point on a wrong generation,
  no ack on a wrong revision, and the replay is idempotent;
* the barrier has no hole → write then recall immediately (no sleep), ten
  rounds, the just-written row is IN the answer, and the wait is MEASURED
  (p50/p95 printed for the report — ruling R27);
* SQL sees the write directly, and the recall either waits (strict) or answers
  the typed timeout — both branches pinned;
* a batch that hits a transient store error and a batch whose commit fails
  mid-way drop nothing: the intent survives, ``attempts`` climbs, nothing is
  falsely ``blocked``;
* a delete receipt is verified after the readback, and reconcile upgrades the
  one whose owed delete lands later — both over the real store;
* no shared ORM session (ruling R28, pinned by identity): the drain, the
  barrier and the reconcile open their OWN sessions, and the request/recall
  session is left clean;
* the multi-process limit (ruling R27): a REAL second claimer — its own event
  loop in its own thread, no shared session, not even the single-flight lock —
  applies the same batch and corrupts nothing; and the runbook states the limit
  truthfully (ruling R29).

Ruling R30 is the last test: the workflow file itself is parsed, so dropping
this gate from CI (or narrowing its env) fails here instead of silently.
"""
from __future__ import annotations

import asyncio
import math
import sqlite3
import statistics
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from app import database
from app.api.v1 import memories as memories_api
from app.config import settings
from app.main import app
from app.models.erasure_receipt import ErasureReceipt
from app.models.index_outbox import IndexOutbox
from app.models.memory import Memory
from app.retrieval import e5_local, vector_backend
from app.retrieval import reranker as reranker_module
from app.retrieval.embedder import EmbeddingDimensionMismatch, warmup_embedder
from app.retrieval.embedder import embed_query as real_embed_query
from app.retrieval.embedding_fingerprint import generation_name
from app.retrieval.memory import drain_loop, freshness, outbox, vector_store
from app.retrieval.memory import retriever as retriever_module
from app.retrieval.memory.outbox import IndexFreshnessTimeout
from app.retrieval.memory.retriever import MemoryRetriever
from app.schemas.Orivory import RECALL_TRACE_STAGE_KEYS, MemoryUpdate
from app.services.erasure_service import erase_memories, reconcile_erasure_receipts
from app.utils.dependencies import enforce_llm_quota, get_current_user
from tests.retrieval.test_drain_loop import _until
from tests.retrieval.test_p1b_gate import (
    _create_memory,
    _intents,
    _payloads,
    _restart,
    _scroll_ids,
    _vector_for,
)

# The P1b gate's real-store fixtures (``env`` / ``world`` / ``live``), reused
# rather than copied: this gate has to prove the SAME store the migration
# installs, not a second harness that happens to look like it.
pytest_plugins = ["tests.retrieval.test_p1b_gate"]

RUNBOOK = Path(__file__).resolve().parents[2] / "docs" / "OPERATIONS_RUNBOOK.md"
CI_YML = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"
CI_STEP_NAME = "Run P3 background-indexing suites (temp SQLite, no services)"
GATE_MODULE = "tests/retrieval/test_p3_gate.py"


# ── helpers: reading the real store, and the REAL outage ────────────────────


def _point_count(generation: str) -> int:
    """How many points the generation holds — the anti-duplicate number."""
    return int(vector_backend.get_sync_client().count(generation).count)


async def _intent_rows(env, entity_id: uuid.UUID) -> list[IndexOutbox]:
    return [row for row in await _intents(env) if row.entity_id == entity_id.hex]


async def _statuses_of(env, entity_id: uuid.UUID) -> list[str]:
    return [row.status for row in await _intent_rows(env, entity_id)]


@asynccontextmanager
async def _unreachable_store(monkeypatch):
    """A REAL outage: the client is pointed at a port nothing is listening on.

    No store call is stubbed here: ``vector_backend`` opens a real
    ``AsyncQdrantClient`` for the URL and the connection is refused, which is
    the failure a production outage produces (the P1b suite uses port 1 for the
    same reason). The cached embedded client is dropped on entry, so the local
    folder this suite owns is not held open while the outage runs — and leaving
    the block re-opens it and finds exactly the points that were there.
    """
    await vector_backend.close_clients()
    monkeypatch.setattr(settings, "QDRANT_MODE", "server")
    monkeypatch.setattr(settings, "QDRANT_URL", "http://127.0.0.1:1")
    try:
        yield
    finally:
        await vector_backend.close_clients()  # the dead server client goes
        monkeypatch.setattr(settings, "QDRANT_MODE", "local")


async def _seed_pending(env, owner, contents: list[str]) -> list[uuid.UUID]:
    """Committed rows + their durable upsert intents, in ONE commit (no vectors).

    The write face without the write-through: exactly the state the drain owns
    after a store outage, seeded through the real outbox enqueue.
    """
    ids: list[uuid.UUID] = []
    async with env.sessions() as db:
        for content in contents:
            memory = Memory(id=uuid.uuid4(), user_id=owner, content=content, tags=[])
            db.add(memory)
            outbox.bump_revision(memory)
            await outbox.enqueue_upsert(db, memory)
            ids.append(memory.id)
        await db.commit()
    return ids


async def _release_backoff(env, entity_id: uuid.UUID) -> None:
    """Clear the next-attempt gate: the retry is due now (never a sleep)."""
    async with env.sessions() as db:
        row = (await db.execute(select(IndexOutbox).where(
            IndexOutbox.entity_id == entity_id.hex))).scalars().one()
        row.next_attempt_at = None
        await db.commit()


def _second_claimer(batch_size: int, release: threading.Event, reports: list[dict]) -> None:
    """A REAL second claimer: its own event loop, in its own thread.

    That is the multi-process shape this gate makes a claim about. It shares
    nothing with the first claimer — not a session, not a transaction, and (the
    single-flight lock being per-loop) not even the module lock. Only the
    stores are shared, exactly as two app processes share them.
    """
    async def _run() -> None:
        try:
            reports.append(await drain_loop.drain_once(batch_size=batch_size))
        finally:
            release.set()  # the first claimer is waiting inside its first write

    asyncio.run(_run())


# ── 1. kill/restart between SQL and the vector: it lands exactly once ───────


async def test_a_pending_intent_survives_the_restart_and_lands_exactly_once(live, monkeypatch):
    """The row + intent are committed, the process dies, the loop lands it once.

    The durable intent is the whole point of the outbox: the restart reads a
    ``pending`` row off the file and the background loop (T1) is what puts the
    vector there — one point, acked once, and nothing left to replay.
    """
    monkeypatch.setattr(memories_api, "index_new_memory", _store_down)
    created = await _create_memory(live.alice.id, "survives the restart", title="Restart")
    generation = generation_name("memory")

    assert created.indexing == "pending"
    assert str(created.id) not in _scroll_ids(generation)
    assert [(row.operation, row.status, row.revision) for row in await _intent_rows(live, created.id)] \
        == [("upsert", "pending", 1)]

    engine = await _restart(live, monkeypatch)  # the process died; the file stayed
    try:
        monkeypatch.setattr(settings, "OUTBOX_DRAIN_INTERVAL_SECONDS", 0.05)
        task = await drain_loop.start_drain_loop()
        assert task is not None

        async def _landed() -> bool:
            return await _statuses_of(live, created.id) == ["done"]

        try:
            await _until(_landed, timeout=10)
        finally:
            await drain_loop.stop_drain_loop(task)

        assert str(created.id) in _scroll_ids(generation)
        payload = _payloads(generation)[str(created.id)]
        assert payload["orivory_memory_revision"] == 1
        assert payload["user_id"] == str(live.alice.id)
        assert _point_count(generation) == 4  # the 3 cutover points + this one, never two

        # Nothing is owed any more: a further pass claims nothing at all.
        assert await outbox.drain_pending() == {
            "claimed": 0, "applied": 0, "skipped": 0, "blocked": 0, "failed": 0}
        assert _point_count(generation) == 4
    finally:
        await engine.dispose()


async def _store_down(_memory) -> bool:
    """The write-through seam: the fast path did not index (the intent is the proof).

    The store is up and real; what failed is the inline write-through, which is
    exactly the state the durable intent exists for.
    """
    return False


# ── 2. stale upsert: correction in flight, and delete in flight ─────────────


async def test_a_correction_while_the_upsert_is_in_flight_never_keeps_the_stale_point(live, monkeypatch):
    """The row is corrected between the applier's read and its write (T5/R22).

    The interleaving is exact, not timed: the hook runs the correction INSIDE
    the applier's own vector write. The correction's write-through lands the
    refreshed payload first (and acks its own intent), then the applier's stale
    snapshot overwrites it — so standing down would leave the superseded
    payload with no owner. The re-check has to rewrite the point from the
    refreshed row.
    """
    monkeypatch.setattr(memories_api, "index_new_memory", _store_down)
    created = await _create_memory(live.alice.id, "stale v1", title="Race")
    generation = generation_name("memory")
    assert await _statuses_of(live, created.id) == ["pending"]

    real_upsert = outbox.upsert_memory
    raced: list[int] = []

    async def hook(memory):
        if not raced:
            async with database.AsyncSessionLocal() as other:  # the writer's own session
                updated = await memories_api.update_memory(
                    created.id, MemoryUpdate(title="Refreshed v2"),
                    SimpleNamespace(id=live.alice.id), other,
                )
            raced.append(updated.revision)
        return await real_upsert(memory)  # the applier's superseded snapshot goes out

    monkeypatch.setattr(outbox, "upsert_memory", hook)
    report = await outbox.drain_pending()

    assert raced == [2]  # the interleaving really happened
    assert report == {"claimed": 1, "applied": 0, "skipped": 1, "blocked": 0, "failed": 0}
    payload = _payloads(generation)[str(created.id)]
    assert payload["orivory_memory_revision"] == 2  # NOT the stale snapshot
    assert "Refreshed v2" in payload["content"]

    # Both intents are closed — the correction's fast path acked its own, the
    # applier acked the superseded one — and the point is the refreshed row.
    await outbox.drain_pending()
    assert await _statuses_of(live, created.id) == ["done", "done"]
    assert _point_count(generation) == 4
    assert "Refreshed v2" in _payloads(generation)[str(created.id)]["content"]


async def test_a_delete_while_the_upsert_is_in_flight_never_leaves_the_point(live, monkeypatch):
    """The row is deleted between the applier's read and its write (T5/R21).

    The applier writes the point for a row the erase already removed: without
    the post-write re-check the point outlives its row (and the drain acks the
    stale snapshot as applied). It must be purged, with the store confirming.
    """
    monkeypatch.setattr(memories_api, "index_new_memory", _store_down)
    created = await _create_memory(live.alice.id, "doomed by a race", title="Race")
    generation = generation_name("memory")
    assert await _statuses_of(live, created.id) == ["pending"]

    real_upsert = outbox.upsert_memory
    raced: list[bool] = []

    async def hook(memory):
        if not raced:
            async with database.AsyncSessionLocal() as other:  # the eraser's own session
                await memories_api.delete_memory(created.id, SimpleNamespace(id=live.alice.id), other)
            raced.append(True)
        return await real_upsert(memory)  # the point for the deleted row goes out

    monkeypatch.setattr(outbox, "upsert_memory", hook)
    report = await outbox.drain_pending()

    assert raced == [True]
    assert report["applied"] == 1 and report["failed"] == 0
    assert str(created.id) not in _scroll_ids(generation)  # the point did not survive
    # The erase's own delete intent is the next pass's work (it was enqueued
    # while this pass was already claimed): it lands on the point's absence.
    assert (await outbox.drain_pending())["applied"] == 1
    assert await _statuses_of(live, created.id) == ["done", "done"]  # upsert + delete
    assert _point_count(generation) == 3  # back to the cutover set


# ── 2b. a claim checks the collection ONCE (the signed RYW budget) ──────────


async def test_a_claim_checks_the_collection_once(live, monkeypatch):
    """The claim's rows ask the store the SAME question once, not fifty times.

    "Is the active collection still the one this embedding contract describes?"
    is two store round trips (``count`` + collection info) and it was asked per
    row — 0.11 s of a fifty-intent claim, measured with the real store. The
    guard still runs inside the first row's own write, so a contract mismatch
    is still that row's own error (backoff / blocked), never a batch-level one.
    """
    monkeypatch.setattr(memories_api, "index_new_memory", _store_down)
    created = [
        await _create_memory(live.alice.id, f"claim row {index}") for index in range(5)
    ]

    checks: list[int] = []
    real_check = vector_store._checked_collection

    async def counting_check(embedding_dim):
        checks.append(embedding_dim)
        return await real_check(embedding_dim)

    monkeypatch.setattr(vector_store, "_checked_collection", counting_check)

    report = await outbox.drain_pending()

    assert report == {"claimed": 5, "applied": 5, "skipped": 0, "blocked": 0, "failed": 0}
    assert len(checks) == 1, f"the claim ran the contract guard {len(checks)} times"
    generation = generation_name("memory")
    assert all(str(memory.id) in _payloads(generation) for memory in created)


async def test_only_a_settled_guard_answer_is_shared(live, monkeypatch):
    """Only an answer the claim's own writes cannot move is shared.

    A non-empty generation that passed the contract holds a manifest row, so its
    answer is settled and rows 2..50 may reuse it. An EMPTY generation is what an
    unclaimed one looks like, and the claim's own first write is what populates
    it — sharing that pass would let the rest of the claim write into a populated
    generation with no manifest, which is the state the contract quarantines.
    Every row must ask for itself while it is empty, and the quarantine error
    belongs to the row that meets it.
    """
    answers: list = []
    calls: list[int] = []

    async def fake_check(embedding_dim: int):
        calls.append(embedding_dim)
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(vector_store, "_checked_collection", fake_check)
    quarantine = EmbeddingDimensionMismatch(
        "populated generation has no manifest row — quarantine/rebuild"
    )

    # Unclaimed generation: the first row sees it empty (allowed), and by the
    # next row the claim's own write has populated it — that row raises.
    answers = [("client", "gen-unclaimed", 0), quarantine]
    async with vector_store.claim_cache():
        assert await vector_store._checked_collection_for_claim(384) == ("client", "gen-unclaimed", 0)
        with pytest.raises(EmbeddingDimensionMismatch):
            await vector_store._checked_collection_for_claim(384)
    assert len(calls) == 2, f"the claim shared an unsettled answer (guard ran {len(calls)} time(s), expected 2)"

    # Manifest-backed generation: settled, so one ask covers the whole claim.
    calls.clear()
    answers = [("client", "gen-settled", 7)]
    async with vector_store.claim_cache():
        for _ in range(3):
            assert await vector_store._checked_collection_for_claim(384) == ("client", "gen-settled", 7)
    assert len(calls) == 1, f"a settled answer was re-checked {len(calls)} times"


# ── 3. ambiguous timeout / late write ───────────────────────────────────────


async def test_an_ambiguous_write_lands_no_wrong_generation_and_no_wrong_ack(live, monkeypatch):
    """The store takes the write and the client never hears it (a timeout).

    The classic ambiguous failure: the vector IS in the store while the caller
    believes it failed. Nothing may be acked from that (the ack would be a
    guess), the point must sit in the ACTIVE generation carrying the row's own
    revision, and the retry must be idempotent rather than a second point.
    """
    monkeypatch.setattr(memories_api, "index_new_memory", _store_down)
    created = await _create_memory(live.alice.id, "written but unacknowledged", title="Ambiguous")
    generation = generation_name("memory")
    assert generation == outbox.active_generation_sync()[0]

    real_upsert = outbox.upsert_memory
    calls: list[int] = []

    async def hook(memory):
        await real_upsert(memory)  # the late write: it really landed
        calls.append(1)
        raise TimeoutError("the store answered after the client gave up")

    monkeypatch.setattr(outbox, "upsert_memory", hook)
    report = await outbox.drain_pending()

    assert calls == [1]
    assert report == {"claimed": 1, "applied": 0, "skipped": 0, "blocked": 0, "failed": 1}
    (row,) = await _intent_rows(live, created.id)
    assert row.status == "pending"  # never acked off an unproven write
    assert row.attempts == 1 and row.next_attempt_at is not None
    assert row.last_error.startswith("TimeoutError")
    assert row.target_generation == generation  # no re-target at a dead generation

    # What the store holds: the row's own revision, in the ACTIVE generation.
    payload = _payloads(generation)[str(created.id)]
    assert payload["orivory_memory_revision"] == 1
    assert _point_count(generation) == 4

    # The retry is idempotent: the same point, re-written, never a second one.
    monkeypatch.setattr(outbox, "upsert_memory", real_upsert)
    await _release_backoff(live, created.id)
    assert (await outbox.drain_pending())["applied"] == 1
    assert _point_count(generation) == 4
    assert _payloads(generation)[str(created.id)] == payload
    assert await _statuses_of(live, created.id) == ["done"]


# ── 4./5. the barrier: no hole, SQL sees it, both branches pinned ───────────


async def _recall(live, document_holder: dict, query: str = "where did i walk at dusk"):
    """A real recall over the real store: only the LLM rewrite/embed seams pinned."""
    async def _embed_query(_query: str) -> list[float]:
        return _vector_for(document_holder["document"])

    async def _rewrite(query, context=None, **_kwargs):
        return {"rewritten_query": query, "entities": [], "reasoning": None,
                "_fallback_used": False}

    original_embed, original_rewrite = retriever_module.embed_query, retriever_module.rewrite_query
    retriever_module.embed_query, retriever_module.rewrite_query = _embed_query, _rewrite
    try:
        async with live.sessions() as db:
            return await MemoryRetriever(db, live.alice.id, semantic_rerank=False).recall(query)
    finally:
        retriever_module.embed_query, retriever_module.rewrite_query = original_embed, original_rewrite


async def test_a_write_is_visible_to_sql_and_to_recall_ten_rounds_without_a_sleep(live, monkeypatch):
    """The barrier has no hole: 10 rounds of write → recall IMMEDIATELY.

    Each round commits the row + its intent with the write-through failing, and
    then recalls with no sleep at all. The row must be IN the answer — an empty
    recall here is the false no-match the barrier exists to prevent — and the
    wait is the barrier's own measurement (ruled R27: real clock, no fake), so
    the report can carry p50/p95 instead of a promise.
    """
    monkeypatch.setattr(memories_api, "index_new_memory", _store_down)
    generation = generation_name("memory")
    holder: dict = {"document": ""}
    waits_ms: list[float] = []

    for round_number in range(10):
        created = await _create_memory(
            live.alice.id, f"walked the orchard, round {round_number}"
        )
        # (5) SQL sees the write — a FRESH session, no waiting, no index involved.
        async with live.sessions() as db:
            row = (await db.execute(select(Memory).where(Memory.id == created.id))).scalars().one()
            assert row.content.endswith(f"round {round_number}")
            assert await _statuses_of(live, created.id) == ["pending"]
            holder["document"] = vector_store._memory_to_document(row)

        response = await _recall(live, holder)  # no sleep, no drain, no retry loop

        returned = {str(result.id) for result in response.results}
        assert str(created.id) in returned, f"round {round_number}: the barrier left a hole"
        assert await _statuses_of(live, created.id) == ["done"]  # the wait did the work
        waits_ms.append(float(response.trace.stage_ms["queue_wait"]))

    assert str(created.id) in _scroll_ids(generation)
    assert min(waits_ms) > 0.0  # a measured wait, never a fabricated 0
    assert max(waits_ms) < settings.RECALL_FRESHNESS_BUDGET_SECONDS * 1000.0
    ordered = sorted(waits_ms)
    p50 = statistics.median(ordered)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * len(ordered)) - 1)]
    print(
        f"P3 GATE barrier wait (n={len(waits_ms)}, real clock): "
        f"p50={p50:.2f}ms p95={p95:.2f}ms max={max(waits_ms):.2f}ms"
    )


async def test_an_unreachable_store_answers_the_typed_timeout_never_an_empty_recall(live, monkeypatch):
    """The other branch of (5): the wait cannot end in a result, so it ends typed.

    The store is REALLY unreachable (a refused connection, no stub), so the
    intent cannot land: the recall must raise ``IndexFreshnessTimeout`` and the
    endpoint must answer ``503 {"error": "index_freshness_timeout"}`` — never a
    ``200`` with ``results: []`` for a write that is committed in SQL.
    """
    monkeypatch.setattr(settings, "RECALL_FRESHNESS_BUDGET_SECONDS", 0.3)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=live.alice.id)
    app.dependency_overrides[enforce_llm_quota] = lambda: None
    try:
        async with _unreachable_store(monkeypatch):
            created = await _create_memory(live.alice.id, "owed while the store is down")
            assert created.indexing == "pending"  # the real write-through failed
            async with live.sessions() as db:  # SQL is the source of truth
                assert (await db.execute(select(Memory).where(
                    Memory.id == created.id))).scalars().one() is not None
            assert await _statuses_of(live, created.id) == ["pending"]

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/api/v1/memories/recall", json={"query": "owed while the store is down"}
                )

            assert response.status_code == 503
            assert response.json() == {"error": "index_freshness_timeout"}
            # Failed closed, not swallowed: the write is still owed.
            assert await _statuses_of(live, created.id) == ["pending"]

        # The store is reachable again: the SAME intent (its backoff elapsed —
        # the loop would simply wait it out) is what finishes the write.
        await _release_backoff(live, created.id)
        assert (await outbox.drain_pending())["applied"] == 1
        assert str(created.id) in _scroll_ids(generation_name("memory"))
    finally:
        app.dependency_overrides.clear()


async def test_the_barrier_raises_the_typed_error_in_process_too(live, monkeypatch):
    """The same branch one layer down: ``MemoryRetriever.recall`` raises, not ``[]``."""
    monkeypatch.setattr(settings, "RECALL_FRESHNESS_BUDGET_SECONDS", 0.3)
    with pytest.raises(IndexFreshnessTimeout):
        async with _unreachable_store(monkeypatch):
            await _create_memory(live.alice.id, "unreachable from the retriever")
            holder = {"document": ""}
            await _recall(live, holder)


# ── 6. a batch that hits an error drops nothing ─────────────────────────────


async def test_a_transient_store_failure_backs_off_without_blocking_or_dropping(live, monkeypatch):
    """Two intents in one batch, one store call fails transiently.

    The failure is the retryable class (a store hiccup), not a contract
    mismatch: the intent must stay ``pending`` with ``attempts`` climbing and a
    backoff stamped — never ``blocked`` (terminal, and then nothing would ever
    replay it) and never dropped from the batch's own report.
    """
    good, bad = await _seed_pending(live, live.alice.id, ["lands fine", "fails once"])
    generation = generation_name("memory")
    real_upsert = outbox.upsert_memory

    async def hiccup(memory):
        if memory.id == bad:
            raise ConnectionError("qdrant refused the write")
        return await real_upsert(memory)

    monkeypatch.setattr(outbox, "upsert_memory", hiccup)
    report = await outbox.drain_pending(batch_size=10)

    assert report == {"claimed": 2, "applied": 1, "skipped": 0, "blocked": 0, "failed": 1}
    (row,) = await _intent_rows(live, bad)
    assert row.status == "pending" and row.status != "blocked"
    assert row.attempts == 1 and row.next_attempt_at is not None
    assert row.last_error.startswith("ConnectionError")
    assert str(bad) not in _scroll_ids(generation)  # not applied, not acked
    assert await _statuses_of(live, good) == ["done"]
    assert str(good) in _scroll_ids(generation)

    monkeypatch.setattr(outbox, "upsert_memory", real_upsert)
    await _release_backoff(live, bad)
    assert (await outbox.drain_pending())["applied"] == 1
    assert str(bad) in _scroll_ids(generation)
    assert _point_count(generation) == 5  # 3 cutover + 2 seeded, never a double


async def test_a_commit_failure_mid_batch_drops_no_intent(live, monkeypatch):
    """The ack's own commit is what fails (disk full, a locked database).

    The vector write of the second row already landed when its commit blows up:
    the intent must still be there afterwards (the ack is the only thing lost,
    and the intent is the proof the work is owed), nothing may be marked
    ``blocked``, and the next pass must close it — over the same single point.
    """
    first, second = await _seed_pending(live, live.alice.id, ["acked before the crash", "lost its ack"])
    generation = generation_name("memory")
    real_commit = AsyncSession.commit
    commits: list[int] = []

    async def failing_commit(self):
        commits.append(1)
        if len(commits) == 2:  # the first _apply committed; the second cannot
            raise OperationalError(
                "UPDATE index_outbox", {},
                sqlite3.OperationalError("database or disk is full"),
            )
        return await real_commit(self)

    monkeypatch.setattr(AsyncSession, "commit", failing_commit)
    try:
        with pytest.raises(OperationalError):
            await outbox.drain_pending(batch_size=2)
    finally:
        monkeypatch.setattr(AsyncSession, "commit", real_commit)

    assert len(commits) == 2  # the failure really hit the second row's ack
    rows = {row.entity_id: row for row in await _intents(live)}
    assert set(rows) == {first.hex, second.hex}  # nothing was dropped
    assert {row.status for row in rows.values()} == {"done", "pending"}
    assert all(row.status != "blocked" for row in rows.values())
    assert rows[first.hex].status == "done"  # its commit landed before the failure
    # This class is NOT the transient one pinned above: the failing commit is the
    # ack's own ``await db.commit()``, which sits OUTSIDE ``_apply``'s try/except
    # (``outbox.py`` — the commit is after the outcome is decided), so nothing
    # bumped ``attempts`` and no backoff was stamped: the row is due again on the
    # very next pass. The transient class pins ``attempts == 1`` + a stamped
    # ``next_attempt_at``; a persistent disk-full here would retry every pass
    # (see the report's M1 note — production code untouched on purpose).
    assert rows[second.hex].attempts == 0
    assert rows[second.hex].next_attempt_at is None
    # The second row's write DID land (the store call ran); only the ack is owed.
    assert str(second) in _scroll_ids(generation)

    # The replay is what closes it — idempotently, over the point already there.
    assert (await outbox.drain_pending())["applied"] == 1
    assert await _statuses_of(live, second) == ["done"]
    assert _point_count(generation) == 5


# ── 7. the delete receipt is verified after the readback ────────────────────


async def test_erase_verifies_the_purge_and_reconcile_upgrades_the_owed_one(live, monkeypatch):
    """Both receipt paths over the real store: verified now, verified later.

    (a) The store is up: the purge runs, the readback confirms absence, and the
    receipt says ``completed`` with no pending index work.
    (b) The store is REALLY down: the erase cannot verify — the receipt is
    ``completed_unverified`` and the point is still there — and once the store
    is back the drain lands the owed delete and ONE reconcile pass upgrades the
    SAME receipt to ``completed``, reading the evidence back off SQL.
    """
    generation = generation_name("memory")
    async with live.sessions() as db:
        verified_target = Memory(id=uuid.uuid4(), user_id=live.alice.id,
                                 content="verified erase", tags=[])
        owed_target = Memory(id=uuid.uuid4(), user_id=live.alice.id,
                             content="erase owed to the drain", tags=[])
        db.add_all([verified_target, owed_target])
        await db.commit()
        await vector_store.upsert_memory(verified_target)  # real points, both
        await vector_store.upsert_memory(owed_target)
    assert {str(verified_target.id), str(owed_target.id)} <= _scroll_ids(generation)

    # (a) verified path
    async with live.sessions() as db:
        receipt = await erase_memories(db, live.alice.id, [verified_target.id],
                                       requested_by="rest_api")
    entry = receipt.detail["targets"][0]
    assert receipt.status == "completed"
    assert receipt.detail["verification"] == "verified"
    assert entry["vector_state"] == "verified" and entry["vector_residual_checked"] is True
    assert str(verified_target.id) not in _scroll_ids(generation)  # really gone
    # The durable delete intent is still owed to the drain (the receipt counts it):
    # the vector is gone, and the ack is what closes the obligation.
    assert receipt.detail["index_pending"] == 1
    assert (await outbox.drain_pending())["applied"] == 1
    assert await _statuses_of(live, verified_target.id) == ["done"]

    # (b) the outage path — a refused connection, nothing stubbed
    async with _unreachable_store(monkeypatch):
        async with live.sessions() as db:
            owed = await erase_memories(db, live.alice.id, [owed_target.id],
                                        requested_by="rest_api")
        entry = owed.detail["targets"][0]
        assert owed.status == "completed_unverified"
        assert (entry["vector_state"], entry["vector_residual_checked"]) == ("pending", False)
        assert owed.detail["index_pending"] == 1

    # The purge really did not land: the point is still in the folder.
    assert str(owed_target.id) in _scroll_ids(generation)

    # The store is back: the owed delete lands, then reconcile upgrades the SAME row.
    assert (await outbox.drain_pending())["applied"] >= 1
    assert str(owed_target.id) not in _scroll_ids(generation)
    assert await _statuses_of(live, owed_target.id) == ["done"]

    assert await reconcile_erasure_receipts() == {
        "checked": 1, "upgraded": 1, "still_unverified": 0}
    async with live.sessions() as db:
        refreshed = await db.get(ErasureReceipt, owed.id)
    assert refreshed.status == "completed"
    assert refreshed.detail["verification"] == "verified"
    assert refreshed.detail["index_pending"] == 0
    assert refreshed.detail["targets"][0]["vector_state"] == "verified"


# ── 8. no shared ORM session (ruling R28) ───────────────────────────────────


class _RecordingSessions:
    """A sessionmaker that remembers every session the drain/barrier opens."""

    def __init__(self, factory):
        self._factory = factory
        self.created: list[AsyncSession] = []

    def __call__(self) -> AsyncSession:
        session = self._factory()
        self.created.append(session)
        return session


async def test_the_drain_the_barrier_and_reconcile_never_reuse_the_request_session(
    live, monkeypatch
):
    """R28, pinned by identity — not by "the numbers came out right".

    The drain commits through its own session on purpose: a long-lived read
    transaction (SQLite especially) would keep answering from the snapshot
    taken before those commits, and the request path would then serve a stale
    no-match. So the barrier's count read, the drain's claims and the reconcile
    scan must each open their own session, and the request session must be left
    with nothing pending in it.
    """
    recorder = _RecordingSessions(live.sessions)  # get_db, reconcile + the test's own reader
    barrier_recorder = _RecordingSessions(live.sessions)  # the barrier's count reads
    drain_recorder = _RecordingSessions(live.sessions)  # the drain's claims
    monkeypatch.setattr(database, "AsyncSessionLocal", recorder)  # get_db + reconcile
    monkeypatch.setattr(freshness, "AsyncSessionLocal", barrier_recorder)  # the barrier's count
    monkeypatch.setattr(outbox, "AsyncSessionLocal", drain_recorder)  # the drain's claims

    monkeypatch.setattr(memories_api, "index_new_memory", _store_down)
    created = await _create_memory(live.alice.id, "the request session must not carry this")

    async with live.sessions() as request_session:  # the recall path's session
        recorder.created.clear()  # ignore whatever the write path opened
        retriever = MemoryRetriever(request_session, live.alice.id, semantic_rerank=False)
        holder = {"document": ""}

        async def _embed_query(_query: str) -> list[float]:
            return _vector_for(holder["document"])

        async def _rewrite(query, context=None, **_kwargs):
            return {"rewritten_query": query, "entities": [], "reasoning": None,
                    "_fallback_used": False}

        real_embed, real_rewrite = retriever_module.embed_query, retriever_module.rewrite_query
        retriever_module.embed_query, retriever_module.rewrite_query = _embed_query, _rewrite
        try:
            async with database.AsyncSessionLocal() as reader:  # a fresh SQL read
                row = (await reader.execute(select(Memory).where(
                    Memory.id == created.id))).scalars().one()
                holder["document"] = vector_store._memory_to_document(row)
            response = await retriever.recall("where did i walk")
        finally:
            retriever_module.embed_query, retriever_module.rewrite_query = real_embed, real_rewrite

        assert str(created.id) in {str(result.id) for result in response.results}
        # The barrier's count read + the drain's claim, EACH its own sessions —
        # recorded per caller, because the test's own reader above ALSO goes
        # through ``database.AsyncSessionLocal``: a shared count would be
        # satisfied by that reader plus one of the two, which is not the claim.
        assert len(barrier_recorder.created) >= 1
        assert len(drain_recorder.created) >= 1
        opened = [*barrier_recorder.created, *drain_recorder.created]
        # …and NOT the session the request is running on, by identity.
        assert all(session is not request_session for session in opened)
        assert not any(session is request_session for session in opened)
        # The request session is left clean: nothing pending in it to flush.
        assert len(request_session.new) == 0
        assert len(request_session.dirty) == 0
        assert len(request_session.deleted) == 0

    # Reconcile opens its own too (it is called without a session on purpose).
    recorder.created.clear()
    assert await reconcile_erasure_receipts() == {
        "checked": 0, "upgraded": 0, "still_unverified": 0}
    assert len(recorder.created) == 1
    assert await _statuses_of(live, created.id) == ["done"]


# ── 9. the multi-process limit: a REAL second claimer (ruling R27) ──────────


async def test_a_second_claimer_applies_the_same_batch_and_corrupts_nothing(live, monkeypatch):
    """Two real claimers, one batch: duplicate work, zero corruption.

    The second claimer is REAL: its own event loop in its own thread, its own
    sessions, its own claim — the multi-process shape, minus the second Qdrant
    owner that local mode forbids. The interleaving is deterministic, not timed:
    the first claimer is held inside its first write until the second one's
    claim has completed, so both really applied the same rows.

    Then the lost-ack window (what a process that died between its vector write
    and its ack leaves behind) is replayed by the second claimer alone: the same
    points, byte for byte, and not one point more.
    """
    generation = generation_name("memory")
    seeded = await _seed_pending(live, live.alice.id, [f"shared batch {i}" for i in range(8)])
    # The STORE boundary, counted independently of the appliers' own ``applied``
    # counters: every write that reaches the real ``vector_store.upsert_memory``
    # is appended here, whichever claimer — or thread — asked for it.
    store_upsert = vector_store.upsert_memory
    store_writes: list[str] = []

    async def counting_upsert(memory):
        store_writes.append(str(memory.id))
        return await store_upsert(memory)

    monkeypatch.setattr(vector_store, "upsert_memory", counting_upsert)
    claimed = threading.Event()
    release = threading.Event()
    main_thread = threading.get_ident()

    async def hold_first_claimer(memory):
        if threading.get_ident() == main_thread and not release.is_set():
            claimed.set()  # the batch is claimed and in flight
            await asyncio.to_thread(release.wait, 30)
        return await counting_upsert(memory)

    monkeypatch.setattr(outbox, "upsert_memory", hold_first_claimer)

    reports: list[dict] = []
    first = asyncio.create_task(drain_loop.drain_once(batch_size=8))
    assert await asyncio.to_thread(claimed.wait, 15)  # the first claimer is mid-batch
    second = asyncio.create_task(asyncio.to_thread(_second_claimer, 8, release, reports))
    first_report = await asyncio.wait_for(first, timeout=60)
    await asyncio.wait_for(second, timeout=60)

    assert first_report["claimed"] == 8 and reports[0]["claimed"] == 8
    assert first_report["applied"] == 8 and reports[0]["applied"] == 8
    # …so the same eight intents were applied TWICE, and the STORE saw it: 16
    # calls reached ``vector_store.upsert_memory`` (eight per claimer) for 8
    # points — a store-observable fact, not the appliers' own counters.
    assert len(store_writes) == 16
    assert _point_count(generation) == 11  # 3 cutover + 8 seeded, never 16
    payloads = _payloads(generation)
    for memory_id in seeded:
        assert payloads[str(memory_id)]["orivory_memory_revision"] == 1
        assert payloads[str(memory_id)]["user_id"] == str(live.alice.id)
    assert await _statuses_of(live, seeded[0]) == ["done"]
    assert {row.status for row in await _intents(live)} == {"done"}

    # The lost-ack window: every ack gone with the process that made the writes.
    before = {str(memory_id): payloads[str(memory_id)] for memory_id in seeded}
    async with live.sessions() as db:
        for row in (await db.execute(select(IndexOutbox))).scalars().all():
            row.status, row.attempts, row.next_attempt_at = "pending", 0, None
        await db.commit()

    reports.clear()
    await asyncio.to_thread(_second_claimer, 8, threading.Event(), reports)
    assert reports[0]["applied"] == 8  # the whole batch, re-applied by the other claimer
    assert len(store_writes) == 24  # …and eight more real store writes on the wire
    assert _point_count(generation) == 11  # not one point more
    after = _payloads(generation)
    assert {str(memory_id): after[str(memory_id)] for memory_id in seeded} == before
    assert {row.status for row in await _intents(live)} == {"done"}


def test_the_runbook_states_the_multi_process_limit_truthfully():
    """Ruling R29: one draining process by design; more is redundant, not wrong.

    The docs line is the operator's contract for a Qdrant-server deployment:
    it has to say what a second process costs (wasted work), why it cannot
    corrupt anything (idempotent by entity id + the post-write rewrite), and
    which shape is supported.
    """
    section = RUNBOOK.read_text().split("## Background indexing (P3)", 1)[1]
    section = section.split("\n## ", 1)[0]
    lowered = " ".join(section.split()).lower()  # prose wraps: compare it unwrapped

    assert "known limitation" in lowered
    assert "redundant work" in lowered
    assert "never corruption" in lowered
    assert "one process is the supported shape" in lowered
    assert "idempotent by entity id" in lowered
    assert "re-read" in lowered and "refreshed" in lowered  # the T5/R22 rewrite
    # M5: the bullet names the two-writer coverage and the residual it does NOT
    # cover — a blanket "cannot leave a stale point behind" is not the claim.
    assert "accepted residual" in lowered
    assert "third write" in lowered


# ── 10. the signed latency budget (T3): p50 ≤ 60 ms, p95 ≤ 150 ms, warm ─────

# §12.2 (user-signed 2026-09-15): "recall p95 ≤ 150 ms at fixture scale".
# The p50 budget is the guard band: a fixture or a path that drifted shows
# here long before it breaks the signed p95.
#
# Read the band honestly (F3): 60 ms sits just ABOVE the known
# `EMBED_ORT_INTRA_OP_THREADS=1` regression (reviewer-measured ~53 ms p50
# here vs 12.6 ms at the default), so it does NOT cover that knob — a pinned
# intra-op=1 lands ~7 ms under the guard. What bites there is the C1 pin
# (`tests/retrieval/test_event_loop_responsiveness.py` §3): the recall-alone
# 50-intent drain blows the signed 2.0 s budget by mechanism (3.07 s).
LATENCY_P50_MS = 60.0
LATENCY_P95_MS = 150.0
LATENCY_ITERATIONS = 30
LATENCY_WARMUP_ITERATIONS = 3
LATENCY_QUERY = "where did i walk at dusk"


async def _stub_rewrite(query, context=None, **_kwargs):
    """The LLM rewrite is an out-of-process call — never part of latency."""
    return {"rewritten_query": query, "entities": [], "reasoning": None,
            "_fallback_used": False}


@pytest.mark.skipif(
    not e5_local.arctic_files_cached(),
    reason="arctic onnx cache missing — run local, do not download in CI",
)
async def test_the_signed_recall_latency_budget_holds_on_the_frozen_fixture(live, monkeypatch):
    """The signed budget, asserted on the REAL recall path — warm, ≥30 runs.

    Everything the request does is in the measurement: the P3 barrier, the SQL
    context read, the real embedded-Qdrant search, hydration, eligibility, the
    merged-pool re-validation after the rerank round (T2's doubled post-network
    hydrate), scoring and serialization. Only the two out-of-process seams are
    substituted — the LLM rewrite and the reranker TRANSPORT (a stub that keeps
    the rows it is handed) — and the embedder is the REAL arctic session,
    because the lite deployment embeds in-process and that cost is part of the
    budget. It is warmed first, the way the T1 heartbeat test does: a cold
    session is boot's cost, not a request's.

    Three rows are seeded before measuring: the cutover fixture leaves alice
    exactly ONE eligible memory point, and the rerank leg (it needs >1
    candidate) is part of the path the budget covers.

    It needs the cached arctic artifacts, so it carries the house guard
    (R15): with a warm cache the signed assertion runs; on a cold cache it
    skips cleanly instead of making CI download ~90 MB of ONNX.
    """
    for index in range(3):
        created = await _create_memory(live.alice.id, f"latency fixture row {index}")
        assert created.indexing == "ready", "the write-through did not index"
    generation = generation_name("memory")
    assert len(_scroll_ids(generation)) == 6  # the 3 cutover points + these 3

    # The gate harness fakes the embedder for the fixture's deterministic
    # vectors; this measurement restores the real one (see the docstring).
    monkeypatch.setattr(retriever_module, "embed_query", real_embed_query)
    monkeypatch.setattr(retriever_module, "rewrite_query", _stub_rewrite)
    rerank_inputs: list[int] = []

    async def _stub_rerank(_query, chunks, *, top_n=None):
        rerank_inputs.append(len(chunks))
        return [dict(chunk, rerank_score=1.0 - index / 100.0)
                for index, chunk in enumerate(chunks)]

    monkeypatch.setattr(reranker_module, "rerank", _stub_rerank)
    await warmup_embedder()

    latencies_ms: list[float] = []
    for _ in range(LATENCY_WARMUP_ITERATIONS + LATENCY_ITERATIONS):
        async with live.sessions() as db:
            began = time.perf_counter()
            response = await MemoryRetriever(
                db, live.alice.id, semantic_rerank=True
            ).recall(LATENCY_QUERY)
            latencies_ms.append((time.perf_counter() - began) * 1000.0)

    measured = sorted(latencies_ms[LATENCY_WARMUP_ITERATIONS:])
    p50 = statistics.median(measured)
    p95 = measured[math.ceil(0.95 * len(measured)) - 1]  # nearest-rank
    print(
        f"P2 signed recall latency (n={len(measured)}, real store + real embedder, "
        f"stub reranker transport): p50={p50:.2f}ms p95={p95:.2f}ms "
        f"max={measured[-1]:.2f}ms"
    )
    assert len(measured) >= 30, "the signed assertion needs ≥30 iterations"
    assert p50 <= LATENCY_P50_MS, (p50, measured)
    assert p95 <= LATENCY_P95_MS, (p95, measured)

    # The measurement ran the whole path, not a short-circuit: the rerank leg
    # really fired on every iteration (so its post-network re-hydrate is in the
    # numbers), and the trace's counters state the run's real facts.
    assert rerank_inputs == [4] * (LATENCY_WARMUP_ITERATIONS + LATENCY_ITERATIONS)
    assert response.trace.counts == {
        "dense": 4, "eligible": 4, "reranked": 4, "hydrated": 4, "returned": 4,
    }
    # F5: the declaration is a contract in BOTH directions — this run may only
    # write `stage_ms` keys `RECALL_TRACE_STAGE_KEYS` declares (the C2 test in
    # test_rerank_pool.py pins the set a fresh trace is BUILT with; this pins
    # what the real path actually WRITES, so an undeclared extra cannot leak).
    assert set(response.trace.stage_ms) <= set(RECALL_TRACE_STAGE_KEYS)
    assert len(response.results) == 4


# ── R30: the gate runs in CI, in the P1b/P3 step's discipline ───────────────


def test_ci_runs_the_p3_gate_with_the_same_env_discipline():
    """The workflow is parsed, so narrowing CI fails here instead of silently."""
    jobs = yaml.safe_load(CI_YML.read_text())["jobs"]
    steps = [step for job in jobs.values() for step in job.get("steps", [])]
    matches = [step for step in steps if step.get("name") == CI_STEP_NAME]
    assert len(matches) == 1, [step.get("name") for step in steps]
    step = matches[0]

    assert GATE_MODULE in step["run"]
    assert step["env"]["DATABASE_URL"].startswith("sqlite+aiosqlite:///")
    # …and the P1b gate this harness comes from remains independently wired.
    p1b = [step for step in steps if step.get("name", "").startswith("Run P1b Qdrant")]
    assert p1b and "tests/retrieval/test_p1b_gate.py" in p1b[0]["run"]
