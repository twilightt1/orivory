"""The barrier must neither be starved by a backlog nor wait for the impossible.

Two findings on ``app/retrieval/memory/freshness.py``:

- 563: the barrier counts its OWN tenant+kind, but the drain it awaits claims
  globally by seq — a foreign chunk backlog filled the first batch and the
  caller's landable write was never claimed.
- 95: a pending intent deliberately delayed by backoff (``next_attempt_at`` in
  the future) made every recall wait out the whole budget and then answer the
  typed 503 — for a write that could not land at all in that window.

Both keep the signed contract: pending stays non-fresh, recall still raises
``IndexFreshnessTimeout`` (the API's typed 503), never an empty 200.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from app import database
from app.config import settings
from app.models.index_outbox import IndexOutbox
from app.retrieval.memory import drain_loop, freshness, outbox
from app.retrieval.memory.outbox import (
    CHUNK_TARGET_GENERATION,
    KIND_CHUNK,
    IndexFreshnessTimeout,
)
from app.retrieval.vector_retriever import VectorUnavailableError
from tests.retrieval.test_drain_loop import _activate_memory_manifest, _statuses

# The drain-loop suite's ``env`` / ``owner`` fixtures (registered as a plugin,
# exactly as the signed barrier suite does).
pytest_plugins = ["tests.retrieval.test_drain_loop"]

CONTENT = "walked the chestnut orchard at dusk"
BATCH = 50
FOREIGN_CHUNKS = 150
CHUNK_APPLY_SECONDS = 0.02


@pytest.fixture(autouse=True)
def _barrier_reads_this_suites_own_outbox(env, monkeypatch):
    monkeypatch.setattr(freshness, "AsyncSessionLocal", database.AsyncSessionLocal)


async def _write_memory(sessions, user_id: uuid.UUID, content: str) -> uuid.UUID:
    """One committed memory + its pending upsert intent (the drain-loop shape)."""
    from app.models.memory import Memory

    async with sessions() as db:
        memory = Memory(id=uuid.uuid4(), user_id=user_id, content=content, tags=[])
        db.add(memory)
        outbox.bump_revision(memory)
        await outbox.enqueue_upsert(db, memory)
        await db.commit()
        return memory.id


async def _status_of(sessions, memory_id: uuid.UUID) -> str:
    async with sessions() as db:
        return await db.scalar(
            select(IndexOutbox.status).where(IndexOutbox.entity_id == memory_id.hex)
        )


# ── 563: a foreign backlog cannot starve the caller's own intent ────────────


async def test_a_foreign_chunk_backlog_cannot_starve_the_barrier(env, owner, monkeypatch):
    """One own memory intent BEHIND a foreign chunk backlog (the probe's shape).

    The drain claims by seq, so with 150 due foreign chunk intents ahead of the
    caller's row the barrier's own drain spent every batch on them — the caller
    waited out the whole 2.0 s budget and answered a typed 503 for a write that
    was landable the entire time. (The probe used 60 rows: two rounds at batch
    50 reach the caller's row, so the backlog is scaled here to outrun the
    budget deterministically.)
    """
    calls: list[int] = []

    async def _slow_chunk(db, row):
        calls.append(1)
        await asyncio.sleep(CHUNK_APPLY_SECONDS)  # one embedding call per chunk
        return "applied"

    monkeypatch.setattr(outbox, "_apply_chunk_intent", _slow_chunk)
    monkeypatch.setattr(settings, "OUTBOX_DRAIN_BATCH_SIZE", BATCH)
    await _activate_memory_manifest(env)

    async with env() as db:
        db.add_all([
            IndexOutbox(
                kind=KIND_CHUNK,
                entity_id=uuid.uuid4().hex,
                tenant_id=uuid.uuid4().hex,  # a stranger's bulk import
                revision=0,
                operation=outbox.OPERATION_DELETE,
                target_generation=CHUNK_TARGET_GENERATION,
                status="pending",
            )
            for _ in range(FOREIGN_CHUNKS)
        ])
        await db.commit()

    memory_id = await _write_memory(env, owner, CONTENT)  # the caller's own write

    claimed: list[str] = []
    real_apply = outbox._apply

    async def _record(db, row):
        claimed.append(row.entity_id)
        return await real_apply(db, row)

    monkeypatch.setattr(outbox, "_apply", _record)

    waited = await freshness.await_freshness(user_id=str(owner), timeout=2.0, poll=0.01)

    assert claimed and claimed[0] == memory_id.hex, (
        "the caller's own intent must be claimed ahead of a foreign backlog")
    assert await _status_of(env, memory_id) == "done", "the write landed inside the barrier"
    assert waited < 2.0
    assert await _statuses(), "the foreign backlog is still there — and not this caller's business"


# ── 95: a backoff-delayed write is a typed retry, not a burned budget ──────


async def test_a_backoff_delayed_intent_fails_fast_with_its_retry_after(env, owner, monkeypatch):
    """The store is down, so the intent backs off 60s+: nothing can land now."""
    await _activate_memory_manifest(env)
    await _write_memory(env, owner, CONTENT)

    async def _store_down(_memory):
        raise VectorUnavailableError("Qdrant refused")

    monkeypatch.setattr(outbox, "upsert_memory", _store_down)

    t0 = time.perf_counter()
    with pytest.raises(IndexFreshnessTimeout) as caught:
        await freshness.await_freshness(user_id=str(owner), timeout=2.0, poll=0.05)
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.0, "the barrier burned its whole budget for an unlandable write"
    retry_after = caught.value.retry_after
    assert retry_after is not None, "the typed state must name when a retry can land"
    assert retry_after > datetime.now(UTC)
    assert "retry" in str(caught.value).lower()
    assert await _statuses() == ["pending"], "fail-closed: nothing was acked, nothing ranked fresh"


async def test_a_still_in_flight_intent_keeps_waiting_the_budget(env, owner, monkeypatch):
    """The fail-closed path is unchanged: a CLAIMABLE intent waits, then 503s."""
    await _activate_memory_manifest(env)
    await _write_memory(env, owner, CONTENT)

    async def _boom(*, batch_size):
        raise RuntimeError("drain exploded")  # the intent is due, but nothing lands

    monkeypatch.setattr(drain_loop, "drain_once", _boom)

    t0 = time.perf_counter()
    with pytest.raises(IndexFreshnessTimeout) as caught:
        await freshness.await_freshness(user_id=str(owner), timeout=0.3, poll=0.02)
    elapsed = time.perf_counter() - t0

    assert elapsed >= 0.3, "a claimable intent is still waited on, not given up early"
    assert caught.value.retry_after is None
    assert await _statuses() == ["pending"]
