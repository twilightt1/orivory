"""P3 — the recall freshness barrier: wait for YOUR OWN pending writes, bounded.

A memory write commits its SQL row and its durable index intent in the SAME
transaction (spec §5.1); the vector lands after that commit, from the
write-through fast path or — when that failed — from the background drain
(``app.retrieval.memory.drain_loop``, one interval away). A recall arriving in
that window used to search the index before the write and answer ``results:
[]``: a false no-match, indistinguishable from a real empty result.

:func:`await_freshness` gives up that read path its freshness: if the caller's
tenant has pending intents, it waits — bounded by
``settings.RECALL_FRESHNESS_BUDGET_SECONDS`` — for them to land, and raises
:class:`~app.retrieval.memory.outbox.IndexFreshnessTimeout` when they cannot.
The typed error is never served as an empty result: it propagates out of
``MemoryRetriever.recall`` to the API's 503 handler (ruling R10).

Ruling R1: this module never claims the outbox itself. It drains through
:func:`app.retrieval.memory.drain_loop.drain_once`, the same single-flight door
the background loop uses, so the process still has exactly one claimer.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid

from sqlalchemy import func, select

from app.config import settings
from app.database import AsyncSessionLocal
from app.models.index_outbox import IndexOutbox
from app.retrieval.memory import drain_loop
from app.retrieval.memory.outbox import IndexFreshnessTimeout

log = logging.getLogger(__name__)

__all__ = ["IndexFreshnessTimeout", "await_freshness"]


async def _pending_count(tenant: str) -> int:
    """This tenant's pending index intents, in a FRESH transaction.

    Never the caller's session: the drain commits through its own, and a
    long-lived read transaction (SQLite especially) would keep answering from
    the snapshot taken before those commits.
    """
    async with AsyncSessionLocal() as db:
        return int(
            (
                await db.execute(
                    select(func.count())
                    .select_from(IndexOutbox)
                    .where(
                        IndexOutbox.status == "pending",
                        IndexOutbox.tenant_id == tenant,
                    )
                )
            ).scalar_one()
        )


async def await_freshness(*, user_id: str, timeout: float, poll: float = 0.05) -> float:
    """Wait — bounded — for this user's pending index intents to land.

    Returns the seconds waited, measured: with nothing pending that is one
    count query's cost (no drain, no sleep), never a fabricated 0.

    Raises :class:`IndexFreshnessTimeout` when the intents are still pending at
    the end of ``timeout`` — the caller must not answer "no matches" for a
    write that is merely still in flight. A drain that fails leaves the intent
    pending with backoff (that is the retry contract), so an outage reaches
    this deadline rather than returning early.

    A queue that cannot even be read is logged and does NOT block the read:
    the barrier protects freshness, it is not a new hard dependency of recall
    on the outbox, and the read path's typed signals stay
    :class:`~app.retrieval.embedder.EmbeddingDimensionMismatch` and
    :class:`~app.retrieval.vector_retriever.VectorUnavailableError`.
    """
    t0 = time.perf_counter()
    try:
        if not settings.OUTBOX_DRAIN_ENABLED:
            # A quiesced deployment (a cutover owns the store) keeps intents
            # pending ON PURPOSE: waiting for them could only end in a
            # spurious 503.
            return time.perf_counter() - t0
        tenant = uuid.UUID(str(user_id)).hex  # the outbox's tenant column
        while True:
            if not await _pending_count(tenant):
                return time.perf_counter() - t0
            waited = time.perf_counter() - t0
            if waited >= timeout:
                raise IndexFreshnessTimeout(
                    f"index intents for tenant {tenant} were still pending after "
                    f"{waited:.2f}s (budget {timeout}s)"
                )
            await drain_loop.drain_once(batch_size=settings.OUTBOX_DRAIN_BATCH_SIZE)
            await asyncio.sleep(poll)
    except IndexFreshnessTimeout:
        raise
    except Exception as e:
        log.warning("Freshness barrier could not read the outbox: %s", e)
    return time.perf_counter() - t0
