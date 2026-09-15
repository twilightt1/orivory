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

Ruling R14: the barrier FAILS CLOSED. A pending-count read that raises, a drain
that raises, and a wait that ends with the count still > 0 all end in the same
typed timeout — no 200 with empty results for a write that may be unindexed.
A transient error is not fatal to the *wait*: it is logged and retried until
``RECALL_FRESHNESS_BUDGET_SECONDS`` is spent. The wall clock from barrier entry
is the budget's, drain included: the drain is awaited through
:func:`asyncio.wait_for` on the remaining budget, so a hung store (local Qdrant
mode has no client timeout) cannot outlive the deadline.
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
from app.retrieval.memory.outbox import KIND_MEMORY, IndexFreshnessTimeout

log = logging.getLogger(__name__)

__all__ = ["IndexFreshnessTimeout", "await_freshness"]


async def _pending_count(tenant: str) -> int:
    """This tenant's pending MEMORY index intents, in a FRESH transaction.

    ``kind == KIND_MEMORY``: the count answers "is the MEMORY index fresh for
    this tenant?" — only this module's caller reads memory vectors. A tenant's
    untended chunk backlog (a bulk import) is not this read path's business and
    must not 503 every memory recall for the whole budget.

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
                        IndexOutbox.kind == KIND_MEMORY,
                    )
                )
            ).scalar_one()
        )


async def await_freshness(*, user_id: str, timeout: float, poll: float = 0.05) -> float:
    """Wait — bounded — for this user's pending index intents to land.

    Returns the seconds waited, measured: with nothing pending that is one
    count query's cost (no drain, no sleep), never a fabricated 0.

    Raises :class:`IndexFreshnessTimeout` when, at the end of ``timeout``, the
    intents are still pending — or when they could not be PROVEN landed, i.e.
    the pending-count read or the drain kept failing. Fail closed (R14): a
    recall must never answer "no matches" for a write that may be unindexed. A
    transient error is a warning and the loop retries it until the deadline; it
    is never a licence to return early.
    """
    t0 = time.perf_counter()
    if not settings.OUTBOX_DRAIN_ENABLED:
        # A quiesced deployment (a cutover owns the store) keeps intents
        # pending ON PURPOSE: waiting for them could only end in a
        # spurious 503.
        return time.perf_counter() - t0
    tenant = uuid.UUID(str(user_id)).hex  # the outbox's tenant column
    deadline = t0 + timeout
    while True:
        try:
            if not await _pending_count(tenant):
                return time.perf_counter() - t0
        except Exception as e:
            # Nothing proven landed: keep waiting (R14), don't read the index.
            log.warning("Freshness barrier could not read the outbox: %s", e)
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        try:
            # The drain answers to the SAME deadline: a hung store must not
            # hang a recall (local Qdrant mode has no client timeout).
            await asyncio.wait_for(
                drain_loop.drain_once(batch_size=settings.OUTBOX_DRAIN_BATCH_SIZE),
                timeout=remaining,
            )
        except TimeoutError:
            if time.perf_counter() >= deadline:
                break  # the drain outlived the budget: the wait is over
            # A drain-internal TimeoutError is a transient failure, not this
            # barrier's deadline: retry it below.
            log.warning("Freshness barrier drain timed out early")
        except Exception as e:
            log.warning("Freshness barrier drain failed: %s", e)
        await asyncio.sleep(min(poll, max(deadline - time.perf_counter(), 0.0)))
    raise IndexFreshnessTimeout(
        f"index intents for tenant {tenant} were still pending after "
        f"{time.perf_counter() - t0:.2f}s (budget {timeout}s)"
    )
