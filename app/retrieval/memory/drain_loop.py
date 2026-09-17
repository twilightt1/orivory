"""P3 — the background outbox drain loop (SQLite and Postgres alike).

The P1a outbox is durable: an intent that could not be indexed in its request
stays ``pending`` with backoff. Before P3 the only drainer was a bounded boot
hook in ``app/main.py`` — so a Postgres deployment never replayed a backlog at
all, and a SQLite one did it only at start. This module owns the drain for both:
one task for the life of the app, started and stopped by the lifespan.

Single-flight (ruling R1): every claim goes through :func:`drain_once`, which
holds the module-level lock, so exactly one claimer exists inside the process —
the loop, and whatever else calls it (the freshness barrier reads through the
same door rather than starting a second one).

The loop also carries the post-drain hooks: the erasure-receipt reconcile
(R15), the consolidation producer (R39, budgeted — see
:func:`_consolidate_after_drain`), and the opt-in retention sweep (P4b/T6,
idle-only — see :func:`_retain_after_drain`).
"""
from __future__ import annotations

import asyncio

import structlog

from app.config import settings
from app.observability.fallbacks import count_fallback
from app.retrieval.memory.consolidation import run_consolidation, users_with_servable_memories
from app.retrieval.memory.outbox import drain_pending
from app.services.erasure_service import reconcile_erasure_receipts
from app.services.retention_service import run_retention

log = structlog.get_logger()

# The fallback every failed drain round counts (ruling R24). The intents are
# untouched — still pending, retried with backoff — but a rising rate means the
# retry path itself is bleeding (a store down for a week looks identical to
# every write landing when only the fast path is watched).
DRAIN_FAILED_FALLBACK = "index.outbox_drain_failed"

# How long a stop may take before the task is cancelled outright. A drain batch
# commits per row, so cancelling mid-batch loses nothing but the current row's
# next_attempt_at.
_STOP_TIMEOUT_SECONDS = 5.0

# The single-flight lock: one drain claimer per process (ruling R1).
_lock = asyncio.Lock()
_lock_owner: asyncio.AbstractEventLoop | None = None

# Set by start_drain_loop, cleared by stop_drain_loop.
_stop: asyncio.Event | None = None


def _should_drain() -> bool:
    """The one gate on draining — both dialects drain; only the setting stops it.

    P3 removed the ``DATABASE_URL.startswith("sqlite")`` branch the old boot hook
    had: a Postgres deployment replays its backlog through the very same loop.
    """
    return settings.OUTBOX_DRAIN_ENABLED


def _single_flight() -> asyncio.Lock:
    """The drain lock, rebound when the running event loop changed.

    An ``asyncio.Lock`` binds to the first loop that ever contends it, and a
    loop binding outlives the loop: a second loop (every test, a re-entered
    ``asyncio.run``) would then fail its next contended drain with "bound to a
    different event loop". A process has one loop, so this never fires there.
    """
    global _lock, _lock_owner
    loop = asyncio.get_running_loop()
    if _lock_owner is not loop:
        _lock, _lock_owner = asyncio.Lock(), loop
    return _lock


async def drain_once(*, batch_size: int) -> dict:
    """Claim and apply ONE batch, serialized against every other claimer.

    The single entry point for draining: the loop below, and the P3 freshness
    barrier, both come through here so a claim never races another claim.
    """
    async with _single_flight():
        return await drain_pending(batch_size=batch_size)


async def _reconcile_after_drain() -> None:
    """Re-verify open erasure receipts after a round that landed work (R15/R16).

    Opportunistic and bounded (50, open receipts only): the request-path
    freshness barrier drains through :func:`drain_once` and must not pay for
    this scan, so it lives on the loop. Never fatal — a reconcile failure must
    not stop the drain.
    """
    try:
        report = await reconcile_erasure_receipts()
        if report["checked"]:
            log.info("erasure receipts reconciled", **report)
    except Exception as e:  # the loop must outlive any failure
        log.warning("erasure receipt reconcile failed", error=str(e))


async def _consolidate_after_drain() -> None:
    """Spend a drain round's spare time on the consolidation producer (R39).

    Runs after a round that landed work AND on an idle tick — a landed batch is
    where new evidence arrives, and an idle tick is free time either way. Opens
    its own session (the erase-reconcile shape: the request-path barrier drains
    through :func:`drain_once` and must not pay for this) and never fatal: a
    producer failure must not stop the drain.

    Budget: ``CONSOLIDATION_BUDGET_PER_RUN`` per user per pass (soft — a run
    publishes at most that many summaries).

    ponytail: one full pass per round, per-tagged-user scan; the producer's
    dedupe key makes a repeat pass cheap (no LLM work), so gate this to idle
    ticks only if a measured cost says to.
    """
    try:
        # Late import: the test fixtures' sessionmaker lives on the module.
        from app.database import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            for user_id in await users_with_servable_memories(db):
                report = await run_consolidation(
                    db, user_id, budget=settings.CONSOLIDATION_BUDGET_PER_RUN)
                if (report.published or report.refused or report.errors
                        or report.truncated):
                    log.info("consolidation pass", user_id=str(user_id),
                             **report._asdict())
    except Exception as e:  # the loop must outlive any failure
        log.warning("consolidation pass failed", error=str(e))


async def _retain_after_drain() -> None:
    """Spend an idle tick on the opt-in retention sweep (P4b/T6, spec §8.1).

    IDLE ONLY — unlike the producer there is nothing a landed batch changes
    about retention: it is a clock (``indexed_at`` vs the user's own window),
    not evidence. Opens its own session (the same shape as the erase-reconcile
    and the producer: the request-path barrier drains through
    :func:`drain_once` and must not pay for this) and never fatal.

    Cost: the sweep is opt-in, so with no user enabled the whole pass is ONE
    SELECT on ``users`` and nothing else (no memory scan at all) — every
    pre-P4b user reads OFF via the ladder's column default.
    """
    try:
        # Late import: the test fixtures' sessionmaker lives on the module.
        from app.database import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            report = await run_retention(db)
            if report.invalidated:
                log.info("retention pass", **report._asdict())
    except Exception as e:  # the loop must outlive any failure
        log.warning("retention pass failed", error=str(e))


async def run_drain_loop(*, interval: float, batch_size: int, stop: asyncio.Event) -> None:
    """Drain the outbox until ``stop`` is set. Never raises out of itself.

    A batch that applied anything is followed immediately by the next one (a
    backlog drains at full speed) and by the erasure-receipt reconcile pass; an
    idle batch then waits out ``interval`` or wakes for the stop flag, whichever
    comes first. Both shapes also run the consolidation producer (R39): after a
    landed round because that is where new evidence arrived, on an idle tick
    because that is free time. Every round logs its counts; a drain-level error
    is a warning and the round after it runs as usual.
    """
    while not stop.is_set():
        try:
            report = await drain_once(batch_size=batch_size)
            log.info("outbox drain", **report)
            applied = report.get("applied", 0)
        except Exception as e:  # the loop must outlive any failure
            count_fallback(DRAIN_FAILED_FALLBACK)
            log.warning("outbox drain failed", error=str(e))
            applied = 0
        if applied:
            # A landed batch is where a receipt's owed deletes get satisfied.
            await _reconcile_after_drain()
            # ... and where the producer picks the new evidence up (R39).
            await _consolidate_after_drain()
            continue  # there may be more work right now: don't wait the interval
        # An idle tick is free time for the producer too (R39).
        await _consolidate_after_drain()
        # ... and for the retention sweep, which rides idle ticks only (T6):
        # it is a clock, not evidence — a landed batch changes nothing for it.
        await _retain_after_drain()
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except TimeoutError:
            pass  # an idle round: wait again


async def start_drain_loop() -> asyncio.Task | None:
    """Start the loop from the settings (``None`` when draining is disabled)."""
    if not _should_drain():
        log.info("Outbox drain loop disabled")
        return None
    global _stop
    _stop = asyncio.Event()
    return asyncio.create_task(
        run_drain_loop(
            interval=settings.OUTBOX_DRAIN_INTERVAL_SECONDS,
            batch_size=settings.OUTBOX_DRAIN_BATCH_SIZE,
            stop=_stop,
        ),
        name="outbox-drain",
    )


async def stop_drain_loop(task: asyncio.Task | None) -> None:
    """Stop the loop: set the flag, wait up to 5s, cancel a task that overruns."""
    global _stop
    if task is None:
        return
    if _stop is not None:
        _stop.set()
    try:
        # An already-cancelled task re-raises CancelledError on await — and
        # CancelledError is not an Exception, so unguarded it escapes the
        # lifespan's ``finally`` and skips ``close_clients()`` (the local-mode
        # folder-lock release this loop exists to protect).
        if not task.cancelled():
            await asyncio.wait_for(task, timeout=_STOP_TIMEOUT_SECONDS)
    except asyncio.CancelledError:
        pass  # cancelled out from under us (or cancelled mid-wait): stopped is stopped
    except TimeoutError:
        log.warning("Outbox drain loop did not stop in time; cancelling")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    except Exception as e:  # a broken loop must not take the shutdown down with it
        log.warning("Outbox drain loop ended with an error", error=str(e))
    finally:
        _stop = None
