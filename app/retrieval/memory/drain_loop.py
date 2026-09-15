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
"""
from __future__ import annotations

import asyncio

import structlog

from app.config import settings
from app.retrieval.memory.outbox import drain_pending

log = structlog.get_logger()

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


async def run_drain_loop(*, interval: float, batch_size: int, stop: asyncio.Event) -> None:
    """Drain the outbox until ``stop`` is set. Never raises out of itself.

    A batch that applied anything is followed immediately by the next one (a
    backlog drains at full speed); an idle batch then waits out ``interval`` or
    wakes for the stop flag, whichever comes first. Every round logs its counts;
    a drain-level error is a warning and the round after it runs as usual.
    """
    while not stop.is_set():
        try:
            report = await drain_once(batch_size=batch_size)
            log.info("outbox drain", **report)
            applied = report.get("applied", 0)
        except Exception as e:  # the loop must outlive any failure
            log.warning("outbox drain failed", error=str(e))
            applied = 0
        if applied:
            continue  # there may be more work right now: don't wait the interval
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
        await asyncio.wait_for(task, timeout=_STOP_TIMEOUT_SECONDS)
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
