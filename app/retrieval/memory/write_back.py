"""Single owner for the "memory persisted → index it" side effects.

When a ``Memory`` row is created or updated (manual API write, connector sync,
or a future save-note node), two follow-up actions must happen so the memory
is actually recallable:

    1. Embed it into the Qdrant memory generation.
    2. Enqueue knowledge-graph extraction (entities + relations).

Both are **best-effort**: the Postgres ``Memory`` row is the source of truth,
and a failure here must never fail the request that created the memory. If an
embed is dropped, the reindex helper (``app.retrieval.memory.reindex``) can
replay it from Postgres later.

Centralizing these here removes the duplicate ``_safe_*`` helpers that
previously lived in both ``api/v1/memories.py`` and ``ingestion/dispatcher.py``
and could drift out of sync.
"""
from __future__ import annotations

import asyncio
import functools
import logging
from uuid import UUID

from app.models.memory import Memory
from app.retrieval.embedder import EmbeddingDimensionMismatch

log = logging.getLogger(__name__)


async def safe_upsert_to_index(memory: Memory) -> bool:
    """Embed a memory into the Qdrant memory generation.

    Transient vector outages return ``False`` for compatibility. Contract
    mismatches are typed integrity failures and are deliberately propagated.
    """
    try:
        from app.retrieval.memory.vector_store import upsert_memory

        await upsert_memory(memory)
        return True
    except EmbeddingDimensionMismatch:
        # A contract mismatch is not a transient best-effort outage: letting
        # it become ``False`` hides an unsafe collection from its caller.
        raise
    except Exception as exc:
        log.warning(
            "Vector upsert failed for memory %s: %s",
            memory.id, exc,
            extra={"memory_id": str(memory.id), "user_id": str(memory.user_id)},
        )
        return False


async def safe_delete_from_index(memory_id: UUID | str) -> bool:
    """Remove a memory's vector from the index. Never raises.

    ``True`` is the only proof the absence happened (R17). ``False`` covers a
    purge that did not land, a point that survived the delete, and a store that
    could not be read back alike — so a caller that owns a durable intent (the
    outbox drain) retries on ``False`` instead of reading it as a purge.
    """
    try:
        from app.retrieval.memory.vector_store import delete_memory

        return await delete_memory(str(memory_id))
    except Exception as exc:
        log.warning(
            "Vector delete failed for memory %s: %s",
            memory_id, exc,
            extra={"memory_id": str(memory_id)},
        )
        return False


def _build_memory_graph_sync(memory_id: UUID | str) -> None:
    """The worker-thread body: the sync builder over its own sync session."""
    from app.database import sync_session
    from app.graph.builder import build_memory_graph_sync

    with sync_session() as db:
        build_memory_graph_sync(db, str(memory_id))


# Strong references to the builds scheduled on a running loop. The loop keeps
# only weak references to tasks, so without this set a still-running build can
# be garbage-collected mid-flight; the done callback removes it and is also the
# one place a BACKGROUND failure is reported (an inline failure logs at its
# call site).
_pending_graph_builds: set[asyncio.Task] = set()

_GRAPH_BUILD_FAILED = (
    "Memory graph build failed for %s: the memory row is committed and "
    "unaffected, its graph is not (best-effort)"
)


def _log_graph_build_failure(memory_id: UUID | str, failure: BaseException) -> None:
    log.error(_GRAPH_BUILD_FAILED, memory_id, exc_info=failure,
              extra={"memory_id": str(memory_id)})


def _graph_build_done(task: asyncio.Task, memory_id: UUID | str) -> None:
    """Drops the strong ref; reports a failed build loudly, never raises."""
    _pending_graph_builds.discard(task)
    if task.cancelled():
        return
    failure = task.exception()
    if failure is not None:
        _log_graph_build_failure(memory_id, failure)


def safe_enqueue_graph_build(memory_id: UUID | str) -> None:
    """Schedule the knowledge-graph build for a memory; never blocks the write.

    The builder is sync-by-design — it opens its own sync session and drives
    the extraction with ``asyncio.run`` — so it must NOT run on the event loop
    every caller here lives on: ``asyncio.run`` raises there and the
    best-effort handler used to swallow the ``RuntimeError``, which is why the
    graph stayed empty for every loop-driven write path (P2/T9). R27(p2): the
    write path must not WAIT for it either — extraction is an LLM call, and
    its latency belongs to the graph, not to the request that saved the row.

    * a running loop (API / MCP / connector sync): the build is scheduled as a
      background task running in a worker thread (stdlib
      ``asyncio.to_thread``, never the ORT-sized embed executor) and this
      function returns immediately. The task is held in
      ``_pending_graph_builds`` and its done callback reports failures at
      ERROR with the traceback — a build cut off by process exit is an
      ACCEPTED best-effort loss: the memory row is committed and the graph can
      be rebuilt;
    * no running loop (CLI / script / worker thread): there is nothing to
      schedule on, so the build runs HERE, synchronously — the shape every
      non-loop caller has always had.

    This is a plain function on purpose: a coroutine-free signature means a
    caller cannot forget the ``await`` and silently drop the build.

    Cost note (accepted): each build drives its own ``asyncio.run`` loop, so
    ``get_llm_client()`` sees a different running loop every time and abandons
    + rebuilds the shared HTTP client for it (``app.agents.llm_client`` — see
    the abandonment comment there). At write frequency that is one discarded
    client per written memory; the extraction itself still passes through the
    shared ``LLM_MAX_CONCURRENCY`` gate.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        try:
            _build_memory_graph_sync(memory_id)
        except Exception as exc:
            _log_graph_build_failure(memory_id, exc)
        return

    task = asyncio.create_task(asyncio.to_thread(_build_memory_graph_sync, memory_id))
    _pending_graph_builds.add(task)
    task.add_done_callback(functools.partial(_graph_build_done, memory_id=memory_id))


async def index_new_memory(memory: Memory) -> bool:
    """Run the full post-persist indexing pipeline for one memory.

    Caller must have already committed the row. Embeds through the async
    embedder (off the loop, P2/T1) and SCHEDULES graph extraction — the write
    returns without waiting for the build (R27(p2)): extraction is an LLM call
    and its latency must not ride the write path. Contract mismatches
    propagate as typed integrity failures. Use this from any async path that
    creates or updates a ``Memory``.

    Returns whether the vector write landed: ``False`` means the durable
    outbox intent enqueued with the row is now the only path to the index
    (``drain_pending``), which is what ``indexing="pending"`` reports.
    """
    indexed = await safe_upsert_to_index(memory)
    # Scheduled, never awaited (R27(p2)): on a running loop the helper hands the
    # build to a background task and reports a failure loudly from its done
    # callback; with no loop it runs the build inline.
    safe_enqueue_graph_build(memory.id)
    return indexed


__all__ = [
    "safe_upsert_to_index",
    "safe_delete_from_index",
    "safe_enqueue_graph_build",
    "index_new_memory",
]
