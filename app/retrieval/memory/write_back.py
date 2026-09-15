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


async def safe_enqueue_graph_build(memory_id: UUID | str) -> None:
    """Build the knowledge graph for a memory, off the event loop.

    The builder is sync-by-design — it opens its own sync session and drives
    the extraction with ``asyncio.run`` — so it must NOT run on the event loop
    every caller here lives on: ``asyncio.run`` raises there and the best-effort
    handler below used to swallow the ``RuntimeError``, which is why the graph
    stayed empty for every loop-driven write path (P2/T9). Hand it to a worker
    thread (R1(p2): stdlib ``asyncio.to_thread``, never the ORT-sized embed
    executor).

    Never raises — the memory row is already committed; a failure is logged at
    ERROR, the level an operator watches, instead of disappearing.
    """
    try:
        await asyncio.to_thread(_build_memory_graph_sync, memory_id)
    except Exception:
        log.exception(
            "Memory graph build failed for %s: the memory row is committed and "
            "unaffected, its graph is not (best-effort)",
            memory_id,
            extra={"memory_id": str(memory_id)},
        )


async def index_new_memory(memory: Memory) -> bool:
    """Run the full post-persist indexing pipeline for one memory.

    Caller must have already committed the row. Embeds through the async
    embedder (off the loop, P2/T1) and enqueues graph extraction. Contract
    mismatches propagate as typed integrity failures. Use this from any async
    path that creates or updates a ``Memory``.

    Returns whether the vector write landed: ``False`` means the durable
    outbox intent enqueued with the row is now the only path to the index
    (``drain_pending``), which is what ``indexing="pending"`` reports.
    """
    indexed = await safe_upsert_to_index(memory)
    # The build is sync-by-design and drives its own ``asyncio.run``: the helper
    # offloads it to a worker thread (P2/T9), so the loop never runs it and a
    # failure is loud instead of silent.
    await safe_enqueue_graph_build(memory.id)
    return indexed


__all__ = [
    "safe_upsert_to_index",
    "safe_delete_from_index",
    "safe_enqueue_graph_build",
    "index_new_memory",
]
