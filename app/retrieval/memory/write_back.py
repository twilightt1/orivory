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

    Returns whether the backend confirmed the delete, so a caller that owns a
    durable intent (the outbox drain) can tell a purge from an outage.
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


def safe_enqueue_graph_build(memory_id: UUID | str) -> None:
    """Build the knowledge graph for a memory, synchronously in-process.

    Never raises — failures are logged; the memory row is already committed.
    """
    try:
        from app.database import sync_session
        from app.graph.builder import build_memory_graph_sync

        with sync_session() as db:
            build_memory_graph_sync(db, str(memory_id))
    except Exception as exc:
        log.warning(
            "Graph build enqueue failed for memory %s: %s",
            memory_id, exc,
            extra={"memory_id": str(memory_id)},
        )


async def index_new_memory(memory: Memory) -> bool:
    """Run the full post-persist indexing pipeline for one memory.

    Caller must have already committed the row. Embeds synchronously (best
    effort for transient outages) and enqueues graph extraction. Contract
    mismatches propagate as typed integrity failures. Use this from any async
    path that creates or updates a ``Memory``.

    Returns whether the vector write landed: ``False`` means the durable
    outbox intent enqueued with the row is now the only path to the index
    (``drain_pending``), which is what ``indexing="pending"`` reports.
    """
    indexed = await safe_upsert_to_index(memory)
    safe_enqueue_graph_build(memory.id)
    return indexed


__all__ = [
    "safe_upsert_to_index",
    "safe_delete_from_index",
    "safe_enqueue_graph_build",
    "index_new_memory",
]
