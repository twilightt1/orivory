"""Single owner for the "memory persisted → index it" side effects.

When a ``Memory`` row is created or updated (manual API write, connector sync,
or a future save-note node), two follow-up actions must happen so the memory
is actually recallable:

    1. Embed it into the ChromaDB ``Orivory_memories`` collection.
    2. Enqueue knowledge-graph extraction (entities + relations).

Both are **best-effort**: the Postgres ``Memory`` row is the source of truth,
and a failure here must never fail the request that created the memory. If an
embed is dropped, the reindex task (``app.tasks.reindex_tasks``) can replay it
from Postgres later.

Centralizing these here removes the duplicate ``_safe_*`` helpers that
previously lived in both ``api/v1/memories.py`` and ``ingestion/dispatcher.py``
and could drift out of sync.
"""
from __future__ import annotations

import logging
from uuid import UUID

from app.config import settings
from app.models.memory import Memory

log = logging.getLogger(__name__)


async def safe_upsert_to_chroma(memory: Memory) -> bool:
    """Embed a memory into ChromaDB. Returns True on success, never raises."""
    try:
        from app.retrieval.memory.vector_store import upsert_memory

        await upsert_memory(memory)
        return True
    except Exception as exc:
        log.warning(
            "ChromaDB upsert failed for memory %s: %s",
            memory.id, exc,
            extra={"memory_id": str(memory.id), "user_id": str(memory.user_id)},
        )
        return False


async def safe_delete_from_chroma(memory_id: UUID | str) -> None:
    """Remove a memory's vector from ChromaDB. Never raises."""
    try:
        from app.retrieval.memory.vector_store import delete_memory

        await delete_memory(str(memory_id))
    except Exception as exc:
        log.warning(
            "ChromaDB delete failed for memory %s: %s",
            memory_id, exc,
            extra={"memory_id": str(memory_id)},
        )


def safe_enqueue_graph_build(memory_id: UUID | str) -> None:
    """Enqueue knowledge-graph extraction for a memory. Never raises.

    Eager (lite/CELERY_TASK_ALWAYS_EAGER): call the builder synchronously
    in-process — no broker, no worker round-trip. Otherwise enqueue via the
    Celery task as before.
    """
    try:
        from app.graph.builder import build_memory_graph_sync
        from app.tasks.db import sync_session

        if settings.CELERY_TASK_ALWAYS_EAGER:
            with sync_session() as db:
                build_memory_graph_sync(db, str(memory_id))
            return

        from app.tasks.graph_tasks import build_memory_graph_task

        build_memory_graph_task.delay(str(memory_id))
    except Exception as exc:
        log.warning(
            "Graph build enqueue failed for memory %s: %s",
            memory_id, exc,
            extra={"memory_id": str(memory_id)},
        )


async def index_new_memory(memory: Memory) -> None:
    """Run the full post-persist indexing pipeline for one memory.

    Caller must have already committed the row. Embeds synchronously (best
    effort) and enqueues graph extraction. Use this from any async path that
    creates or updates a ``Memory``.
    """
    await safe_upsert_to_chroma(memory)
    safe_enqueue_graph_build(memory.id)


__all__ = [
    "safe_upsert_to_chroma",
    "safe_delete_from_chroma",
    "safe_enqueue_graph_build",
    "index_new_memory",
]
