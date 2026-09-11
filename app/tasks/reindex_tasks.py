"""Reindex / backfill personal memories into the vector store.

The Postgres ``memories`` table is the source of truth. The ChromaDB
``Orivory_memories`` collection is a derived index that can be lost
(restart with empty volume, corruption) or fall behind (memories written
before write-through embedding existed). This task replays memories from
Postgres into ChromaDB so recall can always be made whole again.

Usage:
    reindex_user_memories.delay(str(user_id))                 # only missing
    reindex_user_memories.delay(str(user_id), only_missing=False)  # rebuild all
"""
from __future__ import annotations

import logging

from sqlalchemy import select

from app.models.memory import Memory
from app.tasks.celery_app import celery_app
from app.tasks.db import sync_session

log = logging.getLogger(__name__)

_PAGE_SIZE = 200


@celery_app.task(
    bind=True,
    name="tasks.reindex_user_memories",
    max_retries=2,
    default_retry_delay=60,
    retry_backoff=True,
    retry_jitter=True,
    queue="ingestion",
    time_limit=1800,
    soft_time_limit=1680,
)
def reindex_user_memories(self, user_id: str, only_missing: bool = True) -> dict:
    """Embed a user's memories into ChromaDB in batches.

    Returns a summary dict: scanned, already_indexed, reindexed, pages.
    """
    from app.retrieval.memory.vector_store import (
        get_existing_memory_ids_sync,
        upsert_memories_sync,
    )

    scanned = 0
    already_indexed = 0
    reindexed = 0
    pages = 0
    offset = 0

    try:
        with sync_session() as db:
            while True:
                rows = (
                    db.execute(
                        select(Memory)
                        .where(Memory.user_id == user_id)
                        .order_by(Memory.indexed_at)
                        .offset(offset)
                        .limit(_PAGE_SIZE)
                    )
                    .scalars()
                    .all()
                )
                if not rows:
                    break

                pages += 1
                scanned += len(rows)

                to_index = rows
                if only_missing:
                    existing = get_existing_memory_ids_sync([str(m.id) for m in rows])
                    already_indexed += len(existing)
                    to_index = [m for m in rows if str(m.id) not in existing]

                if to_index:
                    reindexed += upsert_memories_sync(to_index)

                offset += _PAGE_SIZE

        summary = {
            "user_id": user_id,
            "only_missing": only_missing,
            "scanned": scanned,
            "already_indexed": already_indexed,
            "reindexed": reindexed,
            "pages": pages,
        }
        log.info("Memory reindex complete", extra=summary)
        return summary
    except Exception as exc:
        log.warning(
            "Memory reindex failed",
            extra={"user_id": user_id, "scanned": scanned, "error": str(exc)},
        )
        raise self.retry(exc=exc) from exc
