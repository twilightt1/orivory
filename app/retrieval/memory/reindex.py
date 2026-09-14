"""Reindex / backfill personal memories into the vector store.

The Postgres ``memories`` table is the source of truth. The ChromaDB
``Orivory_memories`` collection is a derived index that can be lost
(restart with empty volume, corruption) or fall behind (memories written
before write-through embedding existed). This helper replays memories from
Postgres into ChromaDB so recall can always be made whole again.

Usage:
    reindex_user_memories_sync(str(user_id))                     # only missing
    reindex_user_memories_sync(str(user_id), only_missing=False)  # rebuild all
"""
from __future__ import annotations

import logging

from sqlalchemy import select

from app.database import sync_session
from app.models.memory import Memory
from app.retrieval.memory.visibility import current_memory_predicate

log = logging.getLogger(__name__)

_PAGE_SIZE = 200


def reindex_user_memories_sync(user_id: str, only_missing: bool = True) -> dict:
    """Embed a user's memories into ChromaDB in batches.

    Returns a summary dict: scanned, already_indexed, reindexed, pages.
    Raises on failure after logging (the admin caller reports ``queued=False``).

    Current rows only: superseded rows are history and dirty rows are stale —
    neither belongs in the vector index (the predicate applies before paging).

    ``only_missing`` is valid only for a collection whose fingerprint and
    contract-generation token match the active runtime. A mismatch aborts
    rather than mixing users into a partially rebuilt shared collection; a
    fresh active-generation pointer is a later migration phase.
    """
    from app.retrieval.embedder import EmbeddingDimensionMismatch
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
                        .where(Memory.user_id == user_id, current_memory_predicate())
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
                    try:
                        existing = get_existing_memory_ids_sync([str(m.id) for m in rows])
                    except EmbeddingDimensionMismatch as exc:
                        raise EmbeddingDimensionMismatch(
                            "only_missing reindex requires a matching fingerprint and "
                            "contract generation; rebuild the relevant data in a fresh "
                            "collection"
                        ) from exc
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
        raise
