from __future__ import annotations

import logging
import re

import numpy as np
from rank_bm25 import BM25Okapi

log = logging.getLogger(__name__)

# In-process counter holding a monotonically increasing "generation" per
# conversation. Any mutation to a conversation's parent chunks (ingestion
# complete, document delete, conversation delete) bumps it, and each reader
# records the generation its in-memory index was built at and rebuilds lazily
# when it falls behind. The one-container stack runs a single process, so the
# store behind it is the process-local InMemoryRedis.
_GEN_KEY = "bm25_gen:{cid}"

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    words = _TOKEN_RE.findall(text.lower())
    bigrams = [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]
    return words + bigrams


class BM25Retriever:
    def __init__(self):
        # conversation_id -> (index, parent chunk dicts)
        self._indexes: dict[str, tuple[BM25Okapi, list[dict]]] = {}
        # conversation_id -> generation the local index was built at
        self._generations: dict[str, int] = {}

    # ── generation helpers ──────────────────────────────────────────────

    @staticmethod
    async def _read_generation_async(conversation_id: str) -> int | None:
        """Current generation from Redis, or None if Redis is unreachable."""
        try:
            from app.redis_client import get_redis

            redis = await get_redis()
            raw = await redis.get(_GEN_KEY.format(cid=conversation_id))
            return int(raw) if raw is not None else 0
        except Exception as exc:
            log.warning("BM25 generation read failed", extra={"conversation_id": conversation_id, "error": str(exc)})
            return None

    @staticmethod
    async def _bump_generation_async(conversation_id: str) -> None:
        try:
            from app.redis_client import get_redis

            redis = await get_redis()
            await redis.incr(_GEN_KEY.format(cid=conversation_id))
        except Exception as exc:
            log.warning("BM25 generation bump failed", extra={"conversation_id": conversation_id, "error": str(exc)})

    # ── lifecycle ───────────────────────────────────────────────────────

    def has_index(self, conversation_id: str) -> bool:
        return conversation_id in self._indexes

    async def ensure_async(self, db, conversation_id: str) -> dict[str, bool | str]:
        """Ensure this process has a fresh BM25 index for the conversation.

        Rebuilds from the database when there is no local index or when the
        local index is older than the shared Redis generation (i.e. another
        process changed the underlying chunks).
        """
        had_index = self.has_index(conversation_id)
        remote_gen = await self._read_generation_async(conversation_id)
        local_gen = self._generations.get(conversation_id)

        if remote_gen is None:
            # Redis unavailable — fall back to "rebuild only if missing".
            stale = False
        else:
            stale = local_gen is None or local_gen < remote_gen

        rebuilt = False
        if not had_index or stale:
            await self.rebuild_async(db, conversation_id)
            # Record the generation observed *before* the DB read so a write
            # racing the rebuild simply triggers another rebuild next time.
            self._generations[conversation_id] = remote_gen or 0
            rebuilt = True

        return {
            "had_index": had_index,
            "rebuilt": rebuilt,
            "stale": stale,
            "has_index": self.has_index(conversation_id),
        }

    def build_from_parents(self, conversation_id: str, parents: list[dict]) -> None:
        if not parents:
            self._indexes.pop(conversation_id, None)
            return
        tokenized = [_tokenize(p["content"]) for p in parents]
        self._indexes[conversation_id] = (BM25Okapi(tokenized), parents)
        log.info("BM25 built", extra={"conversation_id": conversation_id, "n": len(parents)})

    def rebuild_sync(self, db, conversation_id: str) -> None:
        from sqlalchemy import select

        from app.models.document import Document
        from app.models.document_chunk import DocumentChunk

        rows = db.execute(
            select(DocumentChunk)
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(
                Document.conversation_id == conversation_id,
                Document.status == "ready",
                DocumentChunk.chunk_metadata["chunk_type"].astext == "parent",
            )
            .order_by(DocumentChunk.created_at)
        ).scalars().all()

        if not rows:
            self._indexes.pop(conversation_id, None)
            return

        parents = [{"id": str(c.id), "content": c.content, "metadata": c.chunk_metadata} for c in rows]
        self.build_from_parents(conversation_id, parents)

    async def rebuild_async(self, db, conversation_id: str) -> None:
        from sqlalchemy import select

        from app.models.document import Document
        from app.models.document_chunk import DocumentChunk

        result = await db.execute(
            select(DocumentChunk)
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(
                Document.conversation_id == conversation_id,
                Document.status == "ready",
                DocumentChunk.chunk_metadata["chunk_type"].astext == "parent",
            )
            .order_by(DocumentChunk.created_at)
        )
        rows = result.scalars().all()

        if not rows:
            self._indexes.pop(conversation_id, None)
            return

        parents = [{"id": str(c.id), "content": c.content, "metadata": c.chunk_metadata} for c in rows]
        self.build_from_parents(conversation_id, parents)

    # ── publish: mutate + bump shared generation ────────────────────────

    def publish_build_sync(self, conversation_id: str, parents: list[dict]) -> None:
        """Build the local index. Called from the (synchronous) ingestion task
        after chunks are committed.

        Nothing to signal in the one-container stack: the index built here IS
        the one this process serves, and the in-process generation counter is
        only reachable from a coroutine.
        """
        self.build_from_parents(conversation_id, parents)

    async def publish_rebuild_async(self, db, conversation_id: str) -> None:
        """Rebuild the local index from the DB and signal other processes."""
        await self.rebuild_async(db, conversation_id)
        await self._bump_generation_async(conversation_id)

    async def publish_invalidate_async(self, conversation_id: str) -> None:
        """Drop the local index and signal other processes to do the same."""
        self.invalidate(conversation_id)
        await self._bump_generation_async(conversation_id)

    # ── search ──────────────────────────────────────────────────────────

    async def search(self, query: str, top_k: int, conversation_id: str) -> list[dict]:
        loaded = self._indexes.get(conversation_id)
        if not loaded:
            return []

        index, chunks = loaded
        # BM25 scoring + argsort are CPU-bound over the whole corpus; on a
        # large index they block the event loop for hundreds of ms per
        # request. Run them in a worker thread instead.
        def _score() -> tuple[list[int], list[float]]:
            query_tokens = _tokenize(query)
            scores = index.get_scores(query_tokens)
            top_n = list(np.argsort(scores)[::-1][:top_k])
            return top_n, [float(scores[i]) for i in top_n]

        import asyncio

        top_n, top_scores = await asyncio.to_thread(_score)

        return [
            {
                "content":   chunks[i]["content"],
                "score":     score,
                "source":    "bm25",
                "rank":      rank,
                "metadata":  chunks[i]["metadata"],
                "parent_id": chunks[i]["id"],
            }
            for rank, (i, score) in enumerate(zip(top_n, top_scores, strict=True))
            if score > 0
        ]

    def invalidate(self, conversation_id: str) -> None:
        self._indexes.pop(conversation_id, None)
        self._generations.pop(conversation_id, None)


bm25_retriever = BM25Retriever()
