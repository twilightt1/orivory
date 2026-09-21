from __future__ import annotations

import json
import logging

from app.redis_client import get_redis

log = logging.getLogger(__name__)
TTL = 7200


def _key(conversation_id: str, parent_id: str) -> str:
    return f"parent_chunk:{conversation_id}:{parent_id}"


async def store_parents(conversation_id: str, parents: list[dict]) -> None:
    redis = await get_redis()
    pipe  = redis.pipeline()
    for p in parents:
        pipe.setex(_key(conversation_id, p["id"]), TTL, json.dumps(p))
    await pipe.execute()
    log.info("Cached parents", extra={"conversation_id": conversation_id, "n": len(parents)})


async def get_parent(conversation_id: str, parent_id: str, db=None) -> dict | None:
    redis  = await get_redis()
    cached = await redis.get(_key(conversation_id, parent_id))

    if cached:
        await redis.expire(_key(conversation_id, parent_id), TTL)
        return json.loads(cached)


    if db is not None:
        return await _load_from_db(db, parent_id, conversation_id)

    return None


async def get_parents_batch(
    conversation_id: str,
    parent_ids: list[str],
    db=None,
) -> dict[str, dict]:
    if not parent_ids:
        return {}

    redis    = await get_redis()
    keys     = [_key(conversation_id, pid) for pid in parent_ids]
    values   = await redis.mget(keys)
    result:  dict[str, dict] = {}
    missing: list[str]       = []

    for pid, val in zip(parent_ids, values, strict=False):
        if val:
            result[pid] = json.loads(val)
        else:
            missing.append(pid)


    if result:
        pipe = redis.pipeline()
        for pid in result:
            pipe.expire(_key(conversation_id, pid), TTL)
        await pipe.execute()


    if missing and db is not None:
        db_parents = await _load_batch_from_db(db, missing, conversation_id)
        result.update(db_parents)


        if db_parents:
            pipe = redis.pipeline()
            for pid, p in db_parents.items():
                pipe.setex(_key(conversation_id, pid), TTL, json.dumps(p))
            await pipe.execute()

    return result


async def invalidate_conversation(conversation_id: str) -> None:
    redis  = await get_redis()
    cursor = 0
    while True:
        cursor, keys = await redis.scan(
            cursor, match=f"parent_chunk:{conversation_id}:*", count=100
        )
        if keys:
            await redis.delete(*keys)
        if cursor == 0:
            break




async def _load_from_db(db, parent_id: str, conversation_id: str) -> dict | None:
    from sqlalchemy import select

    from app.models.document import Document
    from app.models.document_chunk import DocumentChunk

    chunk = await db.scalar(
        select(DocumentChunk)
        .join(Document, DocumentChunk.document_id == Document.id)
        .where(
            DocumentChunk.id == parent_id,
            Document.conversation_id == conversation_id,
        )
    )
    if not chunk:
        return None
    return {
        "id":       str(chunk.id),
        "content":  chunk.content,
        "metadata": chunk.chunk_metadata,
    }


async def _load_batch_from_db(
    db,
    parent_ids: list[str],
    conversation_id: str,
) -> dict[str, dict]:
    from sqlalchemy import select

    from app.models.document import Document
    from app.models.document_chunk import DocumentChunk

    result = await db.execute(
        select(DocumentChunk)
        .join(Document, DocumentChunk.document_id == Document.id)
        .where(
            DocumentChunk.id.in_(parent_ids),
            Document.conversation_id == conversation_id,
        )
    )
    chunks = result.scalars().all()
    return {
        str(c.id): {
            "id":       str(c.id),
            "content":  c.content,
            "metadata": c.chunk_metadata,
        }
        for c in chunks
    }




def store_parents_sync(conversation_id: str, parents: list[dict]) -> None:
    # The parent cache lives in the process-local InMemoryRedis, which only the
    # async path can reach (`get_redis()` is a coroutine): `store_parents`
    # owns the write.
    return None
