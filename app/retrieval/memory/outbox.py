"""Durable index intent: enqueue in the same commit, drain after it (spec §5.1-§5.2).

Every canonical (SQL) mutation of an indexed entity stamps a monotonically
increasing ``revision`` on the row and inserts one ``index_outbox`` intent in
the SAME transaction — so a crash between the SQL commit and the vector write
is recoverable: the intent survives and :func:`drain_pending` replays it
against the LATEST SQL state. The existing post-commit write-through stays as
it was (best-effort, fast path); the outbox is the durable backstop, not a
replacement.

P1a keeps Chroma (:mod:`app.retrieval.memory.vector_store`) as the backend.
A drain applies one intent at a time: a dead row collapses to a delete, a
stale revision is skipped (a newer intent owns the entity), a contract
mismatch blocks terminally, a transient failure retries with backoff.
"""
from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.database import AsyncSessionLocal
from app.models.index_outbox import IndexGeneration, IndexOutbox
from app.models.memory import Memory
from app.retrieval.embedder import EmbeddingDimensionMismatch
from app.retrieval.memory.vector_store import COLLECTION_NAME, delete_memory, upsert_memory

log = logging.getLogger(__name__)

KIND_MEMORY = "memory"
OPERATION_UPSERT = "upsert"
OPERATION_DELETE = "delete"

# Transitional fallback when ``index_generations`` has no active row. It must
# keep the spelling the ladder seeds (controller ruling M3):
# ``vector_store.COLLECTION_NAME`` — a different spelling would break the
# unique intent key and enqueue the same logical write twice.
TARGET_GENERATION = COLLECTION_NAME

_BACKOFF_BASE_SECONDS = 60
_BACKOFF_CAP_SECONDS = 3600
_ERROR_TEXT_LIMIT = 500

# The active generation is stable for the life of one session/transaction
# (P1b swaps it between processes), so the manifest is read at most once per
# session — a batch import must not pay one read per row.
_GENERATION_CACHE_KEY = "orivory_outbox_generation"


def bump_revision(memory) -> int:
    """Advance (and return) the memory's monotonic write counter.

    Callers MUST enqueue the bumped revision in the same transaction —
    a bump without an intent is a lost index update by construction.
    """
    memory.revision = int(getattr(memory, "revision", 0) or 0) + 1
    return memory.revision


def _entity_id(value) -> str:
    """UUID as the 32-char hex stored in ``String(32)`` columns (PG-safe width)."""
    return uuid.UUID(str(value)).hex


def _upsert_values(memory, target_generation: str) -> dict[str, Any]:
    if getattr(memory, "id", None) is None:
        # The row's id is a Python-side default, applied at flush; the intent
        # must carry the same id, so materialize it early (same generator).
        memory.id = uuid.uuid4()
    return {
        "kind": KIND_MEMORY,
        "entity_id": _entity_id(memory.id),
        "tenant_id": _entity_id(memory.user_id),
        "revision": int(getattr(memory, "revision", 1) or 1),
        "operation": OPERATION_UPSERT,
        "target_generation": target_generation,
    }


def _delete_values(*, entity_id: str, tenant_id: str, revision: int, target_generation: str) -> dict[str, Any]:
    return {
        "kind": KIND_MEMORY,
        "entity_id": entity_id,
        "tenant_id": tenant_id,
        "revision": revision,
        "operation": OPERATION_DELETE,
        "target_generation": target_generation,
    }


def _intent_stmt(values: dict[str, Any], db):
    """Dialect-correct ``INSERT ... ON CONFLICT DO NOTHING`` (PG + SQLite).

    The unique ``(kind, entity_id, revision, target_generation, operation)``
    key is what makes a replayed enqueue a no-op instead of a duplicate.
    """
    try:
        dialect = db.get_bind().dialect.name
    except AttributeError:  # test double: the statement is never compiled
        dialect = "sqlite"
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as dialect_insert
    else:
        from sqlalchemy.dialects.sqlite import insert as dialect_insert
    return dialect_insert(IndexOutbox).values(**values).on_conflict_do_nothing()


async def _target_generation(db: AsyncSession) -> str:
    """Active generation for kind=memory, falling back to the transitional one.

    Memoized per session: a batch enqueue (import) reads the manifest once,
    never once per row.
    """
    cache = _session_cache(db)
    if cache is not None and _GENERATION_CACHE_KEY in cache:
        return cache[_GENERATION_CACHE_KEY]
    rows = (await db.execute(_active_generation_stmt())).scalars().all()
    generation = rows[0].generation if rows and isinstance(rows[0], IndexGeneration) else TARGET_GENERATION
    if cache is not None:
        cache[_GENERATION_CACHE_KEY] = generation
    return generation


def _target_generation_sync(db: Session) -> str:
    cache = _session_cache(db)
    if cache is not None and _GENERATION_CACHE_KEY in cache:
        return cache[_GENERATION_CACHE_KEY]
    rows = db.execute(_active_generation_stmt()).scalars().all()
    generation = rows[0].generation if rows and isinstance(rows[0], IndexGeneration) else TARGET_GENERATION
    if cache is not None:
        cache[_GENERATION_CACHE_KEY] = generation
    return generation


def _active_generation_stmt():
    return (
        select(IndexGeneration)
        .where(IndexGeneration.kind == KIND_MEMORY, IndexGeneration.is_active.is_(True))
        .limit(1)
    )


def _session_cache(db) -> dict | None:
    """The session's ``info`` dict, when it has one (test doubles may not)."""
    info = getattr(db, "info", None)
    return info if isinstance(info, dict) else None


# ── enqueue: async face (API / services) and sync face (Celery / CLI) ───────


async def enqueue_upsert(db: AsyncSession, memory) -> None:
    """Record the durable upsert intent for ``memory`` in ``db``'s transaction."""
    values = _upsert_values(memory, await _target_generation(db))
    await db.execute(_intent_stmt(values, db))


async def enqueue_delete(db: AsyncSession, *, entity_id: str, tenant_id: str, revision: int) -> None:
    """Record a durable delete intent in ``db``'s transaction."""
    values = _delete_values(entity_id=_entity_id(entity_id), tenant_id=_entity_id(tenant_id),
                            revision=int(revision), target_generation=await _target_generation(db))
    await db.execute(_intent_stmt(values, db))


def enqueue_upsert_sync(db: Session, memory) -> None:
    """Synchronous variant of :func:`enqueue_upsert`."""
    values = _upsert_values(memory, _target_generation_sync(db))
    db.execute(_intent_stmt(values, db))


def enqueue_delete_sync(db: Session, *, entity_id: str, tenant_id: str, revision: int) -> None:
    """Synchronous variant of :func:`enqueue_delete`."""
    values = _delete_values(entity_id=_entity_id(entity_id), tenant_id=_entity_id(tenant_id),
                            revision=int(revision), target_generation=_target_generation_sync(db))
    db.execute(_intent_stmt(values, db))


# ── drain ───────────────────────────────────────────────────────────────────


def _backoff_seconds(attempts: int) -> int:
    return min(_BACKOFF_BASE_SECONDS * 2 ** max(attempts - 1, 0), _BACKOFF_CAP_SECONDS)


def _error_text(exc: Exception) -> str:
    """Error type + message only — never a payload, always bounded."""
    return f"{type(exc).__name__}: {exc}"[:_ERROR_TEXT_LIMIT]


async def drain_pending(*, batch_size: int = 50) -> dict:
    """Apply pending intents in seq order against the latest SQL state.

    Success, a stale skip and a contract mismatch all ack the row (``done`` /
    ``blocked``); only a transient failure stays ``pending`` with exponential
    backoff. A ``blocked`` intent is terminal — an embedding-contract mismatch
    must not be retried silently.
    """
    report = {"claimed": 0, "applied": 0, "skipped": 0, "blocked": 0, "failed": 0}
    now = datetime.now(UTC)
    async with AsyncSessionLocal() as db:
        rows = (
            await db.execute(
                select(IndexOutbox)
                .where(
                    IndexOutbox.status == "pending",
                    or_(IndexOutbox.next_attempt_at.is_(None), IndexOutbox.next_attempt_at <= now),
                )
                .order_by(IndexOutbox.seq)
                .limit(batch_size)
            )
        ).scalars().all()
        report["claimed"] = len(rows)
        for row in rows:
            report[await _apply(db, row)] += 1
    return report


async def _apply(db: AsyncSession, row: IndexOutbox) -> str:
    """Apply one intent, ack it in its own commit, and return its report bucket."""
    if row.kind != KIND_MEMORY:
        # Not reachable in P1a (chunk intents arrive with P1b). Whatever owns
        # the kind must apply it; never guess and delete another kind's target.
        row.status, row.last_error = "blocked", f"unsupported outbox kind {row.kind!r}"
        outcome = "blocked"
    else:
        try:
            outcome = await _apply_memory_intent(db, row)
            row.status = "done"
        except EmbeddingDimensionMismatch as exc:
            row.status, row.last_error = "blocked", _error_text(exc)
            outcome = "blocked"
        except Exception as exc:
            row.attempts = int(row.attempts or 0) + 1
            row.next_attempt_at = datetime.now(UTC) + timedelta(seconds=_backoff_seconds(row.attempts))
            row.last_error = _error_text(exc)
            outcome = "failed"  # status stays 'pending'
            log.warning(
                "Outbox drain retry for %s#%s (attempt %d): %s",
                row.kind, row.entity_id, row.attempts, exc,
            )
    row.updated_at = datetime.now(UTC)
    await db.commit()
    return outcome


async def _apply_memory_intent(db: AsyncSession, row: IndexOutbox) -> str:
    entity_id = uuid.UUID(row.entity_id)
    memory = await db.get(Memory, entity_id)
    if memory is None:
        # The SQL row is gone: upsert and delete intents alike mean "forget".
        # Vector ids are the dashed UUID string (``upsert_memory`` contract).
        await delete_memory(str(entity_id))
        return "applied"
    if row.revision < int(memory.revision):
        # A newer write enqueued its own intent in the same commit; applying
        # this one would index an older revision over it.
        return "skipped"
    if row.operation == OPERATION_DELETE:
        await delete_memory(str(memory.id))
    else:
        await upsert_memory(memory)
    return "applied"


__all__ = [
    "KIND_MEMORY",
    "OPERATION_DELETE",
    "OPERATION_UPSERT",
    "TARGET_GENERATION",
    "bump_revision",
    "drain_pending",
    "enqueue_delete",
    "enqueue_delete_sync",
    "enqueue_upsert",
    "enqueue_upsert_sync",
]
