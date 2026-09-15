"""Durable index intent: enqueue in the same commit, drain after it (spec §5.1-§5.2).

Every canonical (SQL) mutation of an indexed entity stamps a monotonically
increasing ``revision`` on the row and inserts one ``index_outbox`` intent in
the SAME transaction — so a crash between the SQL commit and the vector write
is recoverable: the intent survives and :func:`drain_pending` replays it
against the LATEST SQL state. The existing post-commit write-through stays as
it was (best-effort, fast path); on success it acks its own intent
(:func:`mark_done`), so only an un-indexed write stays pending. The outbox is
the durable backstop, not a replacement.

P1b keeps the memory store and the document-chunk index on Qdrant
(:mod:`app.retrieval.memory.vector_store`, :mod:`app.retrieval.vector_retriever`)
and resolves each kind's physical generation through :func:`active_generation`.
A drain applies one intent at a time: a dead row collapses to a delete, a
stale revision is skipped (a newer intent owns the entity), an intent whose
``target_generation`` is no longer the kind's active one blocks terminally
(the cutover moved the pointer; the migration's backfill covers that write),
a contract mismatch blocks terminally, a transient failure retries with
backoff. ``mark_done`` carries the same generation predicate — a write into
the new generation never acks an old generation's obligation.
"""
from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from random import uniform
from typing import Any

from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.database import AsyncSessionLocal, sync_session
from app.models.conversation import Conversation
from app.models.document_chunk import DocumentChunk
from app.models.index_outbox import IndexGeneration, IndexOutbox
from app.models.memory import Memory
from app.retrieval.embedder import EmbeddingDimensionMismatch
from app.retrieval.memory.vector_store import COLLECTION_NAME, delete_memory, upsert_memory
from app.retrieval.vector_retriever import delete_chunks, upsert_chunks

log = logging.getLogger(__name__)

KIND_MEMORY = "memory"
KIND_CHUNK = "chunk"
OPERATION_UPSERT = "upsert"
OPERATION_DELETE = "delete"

# Transitional fallback when ``index_generations`` has no active row. It must
# keep the spelling the ladder seeds (controller ruling M3):
# ``vector_store.COLLECTION_NAME`` — a different spelling would break the
# unique intent key and enqueue the same logical write twice.
TARGET_GENERATION = COLLECTION_NAME
# The chunk family's fallback (ruling R13): ONE collection per generation,
# payload-filtered by tenant + conversation. Task 5 seeds the real row under
# this same spelling, so nothing here derives a second one.
CHUNK_TARGET_GENERATION = f"{COLLECTION_NAME}__chunks"

_FALLBACK_GENERATIONS = {KIND_MEMORY: TARGET_GENERATION, KIND_CHUNK: CHUNK_TARGET_GENERATION}

_BACKOFF_BASE_SECONDS = 60
_BACKOFF_CAP_SECONDS = 3600
_ERROR_TEXT_LIMIT = 500

# The active generation is stable for the life of one session/transaction
# (P1b swaps it between processes), so the manifest is read at most once per
# kind and session — a batch import must not pay one read per row.
def _generation_cache_key(kind: str) -> str:
    return f"orivory_outbox_generation:{kind}"


class VectorDeleteUnconfirmed(RuntimeError):
    """The vector backend did not confirm a delete.

    Transient by contract: the intent stays ``pending`` and the drain retries
    it instead of acking a delete that never happened.
    """


class IndexFreshnessTimeout(Exception):
    """A read waited for its own tenant's pending index intents and they did
    not land within the budget.

    Readiness, never a no-match: recall raises this instead of answering an
    empty result for a write that is merely still in flight, and the API
    answers 503 with the typed body ``{"error": "index_freshness_timeout"}``
    (see ``app.main``). Raised by
    :func:`app.retrieval.memory.freshness.await_freshness`.
    """


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
    revision = int(getattr(memory, "revision", 0) or 0)
    if revision <= 0:
        # Never guess this write's revision: a guessed one collides with an
        # earlier intent at the same revision (deduped away by the unique key)
        # or is acked stale by the drain — a silently lost index write either
        # way. The caller must have bumped in the same transaction.
        raise ValueError(
            f"memory {memory.id} has no revision to enqueue: call "
            "outbox.bump_revision(memory) in the same transaction as the write"
        )
    return {
        "kind": KIND_MEMORY,
        "entity_id": _entity_id(memory.id),
        "tenant_id": _entity_id(memory.user_id),
        "revision": revision,
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


def _fallback_generation(kind: str) -> str:
    try:
        return _FALLBACK_GENERATIONS[kind]
    except KeyError as exc:
        raise ValueError(f"unknown index kind {kind!r}") from exc


async def _target_generation(db: AsyncSession, kind: str) -> str:
    """Active generation for ``kind``, falling back to its transitional name.

    Memoized per session AND kind: a batch enqueue (import) reads the manifest
    once, never once per row.
    """
    cache = _session_cache(db)
    key = _generation_cache_key(kind)
    if cache is not None and key in cache:
        return cache[key]
    row = await _active_row(db, kind)
    generation = row.generation if row is not None else _fallback_generation(kind)
    if cache is not None:
        cache[key] = generation
    return generation


def _target_generation_sync(db: Session, kind: str) -> str:
    cache = _session_cache(db)
    key = _generation_cache_key(kind)
    if cache is not None and key in cache:
        return cache[key]
    row = _active_row_sync(db, kind)
    generation = row.generation if row is not None else _fallback_generation(kind)
    if cache is not None:
        cache[key] = generation
    return generation


def _active_generation_stmt(kind: str):
    return (
        select(IndexGeneration)
        .where(IndexGeneration.kind == kind, IndexGeneration.is_active.is_(True))
        .limit(1)
    )


async def _active_row(db: AsyncSession, kind: str) -> IndexGeneration | None:
    """The active ``index_generations`` row for ``kind``, or None."""
    rows = (await db.execute(_active_generation_stmt(kind))).scalars().all()
    return rows[0] if rows and isinstance(rows[0], IndexGeneration) else None


def _active_row_sync(db: Session, kind: str) -> IndexGeneration | None:
    rows = db.execute(_active_generation_stmt(kind)).scalars().all()
    return rows[0] if rows and isinstance(rows[0], IndexGeneration) else None


async def active_generation(*, kind: str = KIND_MEMORY) -> tuple[str, str | None]:
    """The active generation for ``kind`` as ``(generation, manifest fingerprint)``.

    The vector stores read their physical collection name from here: P1a's
    manifest stays the only resolution path, so a cutover is a pointer flip.
    The fingerprint is the row's ``fingerprint_generation`` token, which the
    contract guard compares against the active embedding contract; ``None``
    means there is no active row (the transitional fallback), so there is
    nothing to verify and a POPULATED generation must then refuse — P0's
    quarantine law, enforced by the caller's guard.
    """
    async with AsyncSessionLocal() as db:
        row = await _active_row(db, kind)
    return (row.generation, row.fingerprint) if row is not None else (_fallback_generation(kind), None)


def active_generation_sync(*, kind: str = KIND_MEMORY) -> tuple[str, str | None]:
    """Synchronous variant of :func:`active_generation` (Celery / CLI face)."""
    with sync_session() as db:
        row = _active_row_sync(db, kind)
    return (row.generation, row.fingerprint) if row is not None else (_fallback_generation(kind), None)


def _session_cache(db) -> dict | None:
    """The session's ``info`` dict, when it has one (test doubles may not)."""
    info = getattr(db, "info", None)
    return info if isinstance(info, dict) else None


# ── enqueue: async face (API / services) and sync face (Celery / CLI) ───────


async def enqueue_upsert(db: AsyncSession, memory) -> None:
    """Record the durable upsert intent for ``memory`` in ``db``'s transaction."""
    values = _upsert_values(memory, await _target_generation(db, KIND_MEMORY))
    await db.execute(_intent_stmt(values, db))


async def enqueue_delete(db: AsyncSession, *, entity_id: str, tenant_id: str, revision: int) -> None:
    """Record a durable delete intent in ``db``'s transaction."""
    values = _delete_values(entity_id=_entity_id(entity_id), tenant_id=_entity_id(tenant_id),
                            revision=int(revision),
                            target_generation=await _target_generation(db, KIND_MEMORY))
    await db.execute(_intent_stmt(values, db))


def enqueue_upsert_sync(db: Session, memory) -> None:
    """Synchronous variant of :func:`enqueue_upsert`."""
    values = _upsert_values(memory, _target_generation_sync(db, KIND_MEMORY))
    db.execute(_intent_stmt(values, db))


def enqueue_delete_sync(db: Session, *, entity_id: str, tenant_id: str, revision: int) -> None:
    """Synchronous variant of :func:`enqueue_delete`."""
    values = _delete_values(entity_id=_entity_id(entity_id), tenant_id=_entity_id(tenant_id),
                            revision=int(revision),
                            target_generation=_target_generation_sync(db, KIND_MEMORY))
    db.execute(_intent_stmt(values, db))


# ── enqueue: chunk kind (document reingest / document+conversation delete) ───


def _chunk_entity_ids(chunk_ids: Iterable[Any]) -> tuple[list[str], list[str]]:
    """Split ids into ``(hex entity ids, quarantined originals)``.

    A chunk id that is not a valid UUID is QUARANTINED — logged and returned —
    never "fixed", hashed or inserted under a guessed identity (spec §4.2,
    ruling R16). The caller decides whether a quarantine is fatal.
    """
    valid: list[str] = []
    quarantined: list[str] = []
    for value in chunk_ids:
        try:
            valid.append(_entity_id(value))
        except (AttributeError, TypeError, ValueError):
            quarantined.append(str(value))
    return valid, quarantined


def _quarantine_log(quarantined: list[str], *, context: str) -> None:
    log.warning(
        "Quarantined chunk id(s): not a UUID, never indexed",
        extra={"context": context, "n": len(quarantined), "chunk_id": quarantined[0]},
    )


def _chunk_upsert_values(*, entity_id: str, tenant_id: str, revision: int,
                         target_generation: str) -> dict[str, Any]:
    if int(revision) <= 0:
        # Same law as the memory twin (:func:`_upsert_values`): never guess or
        # default this write's revision. A zero would dedupe onto an earlier
        # revision-0 intent (the unique key drops it) or be acked stale by the
        # drain — a silently lost index write either way.
        raise ValueError(
            f"chunk {entity_id} has no revision to enqueue: pass the SQL row's "
            "revision (> 0) in the same transaction as the write"
        )
    return {
        "kind": KIND_CHUNK,
        "entity_id": entity_id,
        "tenant_id": tenant_id,
        "revision": int(revision),
        "operation": OPERATION_UPSERT,
        "target_generation": target_generation,
    }


def _chunk_delete_values(*, entity_id: str, tenant_id: str, target_generation: str) -> dict[str, Any]:
    """A delete intent names an IDENTITY, not a revision.

    Chunk ids are never reused (a reingest mints new UUIDs), so revision 0 is
    the honest "not revision-gated" value — the applier deletes by id and its
    readback decides whether the intent may be acked. It also dedupes a
    document delete and a conversation delete of the same point into one row.
    """
    return {
        "kind": KIND_CHUNK,
        "entity_id": entity_id,
        "tenant_id": tenant_id,
        "revision": 0,
        "operation": OPERATION_DELETE,
        "target_generation": target_generation,
    }


def enqueue_chunk_upsert_sync(
    db: Session, *, chunk_id: Any, tenant_id: Any, revision: int, conversation_id: Any
) -> list[str]:
    """Record the durable upsert intent for one chunk in ``db``'s transaction.

    The ingestion face is the ONLY caller (ingestion is sync); there is no
    async twin to leave dangling. ``conversation_id`` only names the write in
    the quarantine report; the intent's own columns (identity, tenant,
    revision, generation) are what the drain reads back. Returns the
    quarantined ids (empty when the id is valid).
    """
    chunk_ids, quarantined = _chunk_entity_ids([chunk_id])
    if quarantined:
        _quarantine_log(quarantined, context=f"upsert conversation:{conversation_id}")
        return quarantined
    values = _chunk_upsert_values(
        entity_id=chunk_ids[0], tenant_id=_entity_id(tenant_id), revision=revision,
        target_generation=_target_generation_sync(db, KIND_CHUNK),
    )
    db.execute(_intent_stmt(values, db))
    return []


async def enqueue_chunk_delete(db: AsyncSession, *, chunk_ids: Iterable[Any],
                               tenant_id: Any) -> list[str]:
    """Record one durable delete intent per chunk id in ``db``'s transaction."""
    return await _enqueue_chunk_deletes(
        db, chunk_ids=chunk_ids, tenant_id=tenant_id,
        target_generation=await _target_generation(db, KIND_CHUNK),
    )


def enqueue_chunk_delete_sync(db: Session, *, chunk_ids: Iterable[Any],
                              tenant_id: Any) -> list[str]:
    """Synchronous variant of :func:`enqueue_chunk_delete`."""
    return _enqueue_chunk_deletes_sync(
        db, chunk_ids=chunk_ids, tenant_id=tenant_id,
        target_generation=_target_generation_sync(db, KIND_CHUNK),
    )


async def _enqueue_chunk_deletes(db: AsyncSession, *, chunk_ids: Iterable[Any], tenant_id: Any,
                                 target_generation: str) -> list[str]:
    valid, quarantined = _chunk_entity_ids(chunk_ids)
    if quarantined:
        _quarantine_log(quarantined, context="delete")
    tenant = _entity_id(tenant_id)
    for entity_id in valid:
        await db.execute(_intent_stmt(
            _chunk_delete_values(entity_id=entity_id, tenant_id=tenant,
                                 target_generation=target_generation), db))
    return quarantined


def _enqueue_chunk_deletes_sync(db: Session, *, chunk_ids: Iterable[Any], tenant_id: Any,
                                target_generation: str) -> list[str]:
    valid, quarantined = _chunk_entity_ids(chunk_ids)
    if quarantined:
        _quarantine_log(quarantined, context="delete")
    tenant = _entity_id(tenant_id)
    for entity_id in valid:
        db.execute(_intent_stmt(
            _chunk_delete_values(entity_id=entity_id, tenant_id=tenant,
                                 target_generation=target_generation), db))
    return quarantined


# ── ack: the immediate write-through already indexed this revision ──────────


def _mark_done_stmt(*, entity_id, revision: int, kind: str = KIND_MEMORY,
                    target_generation: str | None = None):
    """UPDATE flipping ONLY the pending upsert intent for that entity+revision.

    Delete intents, other revisions, other kinds — and (ruling R27) other
    GENERATIONS — are different obligations and are matched out by the WHERE
    clause: a write-through into the active generation must never ack an intent
    that promised a different one.
    """
    conditions = [
        IndexOutbox.kind == kind,
        IndexOutbox.entity_id == _entity_id(entity_id),
        IndexOutbox.revision == int(revision),
        IndexOutbox.operation == OPERATION_UPSERT,
        IndexOutbox.status == "pending",
    ]
    if target_generation is not None:
        conditions.append(IndexOutbox.target_generation == target_generation)
    return (
        update(IndexOutbox)
        .where(*conditions)
        .values(status="done", updated_at=datetime.now(UTC))
    )


async def mark_done(db: AsyncSession, *, entity_id, revision: int,
                    kind: str = KIND_MEMORY) -> int:
    """Ack this entity+revision's upsert intent after a SUCCESSFUL write-through.

    Callers index the latest SQL state right after their commit; without this
    the row would stay ``pending`` and every boot would re-embed it. Only ever
    called when the vector write landed — a failure must leave the intent
    pending, since that pending row is the proof the vector still owes it.
    Generation-aware (R27): the ack only ever lands on an intent whose target
    generation IS the kind's active one, so a write into the new generation can
    never close an old generation's obligation.

    Never raises: an ack failure is logged and the intent stays ``pending``,
    which is safe (the drain replays it) and must not fail the caller's write.
    Commits its own UPDATE — the ack must not ride a session whose next commit
    may never come (``get_db`` does not commit).
    """
    try:
        result = await db.execute(
            _mark_done_stmt(entity_id=entity_id, revision=revision, kind=kind,
                            target_generation=await _target_generation(db, kind)))
        await db.commit()
        return getattr(result, "rowcount", 0) or 0
    except Exception as exc:
        log.warning("Outbox ack failed for %s@%s: %s", entity_id, revision, exc)
        return 0


def mark_done_sync(db: Session, *, entity_id, revision: int, kind: str = KIND_MEMORY) -> int:
    """Synchronous variant of :func:`mark_done` (Celery / CLI / ingestion)."""
    try:
        result = db.execute(_mark_done_stmt(
            entity_id=entity_id, revision=revision, kind=kind,
            target_generation=_target_generation_sync(db, kind)))
        db.commit()
        return getattr(result, "rowcount", 0) or 0
    except Exception as exc:
        log.warning("Outbox ack failed for %s@%s: %s", entity_id, revision, exc)
        return 0


# ── drain ───────────────────────────────────────────────────────────────────


def _backoff_seconds(attempts: int) -> float:
    """Next wait: exponential ladder + bounded jitter (spec §5.2).

    The jitter (up to 10% of the step, capped at 30s) keeps a fleet of installs
    that failed in the same boot from retrying in lockstep.
    """
    backoff = min(_BACKOFF_BASE_SECONDS * 2 ** max(attempts - 1, 0), _BACKOFF_CAP_SECONDS)
    return backoff + uniform(0, min(backoff * 0.1, 30))


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
    """Apply one intent, ack it in its own commit, and return its report bucket.

    Generation-aware (ruling R27): an intent whose ``target_generation`` is not
    the kind's ACTIVE generation is never applied. The P1b cutover moves the
    pointer, and the migration's backfill already covers those writes — applying
    one would index into a generation the app does not serve. Terminal, exactly
    like a contract mismatch: never retried, never silently dropped.
    """
    if row.kind not in (KIND_MEMORY, KIND_CHUNK):
        # Whatever owns the kind must apply it; never guess and delete another
        # kind's target.
        row.status, row.last_error = "blocked", f"unsupported outbox kind {row.kind!r}"
        outcome = "blocked"
    else:
        active = await _target_generation(db, row.kind)
        if row.target_generation != active:
            row.status = "blocked"
            row.last_error = (
                f"generation {row.target_generation!r} superseded by {active!r}"
            )[:_ERROR_TEXT_LIMIT]
            outcome = "blocked"
        else:
            try:
                if row.kind == KIND_MEMORY:
                    outcome = await _apply_memory_intent(db, row)
                else:
                    outcome = await _apply_chunk_intent(db, row)
                row.status = "done"
            except EmbeddingDimensionMismatch as exc:
                row.status, row.last_error = "blocked", _error_text(exc)
                outcome = "blocked"
            except Exception as exc:
                row.attempts = int(row.attempts or 0) + 1
                row.next_attempt_at = datetime.now(UTC) + timedelta(
                    seconds=_backoff_seconds(row.attempts))
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
        await _delete_vector_or_fail(str(entity_id))
        return "applied"
    if row.revision < int(memory.revision):
        # A newer write enqueued its own intent in the same commit; applying
        # this one would index an older revision over it.
        return "skipped"
    if row.operation == OPERATION_DELETE:
        await _delete_vector_or_fail(str(memory.id))
    else:
        await upsert_memory(memory)
    return "applied"


async def _delete_vector_or_fail(entity_id: str) -> None:
    """Raise unless the vector backend confirmed the delete.

    An unconfirmed delete must never ack ``done``: the intent is the only
    record that the vector still has to go, so a failure stays pending and
    rides the normal backoff retry.
    """
    if not await delete_memory(entity_id):
        raise VectorDeleteUnconfirmed(f"vector delete not confirmed for {entity_id}")


async def _apply_chunk_intent(db: AsyncSession, row: IndexOutbox) -> str:
    """Apply one chunk intent against the LATEST SQL state.

    A dead row and a delete intent both mean "forget": the point is deleted by
    id and only a confirmed readback lets the intent be acked. A live row with
    a newer revision is skipped — the newer intent owns the point.
    """
    entity_id = uuid.UUID(row.entity_id)
    chunk = await db.get(DocumentChunk, entity_id)
    # Invariant: this revision-0 delete bypass is safe ONLY while chunk ids are
    # never reused. A future in-place chunk writer MUST bump ``revision`` AND
    # enqueue its upsert in the same transaction — otherwise a stale delete at
    # revision 0 removes the fresh point and acks itself.
    if chunk is None or row.operation == OPERATION_DELETE:
        await _delete_chunk_vector_or_fail([str(entity_id)])
        return "applied"
    if row.revision < int(chunk.revision or 0):
        # A newer write enqueued its own intent in the same commit; applying
        # this one would index an older revision over it.
        return "skipped"
    owner_id = await _chunk_owner(db, chunk)
    if owner_id is None:
        # No owner left in SQL to scope the payload's tenant clause to: the
        # point must not outlive the identity it was written under.
        await _delete_chunk_vector_or_fail([str(entity_id)])
        return "applied"
    await upsert_chunks([chunk], user_id=owner_id)
    return "applied"


async def _chunk_owner(db: AsyncSession, chunk: DocumentChunk) -> str | None:
    """The chunk's tenant as SQL knows it — never the intent's tenant body."""
    conversation_id = (chunk.chunk_metadata or {}).get("conversation_id")
    if not conversation_id:
        return None
    try:
        conversation = await db.get(Conversation, uuid.UUID(str(conversation_id)))
    except ValueError:
        return None
    return None if conversation is None else str(conversation.user_id)


async def _delete_chunk_vector_or_fail(chunk_ids: list[str]) -> None:
    """Raise unless the vector backend confirmed the delete.

    An unconfirmed delete must never ack ``done``: the intent is the only
    record that the vector still has to go, so a failure stays pending and
    rides the normal backoff retry.
    """
    if not await delete_chunks(chunk_ids):
        raise VectorDeleteUnconfirmed(f"chunk vector delete not confirmed for {chunk_ids}")


__all__ = [
    "CHUNK_TARGET_GENERATION",
    "KIND_CHUNK",
    "KIND_MEMORY",
    "OPERATION_DELETE",
    "OPERATION_UPSERT",
    "TARGET_GENERATION",
    "IndexFreshnessTimeout",
    "VectorDeleteUnconfirmed",
    "active_generation",
    "active_generation_sync",
    "bump_revision",
    "drain_pending",
    "enqueue_chunk_delete",
    "enqueue_chunk_delete_sync",
    "enqueue_chunk_upsert_sync",
    "enqueue_delete",
    "enqueue_delete_sync",
    "enqueue_upsert",
    "enqueue_upsert_sync",
    "mark_done",
    "mark_done_sync",
]
