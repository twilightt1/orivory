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
stale revision is skipped (a newer intent owns the entity), an applied upsert
re-reads its own row once — the drain runs concurrently with writers, so a row
deleted or superseded while its snapshot was in flight has the point just
written deleted (row gone) or REWRITTEN from the row the re-check read
(revision advanced, still reported ``skipped``; ruling R22): the newer
revision's intent may already have been acked by the request-path write-through
(``mark_done``), leaving the superseded payload ownerless forever — the
freshness barrier cannot see it because nothing is pending — and the rewrite is
idempotent when that intent IS still pending (it lands the same payload).
Accepted residual, documented not coded: a THIRD write landing between the
re-check and the rewrite is still unfenced; the true fix is a revision-fenced
write and Qdrant has no compare-and-set, so it is out of P3 scope. See
:func:`_settle_written_snapshot`. An intent whose
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

from sqlalchemy import case, or_, select, update
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

    ``retry_after`` (a UTC datetime) is the earliest moment a pending intent of
    that tenant can even be claimed — set ONLY when that is the reason the wait
    was pointless (every pending intent is waiting out its backoff), and
    ``None`` when the intents were still in flight when the budget ran out. It
    is reported through the message and the warning log, never on the wire: the
    503 body is pinned to the typed error alone (a signed gate).
    """

    def __init__(self, *args: object, retry_after: datetime | None = None) -> None:
        super().__init__(*args)
        self.retry_after = retry_after


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
    """The intent's column values, read off the row that enqueued it.

    There is deliberately no ``namespace`` column here (and no copy of it in
    the payload schema this record writes): the record carries the TENANT, and
    the applier re-reads the row, so the namespace a point is written with is
    the row's own AT APPLY TIME — via ``vector_store._memory_to_metadata``,
    never a second, staler copy of an authorization boundary parked on the
    intent (R32(p4a): a missing key counts as personal; the record itself
    cannot go missing, since the read is by primary key).
    """
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


async def drain_pending(*, batch_size: int = 50, priority_tenant: str | None = None) -> dict:
    """Apply pending intents in seq order against the latest SQL state.

    Success, a stale skip and a contract mismatch all ack the row (``done`` /
    ``blocked``); only a transient failure stays ``pending`` with exponential
    backoff. A ``blocked`` intent is terminal — an embedding-contract mismatch
    must not be retried silently.

    ``priority_tenant`` (the read-path barrier's own tenant) puts that tenant's
    rows FIRST in the batch: the caller is waiting on them, and a foreign
    backlog (a bulk import's chunk intents) otherwise fills every batch it
    triggers. The background loop passes no tenant, so its seq fairness is
    untouched — and within the batch the seq order still holds.
    """
    report = {"claimed": 0, "applied": 0, "skipped": 0, "blocked": 0, "failed": 0}
    now = datetime.now(UTC)
    if priority_tenant is None:
        order_by = [IndexOutbox.seq]
    else:
        order_by = [
            case((IndexOutbox.tenant_id == priority_tenant, 0), else_=1),
            IndexOutbox.seq,
        ]
    async with AsyncSessionLocal() as db:
        rows = (
            await db.execute(
                select(IndexOutbox)
                .where(
                    IndexOutbox.status == "pending",
                    or_(IndexOutbox.next_attempt_at.is_(None), IndexOutbox.next_attempt_at <= now),
                )
                .order_by(*order_by)
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
    # Local import: ``correction`` imports this module by value (real cycle),
    # so the state label cannot be a module-level import here.
    from app.retrieval.memory.correction import state_of

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
        # A delete intent has no snapshot to go stale: no post-write re-check.
        await _delete_vector_or_fail(str(memory.id))
        return "applied"
    if state_of(memory) in ("current", "needs-check") and memory.source_ref:
        # R38/T4: a SERVING payload for a source the user forgot must not
        # (re)enter the index — this is the resurrection path (an intent left
        # over from before the forget, or a writer that touched a row sharing a
        # suppressed source identity).
        #
        # The gate is the STATE, not the suppression alone: soft forget's own
        # payload refresh (R37) is an upsert for a row that is `invalidated` BY
        # CONSTRUCTION and whose source is suppressed by construction too.
        # Blocking it would leave the point serving `visibility_state=current`
        # — exactly the state contract that refresh exists to land.
        from app.ingestion.document_memory import is_suppressed_async

        if await is_suppressed_async(db, user_id=memory.user_id, source_ref=memory.source_ref):
            log.info(
                "Outbox memory upsert skipped: source suppressed",
                extra={"entity_id": str(entity_id), "source_ref": memory.source_ref,
                       "state": state_of(memory)},
            )
            return "skipped"
    await upsert_memory(memory)
    return await _settle_written_snapshot(
        db, row, Memory, entity_id,
        rewrite=upsert_memory,
        purge=lambda: _delete_vector_or_fail(str(entity_id)),
    )


async def _delete_vector_or_fail(entity_id: str) -> None:
    """Raise unless the vector backend confirmed the delete.

    An unconfirmed delete must never ack ``done``: the intent is the only
    record that the vector still has to go, so a failure stays pending and
    rides the normal backoff retry.

    Deleted by POINT ID — the entity's own UUID — so this is identity-exact and
    needs no namespace clause (R32(p4a) is about the READ filters: a point that
    predates the key is personal). A namespace-scoped SELECTOR here would only
    add a way to MISS the point — the intent would then ride the retry loop
    forever while the point stayed served — so the boundary stays on the
    filter-based deletes (the chunk family's sweeps) and on every read.
    """
    if not await delete_memory(entity_id):
        raise VectorDeleteUnconfirmed(f"vector delete not confirmed for {entity_id}")


async def _apply_chunk_intent(db: AsyncSession, row: IndexOutbox) -> str:
    """Apply one chunk intent against the LATEST SQL state.

    A dead row and a delete intent both mean "forget": the point is deleted by
    id and only a confirmed readback lets the intent be acked. A live row with
    a newer revision has the point just written REWRITTEN from the refreshed
    row (reported ``skipped``; R22) — see :func:`_settle_written_snapshot`.
    An applied upsert is re-checked against its row afterwards.
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

    async def rewrite(fresh: DocumentChunk) -> None:
        # R22: land the refreshed row's payload — under the tenant SQL names for
        # it NOW, not the one resolved before the write — and delete it outright
        # when that identity is gone (the pre-write branch's rule, re-applied to
        # the state the re-check saw).
        fresh_owner = await _chunk_owner(db, fresh)
        if fresh_owner is None:
            await _delete_chunk_vector_or_fail([str(entity_id)])
        else:
            await upsert_chunks([fresh], user_id=fresh_owner)

    return await _settle_written_snapshot(
        db, row, DocumentChunk, entity_id,
        rewrite=rewrite,
        purge=lambda: _delete_chunk_vector_or_fail([str(entity_id)]),
    )


async def _settle_written_snapshot(db: AsyncSession, row: IndexOutbox, model, entity_id, *,
                                   rewrite, purge) -> str:
    """Settle a JUST-WRITTEN upsert snapshot against the row it came from (R6/R21/R22).

    The drain runs concurrently with writers, so between the applier's read and
    its write the row can be deleted (the point then outlives its row) or
    superseded by a newer revision (a correction enqueues its own intent in the
    same commit). Either way the point just written came from a snapshot that is
    no longer the row's state, and acking it as-is would leave the store wrong
    with nothing pending to fix it.

    One extra read per applied upsert, same session — ``populate_existing`` is
    what makes it a real read: the pre-read left the instance in the identity
    map, so a plain ``get`` would answer from the snapshot and never see the
    race. The re-check is deliberately bounded (never a reconciliation pass):

    - row gone → the point just written is deleted (``purge``, which raises
      unless the store confirmed absence) and the intent reports ``applied``;
    - revision advanced → the point is REWRITTEN from the refreshed row
      (``rewrite``) and the intent still reports ``skipped`` (R22). Standing
      down is only safe while the newer intent is still pending: the
      request-path write-through can have acked it already (``mark_done``), and
      the superseded payload then has no owner — the freshness barrier cannot
      see it, nothing is pending. Rewriting is idempotent when the newer intent
      IS still pending: it lands the same payload that intent will land, one
      bounded write, no loop, no new session;
    - row unchanged → ``applied``, i.e. the ordinary case.

    Two premises are load-bearing:

    - the row is read fresh (the ``populate_existing`` note above);
    - that read is a STATEMENT-level snapshot — PG READ COMMITTED, or pysqlite's
      per-SELECT view (the driver only opens an implicit BEGIN for writes). At a
      stricter isolation level (REPEATABLE READ / SERIALIZABLE) the re-check
      answers from the transaction's own snapshot and silently degrades to the
      same no-op class the ``populate_existing`` warning covers.

    Accepted residual (R22): a THIRD write landing between this re-read and the
    rewrite stays unfenced — fencing it needs a revision-fenced write, and
    Qdrant has no compare-and-set. Out of P3 scope; documented, not coded.
    """
    intent_revision = int(row.revision)
    fresh = await db.get(model, entity_id, populate_existing=True)
    if fresh is None:
        await purge()
        return "applied"
    if int(getattr(fresh, "revision", 0) or 0) > intent_revision:
        await rewrite(fresh)
        return "skipped"
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
