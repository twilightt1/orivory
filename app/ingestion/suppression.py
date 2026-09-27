"""The suppression ledger: a forgotten identity stays forgotten.

Every path that could bring a forgotten memory back — a re-import, a re-upload
of the same bytes, an outbox replay racing the forget — asks this module, and
only this module, whether the identity is still forbidden. It knows nothing
about documents, chunking or embeddings: those live in ``document_memory``,
and the projection there reads this ledger, never the reverse.

One rule, four call sites. Keeping the rule here rather than inside the
projection is what makes it possible to pin it once and trust every reader.
"""
from __future__ import annotations

import uuid
from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.models.memory import MemorySuppression

__all__ = [
    "CONTENT_HASH_KEY",
    "PRODUCER_SOURCE_PREFIX",
    "is_content_suppressed_async",
    "is_suppressed",
    "is_suppressed_async",
    "projection_content_hash",
    "suppress_source",
    "suppress_source_async",
    "suppressed_refs_async",
]


# ── suppression ledger: a forgotten identity stays forgotten ────────────────

#: The projection metadata key carrying the upload-time content hash (R38).
#: Computed from the uploaded BYTES at ingest, never backfilled onto rows that
#: predate it — an absent key is the honest "unknown", not an invented hash.
CONTENT_HASH_KEY = "content_hash"


def _ledger_stmt(column, user_id):
    """One ledger column, scoped to one owner — the probe's shared half."""
    return select(column).where(MemorySuppression.user_id == user_id)


def _suppression_stmt(user_id, source_ref: str):
    return _ledger_stmt(MemorySuppression.id, user_id).where(
        MemorySuppression.source_ref == source_ref)


def is_suppressed(db: Session, *, user_id, source_ref: str) -> bool:
    """True when this (user, source_ref) was forgotten and must not come back."""
    return db.execute(_suppression_stmt(user_id, source_ref)).first() is not None


async def is_suppressed_async(db: AsyncSession, *, user_id, source_ref: str) -> bool:
    """Async face of :func:`is_suppressed`."""
    return (await db.execute(_suppression_stmt(user_id, source_ref))).first() is not None


async def is_content_suppressed_async(db: AsyncSession, *, user_id,
                                      content_hash: str | None) -> bool:
    """True when this user forgot a source with these BYTES (R38/T4).

    The re-upload guard's probe: a re-upload mints a NEW document id, so a
    ``source_ref`` match cannot see it — the ledger's ``content_hash`` (written
    when the source was forgotten) is the only key that survives the new id.
    ``None`` never matches: an unknown hash is not a forgotten identity.
    """
    if not content_hash:
        return False
    stmt = _ledger_stmt(MemorySuppression.id, user_id).where(
        MemorySuppression.content_hash == content_hash)
    return (await db.execute(stmt)).first() is not None


async def suppressed_refs_async(db: AsyncSession, *, user_id,
                                source_refs: Iterable[str]) -> set[str]:
    """Which of ``source_refs`` this user forgot — ONE batched ledger read.

    The import guard's probe (never one query per item): the same shape as the
    import path's dedup select, against the same ``(user_id, source_ref)`` key.
    The ledger's unique index answers it in one pass.
    """
    refs = [ref for ref in source_refs if ref]
    if not refs:
        return set()
    stmt = _ledger_stmt(MemorySuppression.source_ref, user_id).where(
        MemorySuppression.source_ref.in_(refs))
    return {row for row in (await db.execute(stmt)).scalars().all() if row}


# Producer labels are WRITER identities (the MCP boundary files every agent
# memory under one shared ``agent:<name>`` ref — app/mcp_hub/tools.py), never
# re-importable sources: no pipeline re-creates that ref, so pinning it would
# only block every FUTURE memory the same producer adds (its outbox/reindex
# guards read the ledger by ``source_ref``). Forgotten rows still leave
# serving via ``cm_invalidated``; the ledger entry is simply not written.
PRODUCER_SOURCE_PREFIX = "agent:"


def _fill_missing_suppression_fields(existing, *, namespace: str | None,
                                     content_hash: str | None) -> None:
    """Let a predating row pick up fields it never knew (NULL -> value only).

    R38: nothing is backfilled across time — but when a LATER forget of the
    same identity brings a value the first row was missing, the ledger should
    learn it, otherwise a re-upload of the same bytes slips past the hash
    guard forever. The first writer's values always win.
    """
    if existing.content_hash is None and content_hash is not None:
        existing.content_hash = content_hash
    if existing.namespace is None and namespace is not None:
        existing.namespace = namespace


def _fill_losing_suppression(db: Session, *, user_id, source_ref: str,
                             namespace: str | None, content_hash: str | None) -> None:
    """Teach the ledger what the LOSING side of a suppression race carried.

    The rival's row commits between our pre-read and our insert, so this
    caller saw neither the row nor what it holds: reload it and fill what it
    lacks (same NULL-only rule as the found-row branch — the first writer's
    values always win). Without this the hash that pins the BYTES is dropped on
    the floor and a re-upload of the same file slips past the guard.
    Caller's transaction; the caller commits.
    """
    existing = db.execute(
        select(MemorySuppression).where(
            MemorySuppression.user_id == user_id,
            MemorySuppression.source_ref == source_ref)
    ).scalar_one_or_none()
    if existing is not None:
        _fill_missing_suppression_fields(existing, namespace=namespace,
                                         content_hash=content_hash)
        db.flush()


async def _fill_losing_suppression_async(db: AsyncSession, *, user_id, source_ref: str,
                                          namespace: str | None,
                                          content_hash: str | None) -> None:
    """Async face of :func:`_fill_losing_suppression` — same row, same rule."""
    existing = (await db.execute(
        select(MemorySuppression).where(
            MemorySuppression.user_id == user_id,
            MemorySuppression.source_ref == source_ref)
    )).scalar_one_or_none()
    if existing is not None:
        _fill_missing_suppression_fields(existing, namespace=namespace,
                                         content_hash=content_hash)
        await db.flush()


def projection_content_hash(row) -> str | None:
    """The upload-time content hash a projection row carries, or ``None``.

    Read by the forget path to pin BYTES in the ledger, not only the doc id
    (R38). Only the shape the ingest writes is trusted — a 64-char hex string:
    ``extra_metadata`` is client-reachable, and a forged or oversized value
    must reach the ledger as "unknown" (``None``), never as a pin. A row that
    predates the hash, or a non-document memory, reads ``None``.
    """
    value = (getattr(row, "extra_metadata", None) or {}).get(CONTENT_HASH_KEY)
    if not isinstance(value, str) or len(value) != 64:
        return None
    try:
        bytes.fromhex(value)
    except ValueError:
        return None
    return value


def suppress_source(db: Session, *, user_id, source_ref: str, reason: str = "forgotten",
                    namespace: str | None = None, content_hash: str | None = None) -> bool:
    """Record that this identity was forgotten (idempotent, caller commits).

    The unique ``(user_id, source_ref)`` keeps exactly one suppression row, so
    replaying the forget is a no-op instead of an integrity error. ``namespace``
    is the boundary the suppression was written in; ``content_hash`` is the
    forgotten projection's upload-time hash, supplied by the caller — a forget
    passes ``projection_content_hash(row)``, the re-upload guard passes the
    sha256 it just computed from the incoming bytes — the import/reindex/drain
    guards only read this ledger, they never write it. A caller with no hash
    passes nothing and the column stays NULL (R38: never backfilled).

    Returns whether the ledger now says "suppressed" — ``False`` only for a
    producer label (``PRODUCER_SOURCE_PREFIX``: see the constant), so a
    caller's receipt never counts a pin that was never written. A lost
    check-then-insert race resolves to "already suppressed" instead of
    aborting the caller's transaction.
    """
    if source_ref and source_ref.startswith(PRODUCER_SOURCE_PREFIX):
        return False
    existing = db.execute(
        select(MemorySuppression).where(
            MemorySuppression.user_id == user_id,
            MemorySuppression.source_ref == source_ref)
    ).scalar_one_or_none()
    if existing is not None:
        _fill_missing_suppression_fields(existing, namespace=namespace,
                                         content_hash=content_hash)
        db.flush()
        return True
    # The row is ADDED INSIDE the savepoint: an object that is already pending
    # when the savepoint begins is flushed by a rollback that cannot retract it,
    # and the losing side of a check-then-insert race then left the caller's
    # outer transaction inactive (its next commit died PendingRollbackError).
    # Flush so a replay inside the same transaction sees the row (the session
    # does not autoflush) instead of racing the unique constraint: a rival
    # forget that landed the row first answers IntegrityError, the savepoint
    # rolls back — and THIS caller's other work stays alive.
    try:
        with db.begin_nested():
            db.add(MemorySuppression(id=uuid.uuid4().hex, user_id=user_id,
                                     source_ref=source_ref, reason=reason,
                                     namespace=namespace, content_hash=content_hash))
            db.flush()
    except IntegrityError:
        # A rival forget landed the row between our read and our insert. That
        # row is the ledger, and its own forget may have carried nothing —
        # while THIS one carries the hash that pins the bytes (or the namespace
        # that scopes them). Teach the winner what it lacks: a re-upload of the
        # same file is otherwise unrecognisable to the hash guard forever.
        _fill_losing_suppression(db, user_id=user_id, source_ref=source_ref,
                                 namespace=namespace, content_hash=content_hash)
    return True


async def suppress_source_async(db: AsyncSession, *, user_id, source_ref: str,
                                reason: str = "forgotten", namespace: str | None = None,
                                content_hash: str | None = None) -> bool:
    """Async face of :func:`suppress_source` — same row, same fields, same bool.

    ``content_hash`` is the same caller-supplied value (a forget's
    ``projection_content_hash(row)``, or the re-upload guard's freshly computed
    sha256 of the incoming bytes); nothing here computes or backfills one.
    """
    if source_ref and source_ref.startswith(PRODUCER_SOURCE_PREFIX):
        return False
    existing = (await db.execute(
        select(MemorySuppression).where(
            MemorySuppression.user_id == user_id,
            MemorySuppression.source_ref == source_ref)
    )).scalar_one_or_none()
    if existing is not None:
        _fill_missing_suppression_fields(existing, namespace=namespace,
                                         content_hash=content_hash)
        await db.flush()
        return True
    # Added INSIDE the savepoint, same reason as the sync face above (id 70):
    # a pre-savepoint IntegrityError leaves the OUTER transaction inactive and
    # the caller's next commit fails instead of landing its other work.
    try:
        async with db.begin_nested():
            db.add(MemorySuppression(id=uuid.uuid4().hex, user_id=user_id,
                                     source_ref=source_ref, reason=reason,
                                     namespace=namespace, content_hash=content_hash))
            await db.flush()
    except IntegrityError:
        # Same race, same rule as the sync face: reload the winner and teach it
        # what this forget carries (see ``_fill_losing_suppression``).
        await _fill_losing_suppression_async(
            db, user_id=user_id, source_ref=source_ref,
            namespace=namespace, content_hash=content_hash,
        )
    return True
