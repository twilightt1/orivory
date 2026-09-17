"""Soft forget (P4b Task 3): invalidate + universal suppression + payload intent.

Real SQLite files (fixtures in this package's conftest) — every assertion below
reads the real tables. The vector store is NOT touched at all: R37 says soft
forget never purges a point in place; it refreshes the payload by INTENT
(``bump_revision`` + ``enqueue_upsert``) and the purge belongs to reconciliation
repair.

Rulings this module pins:

- R35: the tool boundary is the tool's (see tests/mcp_hub/test_forget_memory):
  here the service checks ownership per row and refuses to widen a closure.
- R37: the rows stay (provenance + evidence intact) and are ``invalidated``;
  every invalidated row owes the index a real upsert at its bumped revision,
  because ``vector_store._memory_to_metadata`` writes ``visibility_state`` from
  ``state_of`` — the invalidated state must REACH the payload.
- R38: every affected ``source_ref`` — not only the root projection — gets a
  ``MemorySuppression`` row, and the ledger carries the ``namespace`` it pinned
  (``content_hash`` stays NULL until uploads compute one: no backfill).
"""
from __future__ import annotations

import uuid

from sqlalchemy import select

from app import database
from app.models.index_outbox import IndexOutbox
from app.models.memory import Memory, MemorySuppression
from app.models.user import User
from app.retrieval.memory import correction
from app.retrieval.memory.correction import (
    CM_EVIDENCE_IDS,
    CM_SUBJECT,
    state_of,
)
from app.retrieval.memory.outbox import KIND_MEMORY, OPERATION_UPSERT
from app.retrieval.memory.visibility import not_dirty_predicate
from app.services import erasure_service
from app.services.erasure_service import (
    ERASURE_STATUS_COMPLETED,
    ERASURE_STATUS_ERRORS,
    ERASURE_STATUS_RESIDUAL,
    ERASURE_STATUS_UNVERIFIED,
    erase_memories,
    soft_forget,
)


async def _owner(db) -> uuid.UUID:
    """A user row — ``memories.user_id`` is a FK."""
    uid = uuid.uuid4()
    db.add(User(id=uid, email=f"{uid.hex}@test.invalid", hashed_password="x",
                display_name="Owner", is_verified=True, is_active=True))
    await db.commit()
    return uid


def _memory(user_id, content="x", **kwargs) -> Memory:
    return Memory(id=uuid.uuid4(), user_id=user_id, content=content, tags=[], **kwargs)


async def _outbox_rows() -> list[IndexOutbox]:
    async with database.AsyncSessionLocal() as session:
        return list((await session.execute(
            select(IndexOutbox).order_by(IndexOutbox.seq))).scalars().all())


async def _suppressions() -> list[MemorySuppression]:
    async with database.AsyncSessionLocal() as session:
        return list((await session.execute(select(MemorySuppression))).scalars().all())


async def _served(memory_id: uuid.UUID) -> list[uuid.UUID]:
    """What a serving surface would return for this id, through the ONE rule."""
    async with database.AsyncSessionLocal() as session:
        return list((await session.execute(
            select(Memory.id).where(Memory.id == memory_id, not_dirty_predicate())
        )).scalars().all())


# ── the state: invalidated, provenance kept ─────────────────────────────────


async def test_soft_forget_invalidates_the_row_and_keeps_provenance_and_evidence(db):
    owner = await _owner(db)
    mid = uuid.uuid4()
    db.add(Memory(
        id=mid, user_id=owner, content="the fact", title="t", tags=[], source_type="manual_note",
        source_ref="note:42", salience=0.7,
        extra_metadata={CM_SUBJECT: "proj", "cm_attribute": "db",
                        CM_EVIDENCE_IDS: ["ev-1"], "cm_valid_from": "2026-01-01"},
    ))
    await db.commit()

    receipt = await soft_forget(db, owner, [mid], requested_by="agent:test")

    assert receipt.status == ERASURE_STATUS_COMPLETED
    assert receipt.detail["summary"]["invalidated"] == 1
    assert receipt.detail["summary"]["skipped"] == 0
    async with database.AsyncSessionLocal() as session:
        kept = await session.get(Memory, mid)
    assert kept is not None, "soft forget keeps the row: provenance IS the point"
    assert state_of(kept) == "invalidated"
    assert kept.extra_metadata[CM_SUBJECT] == "proj", "provenance survives"
    assert kept.extra_metadata[CM_EVIDENCE_IDS] == ["ev-1"], "evidence survives"
    assert kept.extra_metadata["cm_valid_from"] == "2026-01-01"
    assert (kept.content, kept.title) == ("the fact", "t"), "content is not rewritten"
    assert kept.source_ref == "note:42" and kept.source_type == "manual_note"
    assert await _served(mid) == [], "the row left serving"
    assert receipt.detail["serving_residual"] == []


async def test_soft_forget_records_a_foreign_or_missing_id_and_writes_nothing(db):
    owner = await _owner(db)
    other = await _owner(db)
    theirs = _memory(other, "theirs")
    db.add(theirs)
    await db.commit()

    receipt = await soft_forget(db, owner, [theirs.id, uuid.uuid4()], requested_by="agent:test")

    assert receipt.detail["summary"]["invalidated"] == 0
    assert receipt.detail["summary"]["skipped"] == 2
    assert {t["status"] for t in receipt.detail["targets"]} == {"not_found_or_foreign"}
    async with database.AsyncSessionLocal() as session:
        row = await session.get(Memory, theirs.id)
    assert state_of(row) == "current", "another owner's row is never touched"
    assert await _outbox_rows() == [] and await _suppressions() == []


# ── universal suppression: every affected source, not just the root ──────────


async def test_soft_forget_suppresses_every_affected_source_ref_through_the_closure(db):
    owner = await _owner(db)
    root = _memory(owner, "root", source_type="file_upload", source_ref="doc-root")
    child = _memory(owner, "child", source_type="file_upload", source_ref="doc-child",
                    parent_id=root.id)
    note = _memory(owner, "note", source_type="manual_note",
                   source_ref="https://example.invalid/x")
    db.add_all([root, child, note])
    await db.commit()

    await soft_forget(db, owner, [root.id], requested_by="agent:test")

    rows = await _suppressions()
    assert {row.source_ref for row in rows} == {"doc-root", "doc-child"}, (
        "every affected source is pinned — the old hard path suppressed only the root")
    assert all(row.user_id == owner for row in rows)
    assert all(row.namespace == "personal" for row in rows), (
        "R38: the ledger records the namespace it pinned")
    assert all(row.reason == "forgotten" for row in rows)
    assert all(row.content_hash is None for row in rows), (
        "R38: content hash is computed at upload time — never backfilled here")


async def test_a_manual_note_source_is_suppressed_too(db):
    """The suppression is per source identity, not per ``file_upload`` type.

    A URL/text reference the user forgot must not come back through a later
    re-import either; the universal rule is what T4's guards will read.
    """
    owner = await _owner(db)
    row = _memory(owner, "note", source_type="manual_note", source_ref="clipboard:1")
    db.add(row)
    await db.commit()

    await soft_forget(db, owner, [row.id], requested_by="agent:test")

    assert {r.source_ref for r in await _suppressions()} == {"clipboard:1"}


# ── R37: the payload refresh is a real, durable intent ───────────────────────


async def test_soft_forget_bumps_revision_and_enqueues_one_upsert_per_row(db):
    owner = await _owner(db)
    root = _memory(owner, "root")
    child = _memory(owner, "child", parent_id=root.id)
    db.add_all([root, child])
    await db.commit()

    await soft_forget(db, owner, [root.id], requested_by="agent:test")

    intents = [row for row in await _outbox_rows()
               if row.kind == KIND_MEMORY and row.operation == OPERATION_UPSERT]
    assert {row.entity_id for row in intents} == {root.id.hex, child.id.hex}, (
        "every invalidated row owes the index its new state")
    assert {row.revision for row in intents} == {2}, (
        "each intent carries the revision the invalidation bumped to (R37)")
    async with database.AsyncSessionLocal() as session:
        revisions = {m.id: m.revision for m in (await session.execute(
            select(Memory).where(Memory.id.in_([root.id, child.id])))).scalars()}
    assert revisions == {root.id: 2, child.id: 2}, "the bump is real, not only an intent"


# ── the receipt: completed ONLY after a serving-off readback ────────────────


async def test_a_serving_residual_is_never_reported_as_completed(db, monkeypatch):
    owner = await _owner(db)
    row = _memory(owner, "x")
    db.add(row)
    await db.commit()

    async def _still_serving(db_, memory_ids):
        return set(memory_ids)  # the readback found every row still served

    monkeypatch.setattr(erasure_service, "_serving_memory_ids", _still_serving)
    receipt = await soft_forget(db, owner, [row.id], requested_by="agent:test")

    assert receipt.status == ERASURE_STATUS_RESIDUAL
    assert receipt.detail["serving_residual"] == [str(row.id)]
    assert receipt.detail["summary"]["serving_residual"] == 1


async def test_an_unreadable_serving_check_reports_unverified(db, monkeypatch):
    owner = await _owner(db)
    row = _memory(owner, "x")
    db.add(row)
    await db.commit()

    async def _boom(db_, memory_ids):
        raise RuntimeError("serving readback failed")

    monkeypatch.setattr(erasure_service, "_serving_memory_ids", _boom)
    receipt = await soft_forget(db, owner, [row.id], requested_by="agent:test")

    assert receipt.status == ERASURE_STATUS_UNVERIFIED, (
        "an unreadable check is unknown, never completed (§5.4)")


async def test_a_truncated_closure_is_refused_before_any_write(db, monkeypatch):
    """A partial closure must not be half-forgotten: refuse, write nothing."""
    owner = await _owner(db)
    root = _memory(owner, "root")
    child = _memory(owner, "child", parent_id=root.id)
    db.add_all([root, child])
    await db.commit()

    monkeypatch.setattr(correction, "_MAX_CLOSURE_IDS", 1)  # root + child = 2
    receipt = await soft_forget(db, owner, [root.id], requested_by="agent:test")

    assert receipt.status == ERASURE_STATUS_ERRORS
    target = receipt.detail["targets"][0]
    assert target["status"] == "error" and target["truncated"] is True
    async with database.AsyncSessionLocal() as session:
        kept = await session.get(Memory, root.id)
    assert state_of(kept) == "current", "a refused closure invalidates nothing"
    assert await _suppressions() == [] and await _outbox_rows() == []


# ── the hard path is untouched — and still wins over an invalidated row ─────


async def test_hard_erase_still_erases_an_invalidated_row(db, monkeypatch):
    """Explicit delete beats the soft state: the row and its suppression go."""
    owner = await _owner(db)
    row = _memory(owner, "x", source_ref="doc-1")
    db.add(row)
    await db.commit()
    await soft_forget(db, owner, [row.id], requested_by="agent:test")
    assert await _served(row.id) == []

    async def _no_purge(_memory_id):
        return True

    async def _absent(_memory_ids):
        return set()

    monkeypatch.setattr("app.services.erasure_service.safe_delete_from_index", _no_purge)
    monkeypatch.setattr("app.services.erasure_service._vector_present_ids", _absent)
    receipt = await erase_memories(db, owner, [row.id], requested_by="rest_api")

    assert receipt.status == ERASURE_STATUS_COMPLETED
    assert receipt.detail["summary"]["erased"] == 1
    async with database.AsyncSessionLocal() as session:
        assert await session.get(Memory, row.id) is None
