"""Opt-in retention (P4b Task 6): expire by age, only for users who asked.

Real SQLite files (fixtures in this package's conftest) — every assertion below
reads the real tables. The vector store is never touched: the expiry is a SOFT
invalidate (the same state the forget path uses), so the point keeps existing
while the payload state refreshes by INTENT (``bump_revision`` +
``enqueue_upsert``, R37) and serving is closed by the state itself (T1).

Rulings this module pins:

- Default OFF, system-wide (spec §8.1): a user without the setting is never
  scanned — not a slow sweep, no sweep at all. The ladder's ADD COLUMN default
  makes every pre-existing user OFF.
- Pin protects ONLY against auto-retention (global constraint): an expired
  pinned row survives this sweep, and explicit forget/erase still win over it.
- The expiry is an audit trail: soft invalidate + reason ``retention_expired``
  + one append-only ``MemoryAccessLog`` row per expired memory.
- NO suppression row: retention is automatic expiry, not the user forgetting a
  source — a suppression would block a re-import the user never asked to
  forget (that ledger belongs to explicit forget, T3/T4).
- The window counts ``indexed_at`` — how long the system has HELD the memory —
  not ``captured_at`` (a 2019 email imported yesterday was not held here for
  2019 days).
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from app import database
from app.api.v1 import users as users_api
from app.models.index_outbox import IndexOutbox
from app.models.memory import Memory, MemorySuppression
from app.models.memory_access_log import MemoryAccessLog
from app.models.user import User
from app.retrieval.memory import drain_loop
from app.retrieval.memory.correction import state_of
from app.retrieval.memory.outbox import KIND_MEMORY, OPERATION_UPSERT
from app.retrieval.memory.visibility import not_dirty_predicate
from app.schemas.auth import RetentionSettingsRequest, UserResponse
from app.services import erasure_service
from app.services.erasure_service import erase_memories, soft_forget
from app.services.retention_service import RETENTION_REASON, RetentionReport, run_retention

NOW = datetime(2026, 9, 17, 12, 0, 0, tzinfo=UTC)


async def _owner(db, *, enabled: bool = False, days: int | None = None) -> User:
    """A real user row. Retention is left at its schema default unless asked."""
    user = User(id=uuid.uuid4(), email=f"{uuid.uuid4().hex}@test.invalid",
                hashed_password="x", display_name="Owner", is_verified=True,
                is_active=True, retention_enabled=enabled, retention_days=days)
    db.add(user)
    await db.commit()
    return user


def _memory(user_id, content: str = "x", *, age_days: int = 0, **kwargs) -> Memory:
    """A memory the system has HELD for ``age_days`` (the retention clock)."""
    return Memory(id=uuid.uuid4(), user_id=user_id, content=content, tags=[],
                  indexed_at=NOW - timedelta(days=age_days), **kwargs)


async def _state(memory_id) -> str:
    async with database.AsyncSessionLocal() as session:
        return state_of(await session.get(Memory, memory_id))


async def _revision(memory_id) -> int:
    async with database.AsyncSessionLocal() as session:
        return int((await session.get(Memory, memory_id)).revision)


async def _audit_rows() -> list[MemoryAccessLog]:
    async with database.AsyncSessionLocal() as session:
        return list((await session.execute(
            select(MemoryAccessLog).order_by(MemoryAccessLog.memory_id))).scalars().all())


async def _upsert_intents() -> list[IndexOutbox]:
    async with database.AsyncSessionLocal() as session:
        return [row for row in (await session.execute(select(IndexOutbox))).scalars()
                if row.kind == KIND_MEMORY and row.operation == OPERATION_UPSERT]


async def _served(memory_id) -> list[uuid.UUID]:
    """What a serving surface would return for this id, through the ONE rule."""
    async with database.AsyncSessionLocal() as session:
        return list((await session.execute(
            select(Memory.id).where(Memory.id == memory_id, not_dirty_predicate()))).scalars().all())


# ── default OFF: no setting, no sweep ───────────────────────────────────────


async def test_default_off_a_user_without_the_setting_is_never_scanned(db):
    owner = await _owner(db)  # retention_enabled left at the schema default
    ancient = _memory(owner.id, "from a previous era", age_days=400)
    windowed = await _owner(db, enabled=False, days=30)  # a window, but switched OFF
    also_ancient = _memory(windowed.id, "old but opted out", age_days=400)
    db.add_all([ancient, also_ancient])
    await db.commit()

    report = await run_retention(db, NOW)

    assert (report.users, report.invalidated) == (0, 0), (
        "the sweep is opt-in: a user who never enabled it is not scanned at all")
    assert await _state(ancient.id) == "current"
    assert await _state(also_ancient.id) == "current", (
        "a window alone expires nothing — only the pair enables the sweep")
    assert await _audit_rows() == [] and await _upsert_intents() == []


async def test_the_schema_default_is_off_for_every_user_row(db):
    """Default OFF, system-wide (spec §8.1) — and the ladder never backfills it ON."""
    owner = await _owner(db)
    async with database.AsyncSessionLocal() as session:
        read_back = await session.get(User, owner.id)
    assert read_back.retention_enabled is False
    assert read_back.retention_days is None


async def test_an_enabled_user_without_a_window_is_not_run(db):
    """Enabled but no window says nothing — "no setting → không chạy"."""
    owner = await _owner(db, enabled=True, days=None)
    ancient = _memory(owner.id, "no window", age_days=400)
    db.add(ancient)
    await db.commit()

    report = await run_retention(db, NOW)

    assert (report.users, report.invalidated) == (0, 0)
    assert await _state(ancient.id) == "current"


# ── opt-in: age > window expires; pin exempts ───────────────────────────────


async def test_an_enabled_user_expires_rows_older_than_the_window(db):
    owner = await _owner(db, enabled=True, days=30)
    expired = _memory(owner.id, "old", age_days=45)
    boundary = _memory(owner.id, "exactly the window", age_days=30)
    fresh = _memory(owner.id, "recent", age_days=3)
    pinned = _memory(owner.id, "pinned and old", age_days=45, pinned=True)
    db.add_all([expired, boundary, fresh, pinned])
    await db.commit()

    report = await run_retention(db, NOW)

    assert (report.users, report.invalidated) == (1, 1)
    assert await _state(expired.id) == "invalidated"
    assert await _state(boundary.id) == "current", (
        "age > window is STRICT: a row exactly at the window stays")
    assert await _state(fresh.id) == "current"
    assert await _state(pinned.id) == "current", "pin protects against auto-retention"
    assert await _revision(pinned.id) == 1 and await _revision(fresh.id) == 1, (
        "an untouched row owes the index nothing")
    assert await _served(expired.id) == [], "the invalidated row left serving"
    assert await _served(boundary.id) == [boundary.id]


async def test_the_expiry_keeps_the_row_and_is_audited_with_the_reason(db):
    owner = await _owner(db, enabled=True, days=30)
    row = _memory(owner.id, "audited", age_days=45, source_ref="note:7")
    db.add(row)
    await db.commit()

    await run_retention(db, NOW)

    assert await _state(row.id) == "invalidated"
    assert await _revision(row.id) == 2, "R37: the state must reach the payload (a real write)"
    assert [intent.entity_id for intent in await _upsert_intents()] == [row.id.hex]

    (log_row,) = await _audit_rows()
    assert log_row.action == RETENTION_REASON == "retention_expired"
    assert log_row.memory_id == row.id and log_row.user_id == owner.id
    assert log_row.detail["reason"] == RETENTION_REASON
    assert log_row.detail["retention_days"] == 30
    assert log_row.agent_client_id is None, "a background sweep: no agent client"
    assert log_row.detail["indexed_at"], "the audit names the clock it expired on"

    async with database.AsyncSessionLocal() as session:
        kept = await session.get(Memory, row.id)
        suppressions = list((await session.execute(select(MemorySuppression))).scalars().all())
    assert kept is not None and kept.content == "audited", (
        "soft invalidate: content and provenance are kept")
    assert suppressions == [], (
        "retention is automatic expiry, not a user forget — it must not suppress a re-import")


async def test_a_second_sweep_is_a_no_op(db):
    """Idempotent by construction: an invalidated row is never expired again."""
    owner = await _owner(db, enabled=True, days=30)
    row = _memory(owner.id, "expire me once", age_days=45)
    db.add(row)
    await db.commit()

    first = await run_retention(db, NOW)
    second = await run_retention(db, NOW)

    assert (first.invalidated, second.invalidated) == (1, 0)
    assert await _revision(row.id) == 2, "the second sweep owes the index nothing"
    assert len(await _audit_rows()) == 1, "one audit row per memory, never one per sweep"
    assert len(await _upsert_intents()) == 1


async def test_another_users_rows_are_never_touched(db):
    enabled = await _owner(db, enabled=True, days=30)
    other = await _owner(db)  # never enabled
    mine = _memory(enabled.id, "mine", age_days=45)
    theirs = _memory(other.id, "theirs", age_days=45)
    db.add_all([mine, theirs])
    await db.commit()

    report = await run_retention(db, NOW)

    assert report.invalidated == 1
    assert await _state(theirs.id) == "current", "one user's window is not another's"


# ── explicit forget/erase still win over pin and over the expired state ─────


async def test_explicit_erase_still_deletes_a_retention_invalidated_row(db, monkeypatch):
    """The expired row is history, not a tombstone: the user's erase still hard-deletes."""
    deleted: list[str] = []

    async def _confirmed_delete(memory_id):
        deleted.append(str(memory_id))
        return True

    async def _no_residual(_ids):
        return set()

    monkeypatch.setattr(erasure_service, "safe_delete_from_index", _confirmed_delete)
    monkeypatch.setattr(erasure_service, "_vector_present_ids", _no_residual)

    owner = await _owner(db, enabled=True, days=30)
    row = _memory(owner.id, "expired then erased", age_days=45)
    db.add(row)
    await db.commit()
    await run_retention(db, NOW)
    assert await _state(row.id) == "invalidated"

    receipt = await erase_memories(db, owner.id, [row.id], requested_by="rest_api")

    assert receipt.status == "completed"
    async with database.AsyncSessionLocal() as session:
        assert await session.get(Memory, row.id) is None, (
            "explicit erase wins over the retention-invalidated state")
    assert deleted == [str(row.id)]


async def test_explicit_soft_forget_still_wins_over_a_pin(db):
    """Pin protects only against retention; the user's own forget still applies."""
    owner = await _owner(db, enabled=True, days=30)
    pinned = _memory(owner.id, "pinned old", age_days=45, pinned=True, source_ref="note:1")
    db.add(pinned)
    await db.commit()

    assert (await run_retention(db, NOW)).invalidated == 0, "the pin held"

    receipt = await soft_forget(db, owner.id, [pinned.id], requested_by="agent:test")

    assert receipt.detail["summary"]["invalidated"] == 1
    assert await _state(pinned.id) == "invalidated", (
        "explicit forget is not blocked by the pin")
    async with database.AsyncSessionLocal() as session:
        suppressions = list((await session.execute(select(MemorySuppression))).scalars().all())
    assert {row.source_ref for row in suppressions} == {"note:1"}


# ── the settings endpoint ───────────────────────────────────────────────────


async def test_the_settings_endpoint_writes_and_reads_the_retention_setting(db):
    user = await _owner(db)

    response = await users_api.update_retention_settings(
        RetentionSettingsRequest(retention_enabled=True, retention_days=30),
        current_user=user, db=db)

    assert (response.retention_enabled, response.retention_days) == (True, 30)
    async with database.AsyncSessionLocal() as session:
        stored = await session.get(User, user.id)
    assert (stored.retention_enabled, stored.retention_days) == (True, 30)
    # The read path: GET /me returns it through UserResponse.
    read_back = UserResponse.model_validate(stored)
    assert (read_back.retention_enabled, read_back.retention_days) == (True, 30)

    # ... and turning it back off is a plain write.
    off = await users_api.update_retention_settings(
        RetentionSettingsRequest(retention_enabled=False, retention_days=None),
        current_user=user, db=db)
    assert (off.retention_enabled, off.retention_days) == (False, None)


async def test_the_settings_endpoint_defaults_to_off_and_its_route_is_mounted(db):
    owner = await _owner(db)

    read_back = UserResponse.model_validate(owner)
    assert read_back.retention_enabled is False and read_back.retention_days is None, (
        "default OFF for a user who never touched the setting")
    assert "/users/me/settings" in {route.path for route in users_api.router.routes}


def test_enabling_retention_without_a_window_is_refused():
    with pytest.raises(ValidationError):
        RetentionSettingsRequest(retention_enabled=True, retention_days=None)
    with pytest.raises(ValidationError):
        RetentionSettingsRequest(retention_enabled=True, retention_days=0)
    # A window is fine on its own (disabled, remembered for later).
    assert RetentionSettingsRequest(retention_enabled=False, retention_days=90).retention_days == 90


# ── the drain loop's idle hook ──────────────────────────────────────────────


async def test_the_loop_runs_retention_on_idle_ticks_only(monkeypatch):
    """Retention rides IDLE ticks only (T6): it is a clock, not evidence.

    Unlike the consolidation producer (R39: after applied>0 AND on idle), a
    landed batch changes nothing for the sweep — so the hook must not be paid
    for on every busy round.
    """
    markers: list[str] = []
    state = {"rounds": 0, "landed": False}

    async def _drain(*, batch_size):
        state["rounds"] += 1
        applied = 1 if state["rounds"] <= 2 else 0
        state["landed"] = bool(applied)
        return {"applied": applied}

    async def _noop():
        return None

    async def _run(db_, now=None):
        markers.append("applied" if state["landed"] else "idle")
        return RetentionReport(users=0, invalidated=0)

    monkeypatch.setattr(drain_loop, "drain_once", _drain)
    monkeypatch.setattr(drain_loop, "_reconcile_after_drain", _noop)
    monkeypatch.setattr(drain_loop, "_consolidate_after_drain", _noop)
    monkeypatch.setattr(drain_loop, "run_retention", _run)

    stop = asyncio.Event()
    task = asyncio.create_task(
        drain_loop.run_drain_loop(interval=0.02, batch_size=5, stop=stop))
    deadline = asyncio.get_running_loop().time() + 5.0
    while len(markers) < 3:
        assert asyncio.get_running_loop().time() < deadline, "the hook never ran"
        await asyncio.sleep(0.01)
    stop.set()
    await asyncio.wait_for(task, timeout=5.0)

    assert markers[0] == "idle", "the hook runs on an idle tick"
    assert "applied" not in markers, (
        "a landed round must not pay for the sweep — idle only")
