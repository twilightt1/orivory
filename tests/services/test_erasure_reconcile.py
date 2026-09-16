"""P3 Task 4 — the memory delete readback and the receipt reconciliation.

Red-first (brief cases a-d):

- (a) a delete whose absence was never read back leaves the receipt
  ``completed_unverified`` — never ``completed``;
- (b) the drain lands the owed delete, and ONE reconcile pass upgrades the
  receipt to ``completed`` (exactly once);
- (c) a receipt that was positively verified is never downgraded — nor even
  scanned;
- (d) a receipt that recorded a residual keeps its ``*_with_residual`` status.

Fix round 1 adds the pins the review asked for: the outage branch that is the
ONLY thing between a Qdrant outage and a receipt stamped ``completed`` (I1),
the residual observation written into an open receipt's detail (R19/M7), FIFO
ordering + the scan window (M4/R20), a commit per receipt (M3) and a readback
face that honours the ids it asks about (M8).

The vector store is the only monkeypatched seam (the ``test_durable_erasure``
pattern): every DB assertion below runs against the real service, sessions and
outbox on this package's private temp SQLite file.
"""
from __future__ import annotations

import asyncio
import copy
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from app import database
from app.models.erasure_receipt import ErasureReceipt
from app.models.index_outbox import IndexOutbox
from app.models.memory import Memory
from app.models.user import User
from app.retrieval.memory import drain_loop, outbox, vector_store
from app.services import erasure_service
from app.services.erasure_service import erase_memories, reconcile_erasure_receipts


def _memory(user_id, content="x", **kwargs) -> Memory:
    return Memory(id=uuid.uuid4(), user_id=user_id, content=content, tags=[], **kwargs)


async def _owner(db) -> uuid.UUID:
    """A user row — ``memories.user_id`` is a FK."""
    uid = uuid.uuid4()
    db.add(User(id=uid, email=f"{uid.hex}@test.invalid", hashed_password="x",
                display_name="Owner", is_verified=True, is_active=True))
    await db.commit()
    return uid


async def _receipt(receipt_id: uuid.UUID) -> ErasureReceipt:
    """Read one receipt back through a FRESH session (no identity-map staleness)."""
    async with database.AsyncSessionLocal() as session:
        return await session.get(ErasureReceipt, receipt_id)


def _open_receipt(user_id: uuid.UUID, memory_id: uuid.UUID | None = None, *,
                  at: datetime) -> ErasureReceipt:
    """An open receipt in the shape the erase writes (for the ordering pins).

    ``created_at`` is passed explicitly: the column's server default would put
    every row in the same second and the FIFO pins need a total order.
    """
    memory_id = memory_id or uuid.uuid4()
    return ErasureReceipt(
        user_id=user_id,
        requested_memory_ids=[str(memory_id)],
        status="completed_unverified",
        detail={"requested_by": "rest_api", "targets": [{
            "memory_id": str(memory_id),
            "status": "deleted",
            "vectors_deleted": [str(memory_id)],
            "vector_state": "pending",
            "vector_residual": [],
            "vector_residual_checked": False,
            "db_residual": {"children": 0, "entity_links": 0, "source_links": 0,
                            "cross_user_children": 0, "derived_out_of_namespace": 0},
            "index_pending": 1,
        }]},
        created_at=at,
    )


async def _outbox_rows() -> list[IndexOutbox]:
    async with database.AsyncSessionLocal() as session:
        return list((await session.execute(
            select(IndexOutbox).order_by(IndexOutbox.seq))).scalars().all())


async def _statuses() -> list[str]:
    return [row.status for row in await _outbox_rows()]


async def _until(predicate, *, timeout: float = 1.0) -> None:
    """Poll ``predicate()`` (sync or async) until it holds, or fail the test."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        result = predicate()
        if await result if asyncio.iscoroutine(result) else result:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"condition never held within {timeout}s")


@pytest.fixture()
def store_down(monkeypatch):
    """Erase-time seams: the purge did not land and absence could not be read."""
    async def _purge_fails(_memory_id):
        return False

    async def _unreachable(_memory_ids):
        raise ConnectionError("qdrant down")

    monkeypatch.setattr(erasure_service, "safe_delete_from_index", _purge_fails)
    monkeypatch.setattr(erasure_service, "_vector_present_ids", _unreachable)


# ── (a)/(b) the readback is what makes a receipt `completed` ────────────────


async def test_a_delete_without_readback_is_never_a_completed_receipt(
    db, sessions, monkeypatch, store_down
):
    """(a) An unconfirmed memory delete leaves the receipt ``completed_unverified``.

    The durable outbox acks a delete off ``vector_store.delete_memory``: without
    a readback that ack is a lie, and the receipt is the proof the user reads.
    """
    uid = await _owner(db)
    mem = _memory(uid)
    db.add(mem)
    await db.commit()

    receipt = await erase_memories(db, uid, [mem.id], requested_by="rest_api")
    assert receipt.status == "completed_unverified"  # never `completed` without readback
    assert receipt.detail["targets"][0]["vector_state"] == "pending"

    # The vector is STILL there (the delete never landed / was never read back):
    # reconcile must not upgrade this receipt to `completed`.
    async def _still_present(_memory_ids):
        return {str(mem.id)}

    monkeypatch.setattr(erasure_service, "_vector_present_ids", _still_present)
    report = await reconcile_erasure_receipts()

    assert report == {"checked": 1, "upgraded": 0, "still_unverified": 1}
    assert (await _receipt(receipt.id)).status == "completed_unverified"


async def test_the_drain_lands_the_delete_then_reconcile_upgrades_once(
    db, sessions, monkeypatch, store_down
):
    """(b) The drain finishes the purge; one reconcile pass turns the receipt verified."""
    uid = await _owner(db)
    mem = _memory(uid)
    db.add(mem)
    await db.commit()

    receipt = await erase_memories(db, uid, [mem.id], requested_by="rest_api")
    assert receipt.status == "completed_unverified"  # the index still owes the purge
    assert receipt.detail["index_pending"] == 1
    assert await _statuses() == ["pending"]

    # The drain lands the owed delete (the readback lives inside the store call).
    async def _delete_ok(_memory_id):
        return True

    monkeypatch.setattr(outbox, "AsyncSessionLocal", sessions)
    monkeypatch.setattr(outbox, "delete_memory", _delete_ok)
    assert (await outbox.drain_pending())["applied"] == 1
    assert await _statuses() == ["done"]

    # The store now answers absence: the re-verification rewrites the SAME row.
    async def _absent(_memory_ids):
        return set()

    monkeypatch.setattr(erasure_service, "_vector_present_ids", _absent)
    report = await reconcile_erasure_receipts()

    assert report == {"checked": 1, "upgraded": 1, "still_unverified": 0}
    refreshed = await _receipt(receipt.id)
    assert refreshed.status == "completed"
    assert refreshed.detail["verification"] == "verified"
    assert refreshed.detail["index_pending"] == 0
    assert refreshed.detail["targets"][0]["vector_state"] == "verified"
    assert refreshed.detail["targets"][0]["db_residual"] == {
        "children": 0, "entity_links": 0, "source_links": 0, "cross_user_children": 0,
        "derived_out_of_namespace": 0}

    # Exactly once: the upgraded receipt is terminal, so the next pass skips it.
    assert await reconcile_erasure_receipts() == {
        "checked": 0, "upgraded": 0, "still_unverified": 0}
    assert (await _receipt(receipt.id)).status == "completed"


async def test_a_receipt_that_left_a_derivative_behind_never_upgrades(
    db, monkeypatch, store_down
):
    """I3: a derivative another namespace holds is data that OUTLIVED the erase,
    so the residual is PRESERVED through a clean readback — never re-counted
    away into a ``completed``.

    The residual is written directly: a recorded residual makes the erase's own
    verdict ``completed_with_residual`` (terminal, never scanned), so the gate is
    pinned on the state reconcile would face if such a receipt were ever open.
    """
    uid = await _owner(db)
    mem = _memory(uid)
    db.add(mem)
    await db.commit()

    receipt = await erase_memories(db, uid, [mem.id], requested_by="rest_api")
    assert receipt.status == "completed_unverified"  # open: the readback never ran

    async with database.AsyncSessionLocal() as session:
        row = await session.get(ErasureReceipt, receipt.id)
        assert row is not None
        targets = copy.deepcopy(row.detail["targets"])
        targets[0]["db_residual"]["derived_out_of_namespace"] = 1
        row.detail = {**row.detail, "targets": targets}
        await session.commit()

    async def _absent(_memory_ids):
        return set()

    monkeypatch.setattr(erasure_service, "_vector_present_ids", _absent)
    report = await reconcile_erasure_receipts()

    assert report == {"checked": 1, "upgraded": 0, "still_unverified": 1}
    refreshed = await _receipt(receipt.id)
    assert refreshed.status == "completed_unverified"
    assert refreshed.detail["targets"][0]["db_residual"]["derived_out_of_namespace"] == 1


# ── (c)/(d) terminal receipts are never revised ─────────────────────────────


async def test_reconcile_never_downgrades_a_verified_receipt(db, sessions, monkeypatch):
    """(c) A verified receipt is terminal: later bad news never rewrites it."""
    async def _purge_ok(_memory_id):
        return True

    async def _absent(_memory_ids):
        return set()

    monkeypatch.setattr(erasure_service, "safe_delete_from_index", _purge_ok)
    monkeypatch.setattr(erasure_service, "_vector_present_ids", _absent)

    uid = await _owner(db)
    mem = _memory(uid)
    db.add(mem)
    await db.commit()

    verified = await erase_memories(db, uid, [mem.id], requested_by="rest_api")
    assert verified.status == "completed"
    before = copy.deepcopy(verified.detail)

    # The store now answers the opposite (a stale replica, a late write)…
    async def _present(_memory_ids):
        return {str(mem.id)}

    monkeypatch.setattr(erasure_service, "_vector_present_ids", _present)
    report = await reconcile_erasure_receipts()

    assert report["checked"] == 0  # a terminal receipt is not even scanned
    again = await _receipt(verified.id)
    assert again.status == "completed"
    assert again.detail == before  # not one byte of the evidence rewritten


async def test_a_receipt_with_residual_keeps_its_residual_status(
    db, sessions, monkeypatch, store_down
):
    """(d) A receipt that recorded a residual keeps ``*_with_residual``."""
    uid = await _owner(db)
    mem = _memory(uid)
    db.add(mem)
    await db.commit()

    # The erase itself finds the vector still present: residual, not unverified.
    async def _present(_memory_ids):
        return {str(mem.id)}

    monkeypatch.setattr(erasure_service, "_vector_present_ids", _present)
    residual = await erase_memories(db, uid, [mem.id], requested_by="rest_api")
    assert residual.status == "completed_with_residual"

    # Even once the store reads clean, the residual verdict is not revised.
    async def _absent(_memory_ids):
        return set()

    monkeypatch.setattr(erasure_service, "_vector_present_ids", _absent)
    report = await reconcile_erasure_receipts()

    assert report["checked"] == 0
    assert (await _receipt(residual.id)).status == "completed_with_residual"


# ── fix round 1: the safety branch, the evidence, the window, the commits ───


async def test_an_unreadable_store_at_reconcile_time_leaves_the_receipt_alone(
    db, sessions, monkeypatch, store_down
):
    """I1/R19: the outage branch is the only thing between an outage and `completed`.

    The erase-time readback already failed, the drain has since landed the owed
    delete, and the store is unreachable AGAIN when reconcile asks: nothing was
    observed, so nothing is written — and the receipt stays open, so the next
    pass that can read still upgrades it.
    """
    uid = await _owner(db)
    mem = _memory(uid)
    db.add(mem)
    await db.commit()

    receipt = await erase_memories(db, uid, [mem.id], requested_by="rest_api")
    before = copy.deepcopy(receipt.detail)

    async def _unreachable(_memory_ids):
        raise ConnectionError("qdrant down")

    monkeypatch.setattr(erasure_service, "_vector_present_ids", _unreachable)
    assert await reconcile_erasure_receipts() == {
        "checked": 1, "upgraded": 0, "still_unverified": 1}

    refused = await _receipt(receipt.id)
    assert refused.status == "completed_unverified"  # never a completion on a failed read
    assert refused.detail == before  # not one byte written by an unobserved pass

    # Still open and re-checkable: the pass that CAN read upgrades it (R19).
    async def _absent(_memory_ids):
        return set()

    monkeypatch.setattr(erasure_service, "_vector_present_ids", _absent)
    assert await reconcile_erasure_receipts() == {
        "checked": 1, "upgraded": 1, "still_unverified": 0}
    assert (await _receipt(receipt.id)).status == "completed"


async def test_a_residual_read_is_recorded_as_evidence_not_as_a_status(
    db, sessions, monkeypatch, store_down
):
    """R19/M7: a residual found at reconcile time goes into `detail`, nothing else.

    Relabelling to the terminal `completed_with_residual` would freeze a row
    that is still re-checkable, so the status is left alone on purpose — the
    observation is written where a later reader can see it instead.
    """
    uid = await _owner(db)
    mem = _memory(uid)
    db.add(mem)
    await db.commit()

    receipt = await erase_memories(db, uid, [mem.id], requested_by="rest_api")
    before = copy.deepcopy(receipt.detail)

    async def _present(_memory_ids):
        return {str(mem.id)}

    monkeypatch.setattr(erasure_service, "_vector_present_ids", _present)
    assert await reconcile_erasure_receipts() == {
        "checked": 1, "upgraded": 0, "still_unverified": 1}

    refused = await _receipt(receipt.id)
    assert refused.status == "completed_unverified"  # no terminal relabel

    expected = copy.deepcopy(before)
    expected["targets"][0]["vector_residual_checked"] = True
    expected["targets"][0]["vector_residual"] = [str(mem.id)]
    assert refused.detail == expected  # exactly those two keys; nothing else moved

    # The receipt stayed open, so a later clean read still completes it.
    async def _absent(_memory_ids):
        return set()

    monkeypatch.setattr(erasure_service, "_vector_present_ids", _absent)
    assert (await reconcile_erasure_receipts())["upgraded"] == 1
    assert (await _receipt(receipt.id)).status == "completed"


async def test_an_unparseable_detail_is_scanned_and_refused_untouched(
    db, sessions, monkeypatch
):
    """M7: a receipt whose recorded evidence is not UUIDs is never guessed at."""
    async def _absent(_memory_ids):
        return set()

    monkeypatch.setattr(erasure_service, "_vector_present_ids", _absent)
    uid = await _owner(db)
    receipt = _open_receipt(uid, at=datetime.now(UTC))
    receipt.detail["targets"][0]["vectors_deleted"] = ["not-a-uuid"]
    db.add(receipt)
    await db.commit()
    before = copy.deepcopy(receipt.detail)

    assert await reconcile_erasure_receipts() == {
        "checked": 1, "upgraded": 0, "still_unverified": 1}

    refused = await _receipt(receipt.id)
    assert refused.status == "completed_unverified"
    assert refused.detail == before


async def test_reconcile_scans_oldest_first_and_clamps_the_window(
    db, sessions, monkeypatch
):
    """M4/R20: FIFO ordering, and the window is `max(1, min(limit, 200))`."""
    seen: list[str] = []

    async def _absent(memory_ids):
        seen.extend(memory_ids)
        return set()

    monkeypatch.setattr(erasure_service, "_vector_present_ids", _absent)
    uid = await _owner(db)
    stamp = datetime.now(UTC) - timedelta(hours=1)
    receipts = []
    for i in range(3):
        receipt = _open_receipt(uid, at=stamp + timedelta(minutes=i))
        db.add(receipt)
        receipts.append(receipt)
    await db.commit()
    oldest_first = [r.detail["targets"][0]["memory_id"] for r in receipts]

    # limit=1: the OLDEST is scanned — newest-first would have taken the last.
    assert await reconcile_erasure_receipts(limit=1) == {
        "checked": 1, "upgraded": 1, "still_unverified": 0}
    assert seen == oldest_first[:1]

    # The knob cannot exceed _MAX_LIMIT (200 — the documented endpoint maximum)…
    assert erasure_service._MAX_LIMIT == 200
    monkeypatch.setattr(erasure_service, "_MAX_LIMIT", 1)
    assert await reconcile_erasure_receipts(limit=10 ** 6) == {
        "checked": 1, "upgraded": 1, "still_unverified": 0}
    assert (await _receipt(receipts[2].id)).status == "completed_unverified"  # windowed, not dropped
    monkeypatch.setattr(erasure_service, "_MAX_LIMIT", 200)

    # …and 0 floors to 1: a bad knob never skips an open receipt.
    assert await reconcile_erasure_receipts(limit=0) == {
        "checked": 1, "upgraded": 1, "still_unverified": 0}
    assert seen == oldest_first  # FIFO across every pass


async def test_reconcile_breaks_created_at_ties_by_id(db, sessions, monkeypatch):
    """M4/R20: same-second receipts have a total order — none is unreachable."""
    seen: list[str] = []

    async def _absent(memory_ids):
        seen.extend(memory_ids)
        return set()

    monkeypatch.setattr(erasure_service, "_vector_present_ids", _absent)
    uid = await _owner(db)
    stamp = datetime.now(UTC)
    # Two receipts stamped the same second: their OWN ids decide the order.
    receipt_ids = sorted((uuid.uuid4(), uuid.uuid4()), key=lambda value: value.hex)
    first, second = (_open_receipt(uid, at=stamp) for _ in range(2))
    first.id, second.id = receipt_ids
    db.add_all([first, second])
    await db.commit()

    assert await reconcile_erasure_receipts(limit=1) == {
        "checked": 1, "upgraded": 1, "still_unverified": 0}
    assert seen == [first.detail["targets"][0]["memory_id"]]  # the lower id wins


async def test_reconcile_commits_each_receipt_as_it_goes(db, sessions, monkeypatch, store_down):
    """M3: per-receipt commits (the drain's shape), not one transaction per pass.

    Two open receipts: while the SECOND one is being processed the first must
    already be durable. A pass-long transaction is what a concurrent writer
    waits out (5s busy timeout, then `database is locked`), and it is what a
    single commit at the end would lose wholesale.
    """
    uid = await _owner(db)
    receipts = []
    for _ in range(2):
        mem = _memory(uid)
        db.add(mem)
        await db.commit()
        receipts.append(await erase_memories(db, uid, [mem.id], requested_by="rest_api"))
    assert [r.status for r in receipts] == ["completed_unverified"] * 2

    async def _absent(_memory_ids):
        return set()

    monkeypatch.setattr(erasure_service, "_vector_present_ids", _absent)
    real = erasure_service._upgrade_receipt
    processed: list[uuid.UUID] = []
    committed_mid_pass: list[str] = []

    async def _upgrade_watching_the_previous_receipt(session, receipt):
        processed.append(receipt.id)
        if len(processed) == 2:
            async with sessions() as other:  # the drain writes mid-pass
                other.add(ErasureReceipt(user_id=uid, status="completed_unverified",
                                         detail={"targets": []}))
                await other.commit()
                # …and a reader already sees what THIS pass committed:
                committed_mid_pass.append(
                    (await other.get(ErasureReceipt, processed[0])).status)
        return await real(session, receipt)

    monkeypatch.setattr(erasure_service, "_upgrade_receipt",
                        _upgrade_watching_the_previous_receipt)

    assert await reconcile_erasure_receipts() == {
        "checked": 2, "upgraded": 2, "still_unverified": 0}
    assert committed_mid_pass == ["completed"]


async def test_a_failed_receipt_cannot_take_the_earlier_upgrades_with_it(
    db, sessions, monkeypatch, store_down
):
    """M3: each upgrade is committed as it lands — a later failure is not a lost pass."""
    uid = await _owner(db)
    receipts = []
    for _ in range(2):
        mem = _memory(uid)
        db.add(mem)
        await db.commit()
        receipts.append(await erase_memories(db, uid, [mem.id], requested_by="rest_api"))
    assert [r.status for r in receipts] == ["completed_unverified"] * 2

    async def _absent(_memory_ids):
        return set()

    monkeypatch.setattr(erasure_service, "_vector_present_ids", _absent)
    real = erasure_service._upgrade_receipt
    processed: list[uuid.UUID] = []

    async def _upgrade_boom_on_the_second(session, receipt):
        processed.append(receipt.id)
        if len(processed) == 2:
            raise RuntimeError("the second receipt's scan blew up")
        return await real(session, receipt)

    monkeypatch.setattr(erasure_service, "_upgrade_receipt", _upgrade_boom_on_the_second)

    with pytest.raises(RuntimeError):
        await reconcile_erasure_receipts()
    assert len(processed) == 2

    assert (await _receipt(processed[0])).status == "completed"  # durable, not rolled back
    assert (await _receipt(processed[1])).status == "completed_unverified"


# ── R17: the memory delete confirms absence by readback ─────────────────────


async def test_delete_confirms_absence_by_readback(monkeypatch):
    """R17: `delete_memory`/`delete_memories` are `True` only after the readback."""
    points: list = []

    class _Client:
        async def delete(self, **_kwargs):
            return None

        async def retrieve(self, *, ids, **_kwargs):
            return [point for point in points if str(point.id) in set(ids)]

    async def _open(_dim):
        return _Client(), "generation", None

    monkeypatch.setattr(vector_store, "_open_collection", _open)

    assert await vector_store.delete_memory("mem-1") is True  # absence confirmed
    assert await vector_store.delete_memories(["mem-1", "mem-2"]) is True

    points.append(SimpleNamespace(id="mem-1"))  # the point survived the delete
    assert await vector_store.delete_memory("mem-1") is False
    assert await vector_store.delete_memories(["mem-1", "mem-2"]) is False

    points[:] = [SimpleNamespace(id="mem-2")]  # a survivor that is NOT the first id
    assert await vector_store.delete_memories(["mem-1", "mem-2"]) is False
    assert await vector_store.delete_memory("mem-1") is True  # this one really is gone


async def test_a_delete_the_store_cannot_read_back_is_never_true(monkeypatch):
    """R17/I1: no answer from the store is not an absence — unknown ≠ verified."""

    class _Client:
        async def delete(self, **_kwargs):
            return None

        async def retrieve(self, **_kwargs):
            raise ConnectionError("qdrant down")  # the readback never answered

    async def _open(_dim):
        return _Client(), "generation", None

    monkeypatch.setattr(vector_store, "_open_collection", _open)

    assert await vector_store.delete_memory("mem-1") is False
    assert await vector_store.delete_memories(["mem-1", "mem-2"]) is False


# ── R15: reconcile runs on the loop only, after a productive round ──────────


def _report(applied: int) -> dict:
    return {"claimed": applied, "applied": applied, "skipped": 0, "blocked": 0, "failed": 0}


async def test_the_loop_reconciles_after_a_round_that_applied_work(monkeypatch):
    """R15: `applied>0` → reconcile; an idle round pays nothing."""
    reports = [_report(1), _report(0)]
    drains: list[int] = []
    reconciled: list[dict] = []

    async def fake_drain(*, batch_size):
        drains.append(batch_size)
        return reports.pop(0) if reports else _report(0)

    async def fake_reconcile(**kwargs):
        reconciled.append(kwargs)
        return {"checked": 2, "upgraded": 1, "still_unverified": 1}

    monkeypatch.setattr(drain_loop, "drain_pending", fake_drain)
    monkeypatch.setattr(drain_loop, "reconcile_erasure_receipts", fake_reconcile)

    stop = asyncio.Event()
    task = asyncio.create_task(
        drain_loop.run_drain_loop(interval=30.0, batch_size=3, stop=stop))
    try:
        await _until(lambda: len(drains) >= 2 and reconciled)
    finally:
        stop.set()
        await asyncio.wait_for(task, timeout=5)

    assert drains == [3, 3]
    assert reconciled == [{}]  # once, after the productive round — not after the idle one


async def test_drain_once_never_pays_for_reconcile(monkeypatch):
    """R15: the request-path barrier drains through `drain_once` — no reconcile there."""
    reconciled: list[int] = []

    async def fake_drain(*, batch_size):
        return _report(1)

    async def fake_reconcile(**kwargs):
        reconciled.append(1)

    monkeypatch.setattr(drain_loop, "drain_pending", fake_drain)
    monkeypatch.setattr(drain_loop, "reconcile_erasure_receipts", fake_reconcile)

    assert (await drain_loop.drain_once(batch_size=5))["applied"] == 1
    assert reconciled == []


async def test_a_reconcile_failure_is_logged_and_the_loop_keeps_going(monkeypatch):
    """A reconcile failure must not stop the drain loop (`drain_loop` never raises)."""
    drains: list[int] = []

    async def fake_drain(*, batch_size):
        drains.append(batch_size)
        await asyncio.sleep(0)  # yield: a real drain awaits its DB round-trips
        return _report(1)

    async def boom(**kwargs):
        raise RuntimeError("receipts table is gone")

    captured = []

    class _Log:
        def info(self, event, **kw):
            captured.append(("info", event))

        def warning(self, event, **kw):
            captured.append(("warning", event))

    monkeypatch.setattr(drain_loop, "drain_pending", fake_drain)
    monkeypatch.setattr(drain_loop, "reconcile_erasure_receipts", boom)
    monkeypatch.setattr(drain_loop, "log", _Log())

    stop = asyncio.Event()
    task = asyncio.create_task(
        drain_loop.run_drain_loop(interval=30.0, batch_size=1, stop=stop))
    try:
        await _until(lambda: len(drains) >= 3)  # it kept draining after the failure
        assert not task.done()
    finally:
        stop.set()
        await asyncio.wait_for(task, timeout=5)

    assert ("warning", "erasure receipt reconcile failed") in captured


# ── R18: the admin endpoint is the same dict, admin-only ────────────────────


async def test_the_admin_endpoint_returns_the_function_dict(monkeypatch):
    """R18: `POST /admin/erasure/reconcile` follows the `/memories/reindex` shape."""
    from app.api.v1 import admin

    calls: list[dict] = []

    async def fake_reconcile(**kwargs):
        calls.append(kwargs)
        return {"checked": 3, "upgraded": 2, "still_unverified": 1}

    monkeypatch.setattr(erasure_service, "reconcile_erasure_receipts", fake_reconcile)

    route = next(r for r in admin.router.routes
                 if getattr(r, "path", None) == "/admin/erasure/reconcile")
    assert "POST" in route.methods
    # require_admin is wired as a dependency, never left to annotation resolution.
    assert any(d.call.__name__ == "require_admin" for d in route.dependant.dependencies)

    body = await route.endpoint(admin_user=SimpleNamespace(id=uuid.uuid4()))
    assert body.model_dump() == {"checked": 3, "upgraded": 2, "still_unverified": 1}
    assert calls == [{}]  # the endpoint passes no extra knobs
