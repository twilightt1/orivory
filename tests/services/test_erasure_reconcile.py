"""P3 Task 4 — the memory delete readback and the receipt reconciliation.

Red-first (brief cases a-d):

- (a) a delete whose absence was never read back leaves the receipt
  ``completed_unverified`` — never ``completed``;
- (b) the drain lands the owed delete, and ONE reconcile pass upgrades the
  receipt to ``completed`` (exactly once);
- (c) a receipt that was positively verified is never downgraded — nor even
  scanned;
- (d) a receipt that recorded a residual keeps its ``*_with_residual`` status.

The vector store is the only monkeypatched seam (the ``test_durable_erasure``
pattern): every DB assertion below runs against the real service, sessions and
outbox on this package's private temp SQLite file.
"""
from __future__ import annotations

import asyncio
import copy
import uuid
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
        "children": 0, "entity_links": 0, "source_links": 0, "cross_user_children": 0}

    # Exactly once: the upgraded receipt is terminal, so the next pass skips it.
    assert await reconcile_erasure_receipts() == {
        "checked": 0, "upgraded": 0, "still_unverified": 0}
    assert (await _receipt(receipt.id)).status == "completed"


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


# ── R17: the memory delete confirms absence by readback ─────────────────────


async def test_delete_confirms_absence_by_readback(monkeypatch):
    """R17: `delete_memory`/`delete_memories` are `True` only after the readback."""
    points: list = []

    class _Client:
        async def delete(self, **_kwargs):
            return None

        async def retrieve(self, **_kwargs):
            return list(points)  # what survives the delete

    async def _open(_dim):
        return _Client(), "generation", None

    monkeypatch.setattr(vector_store, "_open_collection", _open)

    assert await vector_store.delete_memory("mem-1") is True  # absence confirmed
    assert await vector_store.delete_memories(["mem-1", "mem-2"]) is True

    points.append(SimpleNamespace(id="mem-1"))  # the point survived the delete
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
