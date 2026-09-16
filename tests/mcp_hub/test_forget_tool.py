"""Unit tests for the forget_memory MCP tool: scope enforcement + ledger."""
from __future__ import annotations

import uuid

import pytest

from app.mcp_hub import tools as hub_tools
from app.mcp_hub.identity import ACTION_FORGET, AgentPrincipal
from app.models.erasure_receipt import ErasureReceipt
from app.services.erasure_service import ERASURE_STATUS_COMPLETED


def _principal(scopes: tuple[str, ...]) -> AgentPrincipal:
    return AgentPrincipal(user_id=uuid.uuid4(), agent_client_id=uuid.uuid4(), name="TestAgent", scopes=frozenset(scopes))


def _receipt(user_id: uuid.UUID, memory_id: uuid.UUID, *, erased: int, skipped: int) -> ErasureReceipt:
    return ErasureReceipt(
        id=uuid.uuid4(),
        user_id=user_id,
        requested_memory_ids=[str(memory_id)],
        status=ERASURE_STATUS_COMPLETED,
        detail={"summary": {"requested": 1, "erased": erased, "skipped": skipped, "residual_vectors": 0, "residual_rows": 0}},
    )


class _FakeRows:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _FakeDB:
    def __init__(self, *, allowed: list | None = None):
        self.added = []
        self.committed = 0
        self.allowed = allowed  # the R35 pre-read's answer; None = every id requested

    async def execute(self, stmt):
        """The namespace pre-read (R35) — the tool's one SELECT.

        By default it resolves every id the statement asked for (the tests below
        are about the service call, not the filter); ``allowed=[]`` is the
        cross-namespace case.
        """
        if self.allowed is not None:
            return _FakeRows(list(self.allowed))
        asked = next((value for value in stmt.compile().params.values()
                      if isinstance(value, (list, tuple))), [])
        return _FakeRows(list(asked))

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.committed += 1


class _FakeCtx:
    def __init__(self, db):
        self.db = db

    async def __aenter__(self):
        return self.db

    async def __aexit__(self, *exc):
        return False


@pytest.fixture()
def writer(monkeypatch):
    p = _principal(("memory:read", "memory:write"))
    db = _FakeDB()

    async def _fake_erase(db_, user_id, memory_ids, *, requested_by):
        assert user_id == p.user_id
        assert requested_by == "agent:TestAgent"
        await db_.commit()  # receipt commit happens inside the real service
        return _receipt(user_id, memory_ids[0], erased=len(memory_ids), skipped=0)

    monkeypatch.setattr(hub_tools, "_current_principal", lambda: p)
    monkeypatch.setattr(hub_tools, "_session", lambda: _FakeCtx(db))
    monkeypatch.setattr(hub_tools, "erase_memories", _fake_erase)
    return p, db


async def test_forget_requires_identity(monkeypatch):
    calls: list[object] = []

    async def _no_erase(*args, **kwargs):
        calls.append(args)
        raise AssertionError("erase_memories must not run on denial")

    monkeypatch.setattr(hub_tools, "_current_principal", lambda: None)
    monkeypatch.setattr(hub_tools, "erase_memories", _no_erase)
    assert await hub_tools.forget_memory(memory_ids=[str(uuid.uuid4())]) == {"error": "agent identity required"}
    assert not calls  # spy never raised


async def test_forget_requires_write_scope(monkeypatch):
    calls: list[object] = []

    async def _no_erase(*args, **kwargs):
        calls.append(args)
        raise AssertionError("erase_memories must not run on denial")

    monkeypatch.setattr(hub_tools, "_current_principal", lambda: _principal(("memory:read",)))
    monkeypatch.setattr(hub_tools, "erase_memories", _no_erase)
    assert await hub_tools.forget_memory(memory_ids=[str(uuid.uuid4())]) == {"error": "scope memory:write required"}
    assert not calls  # spy never raised


async def test_forget_all_invalid_ids(monkeypatch):
    monkeypatch.setattr(hub_tools, "_current_principal", lambda: _principal(("memory:write",)))
    assert await hub_tools.forget_memory(memory_ids=["not-a-uuid"]) == {"error": "invalid memory id"}


async def test_forget_calls_erasure_service_returns_summary_and_logs(writer):
    _p, db = writer
    mid = uuid.uuid4()
    result = await hub_tools.forget_memory(memory_ids=[str(mid)])

    assert result["erased"] == 1 and result["skipped"] == 0
    assert result["invalid"] == []
    assert result["status"] == ERASURE_STATUS_COMPLETED
    assert uuid.UUID(result["receipt_id"])
    ledger = [o for o in db.added if type(o).__name__ == "MemoryAccessLog"]
    assert ledger and ledger[0].action == ACTION_FORGET == "mcp_forget"
    assert ledger[0].detail["receipt_id"] == result["receipt_id"]
    assert ledger[0].detail["requested"] == [str(mid)]
    assert db.committed == 2  # receipt commit inside service + ledger commit


async def test_forget_filters_invalid_ids_and_reports_them(writer, monkeypatch):
    _p, _ = writer
    seen: list[list[uuid.UUID]] = []

    async def _spy(db_, user_id, memory_ids, *, requested_by):
        seen.append(list(memory_ids))
        await db_.commit()  # receipt commit happens inside the real service
        return _receipt(user_id, memory_ids[0], erased=len(memory_ids), skipped=0)

    monkeypatch.setattr(hub_tools, "erase_memories", _spy)
    good = str(uuid.uuid4())
    result = await hub_tools.forget_memory(memory_ids=[good, "nope"])

    assert [str(m) for m in seen[0]] == [good]
    assert result["invalid"] == ["nope"]


async def test_forget_never_hands_an_id_outside_the_namespace_to_erasure(monkeypatch):
    """R35a: the boundary is resolved BEFORE the service sees the ids.

    The reviewer's reproduce, at the tool boundary: a same-account row in
    another namespace must not reach the erase path at all.
    """
    p = _principal(("memory:write",))
    db = _FakeDB(allowed=[])  # the pre-read resolves nothing: not the caller's row
    seen: list[list[uuid.UUID]] = []

    async def _spy(db_, user_id, memory_ids, *, requested_by):
        seen.append(list(memory_ids))
        await db_.commit()
        return _receipt(user_id, uuid.uuid4(), erased=len(memory_ids), skipped=0)

    monkeypatch.setattr(hub_tools, "_current_principal", lambda: p)
    monkeypatch.setattr(hub_tools, "_session", lambda: _FakeCtx(db))
    monkeypatch.setattr(hub_tools, "erase_memories", _spy)

    out = await hub_tools.forget_memory(memory_ids=[str(uuid.uuid4())])

    assert seen == [[]], "a row outside the namespace never reaches the erasure service"
    assert out["erased"] == 0 and out["skipped"] == 1
