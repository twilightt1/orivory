"""Unit tests for MCP hub tools: scope enforcement + ledger writes.

The DB is faked; the retriever and the write-back side effects are
monkeypatched (CI-safe: no Chroma, no Celery, no real Postgres). These tests
call the tool implementations directly — the thin FastMCP wrappers are
exercised by the wiring test in test_server.py.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from app.mcp_hub import tools as hub_tools
from app.mcp_hub.identity import AgentPrincipal
from app.models.memory import Memory


def _principal(scopes: tuple[str, ...]) -> AgentPrincipal:
    return AgentPrincipal(
        user_id=uuid.uuid4(),
        agent_client_id=uuid.uuid4(),
        name="TestAgent",
        scopes=frozenset(scopes),
    )


def _memory(user_id=None, memory_id=None) -> Memory:
    """Convenience factory for appended progressive-disclosure tests."""
    return _memory_row(memory_id or uuid.uuid4(), user_id or uuid.uuid4())


def _memory_row(memory_id: uuid.UUID, user_id: uuid.UUID) -> Memory:
    return Memory(
        id=memory_id,
        user_id=user_id,
        title="pg indexing",
        content="Gin indexes speed up postgres lookups",
        tags=["db"],
        salience=0.9,
        captured_at=datetime.now(UTC),
    )


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows

    def first(self):
        return self._rows[0] if self._rows else None


class _FakeDB:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.added = []
        self.deleted = []
        self.committed = 0

    async def get(self, model, obj_id):
        for row in self.rows:
            if getattr(row, "id", None) == obj_id:
                return row
        return None

    async def execute(self, _stmt):
        return _FakeResult(self.rows)

    def add(self, obj):
        self.added.append(obj)

    async def delete(self, obj):
        self.deleted.append(obj)

    async def commit(self):
        self.committed += 1

    async def refresh(self, _obj):
        return None


class _FakeCtx:
    def __init__(self, db):
        self.db = db

    async def __aenter__(self):
        return self.db

    async def __aexit__(self, *exc):
        return False


def _fake_recall(ids):
    async def _recall(_query, _limit):
        return [(mid, 0.9) for mid in ids]

    return _recall


@pytest.fixture()
def reader(monkeypatch):
    p = _principal(("memory:read",))
    db = _FakeDB()
    monkeypatch.setattr(hub_tools, "_current_principal", lambda: p)
    monkeypatch.setattr(hub_tools, "_session", lambda: _FakeCtx(db))
    return p, db


@pytest.fixture()
def writer(monkeypatch):
    p = _principal(("memory:read", "memory:write"))
    db = _FakeDB()
    monkeypatch.setattr(hub_tools, "_current_principal", lambda: p)
    monkeypatch.setattr(hub_tools, "_session", lambda: _FakeCtx(db))
    return p, db


async def test_search_requires_read_scope(monkeypatch):
    p = _principal(())  # no scopes
    monkeypatch.setattr(hub_tools, "_current_principal", lambda: p)
    result = await hub_tools.search_memory(query="postgres")
    assert result == {"error": "scope memory:read required"}


async def test_search_requires_identity(monkeypatch):
    monkeypatch.setattr(hub_tools, "_current_principal", lambda: None)
    result = await hub_tools.search_memory(query="postgres")
    assert result == {"error": "agent identity required"}


async def test_search_returns_results_and_writes_ledger(reader, monkeypatch):
    p, db = reader
    memory = uuid.uuid4()
    db.rows = [_memory_row(memory, p.user_id)]
    monkeypatch.setattr(hub_tools, "_recall_memory_ids", _fake_recall([memory]))
    result = await hub_tools.search_memory(query="postgres indexing")
    assert result["results"][0]["id"] == str(memory)
    ledger = [o for o in db.added if type(o).__name__ == "MemoryAccessLog"]
    assert ledger and ledger[0].action == "mcp_search"
    assert ledger[0].detail["query"] == "postgres indexing"


async def test_add_memory_requires_write_scope(reader):
    result = await hub_tools.add_memory(title="t", content="c")
    assert result == {"error": "scope memory:write required"}


async def test_add_memory_creates_and_logs(writer, monkeypatch):
    p, db = writer
    indexed = []

    async def _fake_index(memory):
        indexed.append(memory)

    monkeypatch.setattr(hub_tools, "index_new_memory", _fake_index)
    result = await hub_tools.add_memory(title="Decision", content="Use pgvector", tags=["db"])
    assert result["title"] == "Decision"
    assert any(type(o).__name__ == "Memory" for o in db.added)
    assert any(type(o).__name__ == "MemoryAccessLog" and o.action == "mcp_add" for o in db.added)
    assert db.committed >= 1
    assert indexed and indexed[0].source_type == "mcp_agent"
    assert indexed[0].source_ref == "agent:TestAgent"
    assert indexed[0].user_id == p.user_id


async def test_delete_requires_write_scope(reader):
    result = await hub_tools.delete_memory(memory_id=str(uuid.uuid4()))
    assert result == {"error": "scope memory:write required"}


async def test_delete_owned_memory_deletes_and_logs(writer, monkeypatch):
    p, db = writer
    removed_from_chroma = []

    async def _fake_chroma_delete(memory_id):
        removed_from_chroma.append(memory_id)

    memory_id = uuid.uuid4()
    db.rows = [_memory_row(memory_id, p.user_id)]
    monkeypatch.setattr(hub_tools, "safe_delete_from_chroma", _fake_chroma_delete)
    result = await hub_tools.delete_memory(memory_id=str(memory_id))
    assert result["deleted"] is True
    assert db.deleted and db.deleted[0].id == memory_id
    ledger = [o for o in db.added if type(o).__name__ == "MemoryAccessLog"]
    assert ledger and ledger[0].action == "mcp_delete"
    assert ledger[0].memory_id == memory_id
    assert removed_from_chroma == [memory_id]
    assert db.committed >= 1


async def test_get_memory_returns_owned_row_and_logs(reader):
    p, db = reader
    memory_id = uuid.uuid4()
    db.rows = [_memory_row(memory_id, p.user_id)]
    result = await hub_tools.get_memory(memory_id=str(memory_id))
    assert result["id"] == str(memory_id)
    assert result["title"] == "pg indexing"
    ledger = [o for o in db.added if type(o).__name__ == "MemoryAccessLog"]
    assert ledger and ledger[0].action == "mcp_get"


async def test_list_recent_returns_rows_and_logs(reader):
    p, db = reader
    memory_id = uuid.uuid4()
    db.rows = [_memory_row(memory_id, p.user_id)]
    result = await hub_tools.list_recent(limit=5)
    assert result["results"][0]["id"] == str(memory_id)
    ledger = [o for o in db.added if type(o).__name__ == "MemoryAccessLog"]
    assert ledger and ledger[0].action == "mcp_list"
    assert ledger[0].detail["returned"] == 1


# ── Progressive disclosure: index rows + timeline ────────────────────────────


def test_memory_index_row_no_full_content():
    memory = _memory(user_id=uuid.uuid4())
    memory.content = "x" * 400
    row = hub_tools._memory_index_row(memory)
    assert row["id"] == str(memory.id)
    assert len(row["snippet"]) == 161  # 160 + ellipsis
    assert "content" not in row  # full body never in the index


def test_memory_index_row_short_content_kept():
    memory = _memory(user_id=uuid.uuid4())
    memory.content = "short note"
    row = hub_tools._memory_index_row(memory)
    assert row["snippet"] == "short note"


async def test_search_returns_index_rows(writer, monkeypatch):
    writer, db = writer
    """Search results carry snippets, not full bodies — progressive
    disclosure step 1. Callers filter on the index, then get_memory."""
    m1 = _memory(user_id=writer.user_id)
    m2 = _memory(user_id=writer.user_id)
    m2.content = "y" * 500
    monkeypatch.setattr(hub_tools, "_recall_memory_ids", _fake_recall([m1.id, m2.id]))
    db.rows = [m1, m2]
    out = await hub_tools.search_memory("postgres")
    assert out["results"][0]["snippet"] == m1.content
    assert len(out["results"][1]["snippet"]) == 161
    assert all("content" not in r for r in out["results"])
    # ledgered as search
    assert any(getattr(a, "action", "") == hub_tools.ACTION_SEARCH for a in db.added)


async def test_timeline_returns_anchor_and_neighbours(reader, monkeypatch):
    reader, db = reader
    """Timeline = anchor + before/after windows, snippets only (step 2)."""
    base = datetime.now(UTC)
    older = _memory(user_id=reader.user_id)
    older.captured_at = base - timedelta(hours=2)
    older.content = "older context"
    anchor = _memory(user_id=reader.user_id)
    anchor.captured_at = base
    newer = _memory(user_id=reader.user_id)
    newer.captured_at = base + timedelta(hours=2)
    newer.content = "newer context"
    db.rows = [newer, anchor, older]  # query returns desc order

    out = await hub_tools.timeline(memory_id=str(anchor.id), window=4)
    assert out["anchor"]["id"] == str(anchor.id)
    assert [b["id"] for b in out["before"]] == [str(older.id)]
    assert [a["id"] for a in out["after"]] == [str(newer.id)]
    assert all("content" not in row for row in [out["anchor"], *out["before"], *out["after"]])


async def test_timeline_foreign_memory_rejected(reader):
    reader, _db = reader
    out = await hub_tools.timeline(memory_id=str(uuid.uuid4()))
    assert out == {"error": "memory not found"}


async def test_timeline_invalid_id(reader):
    _reader, _db = reader
    out = await hub_tools.timeline(memory_id="not-a-uuid")
    assert out == {"error": "invalid memory id"}


async def test_timeline_windows_capped(reader, monkeypatch):
    reader, db = reader
    base = datetime.now(UTC)
    anchor = _memory(user_id=reader.user_id)
    anchor.captured_at = base
    rows = [anchor]
    for i in range(7):
        m = _memory(user_id=reader.user_id)
        m.captured_at = base - timedelta(hours=i + 1)
        rows.append(m)
    db.rows = rows
    out = await hub_tools.timeline(memory_id=str(anchor.id), window=4)
    assert len(out["before"]) == 4  # capped, not all 7


# ── Task 4: correct_memory ───────────────────────────────────────────────────


async def test_correct_memory_requires_write_scope(reader):
    result = await hub_tools.correct_memory(content="Postgres")
    assert result == {"error": "scope memory:write required"}


async def test_correct_memory_supersedes_and_logs(writer, monkeypatch):
    p, db = writer
    old = _memory_row(uuid.uuid4(), p.user_id)
    old.extra_metadata = {"cm_subject": "proj-x", "cm_attribute": "db", "cm_scope": "prod"}
    db.rows = [old]

    async def _noop_index(memory):
        return None

    monkeypatch.setattr(hub_tools, "index_new_memory", _noop_index)
    out = await hub_tools.correct_memory(subject="proj-x", attribute="db",
        scope="prod", title="DB", content="Postgres")
    assert out["status"] == "superseded"
    assert out["superseded"] == [str(old.id)]
    ledger = [o for o in db.added if type(o).__name__ == "MemoryAccessLog"]
    assert ledger and ledger[0].action == "mcp_correct"


async def test_correct_memory_foreign_id_rejected(writer):
    _p, _db = writer
    out = await hub_tools.correct_memory(memory_id=str(uuid.uuid4()), content="x")
    assert out == {"error": "memory not found"}
