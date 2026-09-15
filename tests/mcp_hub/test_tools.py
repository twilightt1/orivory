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
from app.models.erasure_receipt import ErasureReceipt
from app.models.memory import Memory
from app.observability.fallbacks import fallback_counts, reset_fallback_counts
from app.retrieval.embedder import EmbeddingDimensionMismatch
from app.retrieval.memory.outbox import IndexFreshnessTimeout
from app.retrieval.vector_retriever import VectorUnavailableError


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


def _stub_retriever(ids, *, error=None):
    """Stands in for ``MemoryRetriever``: only ``recall_ids`` is on the seam."""
    seen: dict = {}

    class _Stub:
        def __init__(self, db, user_id):
            seen["user_id"] = user_id

        async def recall_ids(self, query, top_k=10):
            seen["query"] = query
            seen["top_k"] = top_k
            if error is not None:
                raise error
            return [(mid, 0.9) for mid in ids]

    return _Stub, seen


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


async def test_delete_owned_memory_goes_through_erasure_and_logs(writer, monkeypatch):
    p, db = writer
    memory_id = uuid.uuid4()
    db.rows = [_memory_row(memory_id, p.user_id)]
    receipt_id = uuid.uuid4()
    seen: list[dict] = []

    async def _fake_erase(db_, user_id, memory_ids, *, requested_by):
        seen.append({"user_id": user_id, "ids": list(memory_ids), "requested_by": requested_by})
        return ErasureReceipt(id=receipt_id, user_id=user_id,
                              requested_memory_ids=[str(m) for m in memory_ids],
                              status="completed",
                              detail={"targets": [{"memory_id": str(memory_ids[0]), "status": "deleted"}]})

    monkeypatch.setattr(hub_tools, "erase_memories", _fake_erase)
    result = await hub_tools.delete_memory(memory_id=str(memory_id))

    assert result["deleted"] is True and result["id"] == str(memory_id)
    assert result["receipt_id"] == str(receipt_id)
    # The ad-hoc delete is gone: the durable erasure path is the only one.
    assert seen == [{"user_id": p.user_id, "ids": [memory_id], "requested_by": "agent:TestAgent"}]
    ledger = [o for o in db.added if type(o).__name__ == "MemoryAccessLog"]
    assert ledger and ledger[0].action == "mcp_delete"
    # The memory is gone by the time the ledger row lands, and `memory_id` FKs
    # memories.id (SET NULL) — so the id rides the detail, like forget does.
    assert ledger[0].memory_id is None
    assert ledger[0].detail == {"deleted": True, "memory_id": str(memory_id),
                                "receipt_id": str(receipt_id)}
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


# ── Task 5: add via resolve, search state, get provenance ────────────────────


async def test_search_hides_superseded_by_default(writer, monkeypatch):
    p, db = writer
    new = _memory_row(uuid.uuid4(), p.user_id)
    old = _memory_row(uuid.uuid4(), p.user_id)
    old.extra_metadata = {"cm_superseded_by": str(new.id)}
    db.rows = [old, new]
    monkeypatch.setattr(hub_tools, "_recall_memory_ids", _fake_recall([old.id, new.id]))
    out = await hub_tools.search_memory("x")
    assert [r["id"] for r in out["results"]] == [str(new.id)]
    assert out["results"][0]["state"] == "current"
    out2 = await hub_tools.search_memory("x", include_history=True)
    assert {r["id"] for r in out2["results"]} == {str(old.id), str(new.id)}


async def test_add_never_supersedes(writer, monkeypatch):
    p, db = writer
    old = _memory_row(uuid.uuid4(), p.user_id)
    old.extra_metadata = {"cm_subject": "proj-x", "cm_attribute": "db", "cm_scope": "prod"}
    db.rows = [old]

    async def _noop_index(memory):
        return None

    monkeypatch.setattr(hub_tools, "index_new_memory", _noop_index)
    out = await hub_tools.add_memory(title="DB", content="SQLite")
    assert old.extra_metadata.get("cm_superseded_by") is None
    assert out["state"] == "current"


async def test_get_carries_provenance(reader):
    p, db = reader
    m = _memory_row(uuid.uuid4(), p.user_id)
    m.extra_metadata = {"cm_subject": "proj-x", "cm_scope": "prod",
                        "cm_supersedes": "prev-id", "cm_evidence_ids": ["e1"]}
    db.rows = [m]
    out = await hub_tools.get_memory(str(m.id))
    assert out["scope"] == "prod" and out["supersedes"] == "prev-id"
    assert out["evidence_ids"] == ["e1"] and out["state"] == "current"


# ── Task 6: the shared recall seam + the R23 fallback ────────────────────────


async def test_search_ranks_through_the_shared_recall_seam(reader, monkeypatch):
    """R22(p2): the ordering is the retriever's id seam — MCP re-ranks nothing,
    and the SQL ordering is not consulted at all on the healthy path."""
    p, db = reader
    semantic = _memory_row(uuid.uuid4(), p.user_id)
    semantic.salience = 0.05  # the fixture is honest: salience says LAST
    salient = _memory_row(uuid.uuid4(), p.user_id)
    salient.salience = 0.99
    db.rows = [semantic, salient]
    stub, seen = _stub_retriever([semantic.id, salient.id])
    monkeypatch.setattr(hub_tools, "MemoryRetriever", stub)

    async def _tripwire(*_args, **_kwargs):
        raise AssertionError("the SQL ordering is only the R23 fallback")

    monkeypatch.setattr(hub_tools, "_sql_recall_ids", _tripwire)

    out = await hub_tools.search_memory("semantic query")

    assert [r["id"] for r in out["results"]] == [str(semantic.id), str(salient.id)]
    assert seen == {"user_id": p.user_id, "query": "semantic query", "top_k": 8}
    # Payload contract unchanged: index-only rows with the 160-char snippet.
    for row in out["results"]:
        assert set(row) == {"id", "title", "snippet", "tags", "salience",
                            "captured_at", "state"}
    assert out["results"][0]["snippet"] == semantic.content
    ledger = [o for o in db.added if type(o).__name__ == "MemoryAccessLog"]
    assert len(ledger) == 1 and ledger[0].detail["returned"] == 2


async def test_search_cap_reaches_the_seam(reader, monkeypatch):
    """MAX_SEARCH_LIMIT=20 stays the tool's own cap (it does not fork the
    retriever's top_k)."""
    _p, _db = reader
    stub, seen = _stub_retriever([])
    monkeypatch.setattr(hub_tools, "MemoryRetriever", stub)

    await hub_tools.search_memory("q", limit=999)
    assert seen["top_k"] == hub_tools.MAX_SEARCH_LIMIT
    await hub_tools.search_memory("q", limit=0)
    assert seen["top_k"] == 1


async def test_search_barrier_timeout_falls_back_to_sql_order(reader, monkeypatch):
    """R23(p2): the barrier's typed timeout never becomes a tool error — the
    call is answered from the SQL ordering, ledger and payload unchanged."""
    p, db = reader
    first = _memory_row(uuid.uuid4(), p.user_id)
    second = _memory_row(uuid.uuid4(), p.user_id)
    db.rows = [first, second]
    stub, _ = _stub_retriever([], error=IndexFreshnessTimeout("pending writes"))
    monkeypatch.setattr(hub_tools, "MemoryRetriever", stub)
    reset_fallback_counts()

    out = await hub_tools.search_memory("q")

    assert "error" not in out
    assert [r["id"] for r in out["results"]] == [str(first.id), str(second.id)]
    assert fallback_counts()[hub_tools.SQL_FALLBACK_PATH] == 1
    ledger = [o for o in db.added if type(o).__name__ == "MemoryAccessLog"]
    assert len(ledger) == 1 and ledger[0].detail["returned"] == 2


async def test_search_vector_outage_falls_back_to_sql_order(reader, monkeypatch):
    """R23(p2): the outage that would be the API's typed 503 answers here from
    the SQL ordering instead."""
    p, db = reader
    row = _memory_row(uuid.uuid4(), p.user_id)
    db.rows = [row]
    stub, _ = _stub_retriever([], error=VectorUnavailableError("vector store down"))
    monkeypatch.setattr(hub_tools, "MemoryRetriever", stub)
    reset_fallback_counts()

    out = await hub_tools.search_memory("q")

    assert "error" not in out
    assert [r["id"] for r in out["results"]] == [str(row.id)]
    assert fallback_counts()[hub_tools.SQL_FALLBACK_PATH] == 1


async def test_search_embedding_contract_mismatch_still_raises(reader, monkeypatch):
    """The integrity contract the module pins: a contract mismatch is typed and
    raised, never silently degraded into a different ranking."""
    _p, _db = reader
    stub, _ = _stub_retriever([], error=EmbeddingDimensionMismatch("384 vs 1536"))
    monkeypatch.setattr(hub_tools, "MemoryRetriever", stub)

    with pytest.raises(EmbeddingDimensionMismatch):
        await hub_tools.search_memory("q")
