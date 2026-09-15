"""Task 3 tests: recall hides superseded/dirty, fast-path skip, stage timings."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime


def _mem(meta: dict | None, uid=None):
    from app.models.memory import Memory

    return Memory(
        id=uuid.uuid4(),
        user_id=uid or uuid.uuid4(),
        title="t",
        content="x",
        tags=[],
        salience=0.5,
        pinned=False,
        source_type="manual_note",
        recall_count=0,
        captured_at=datetime.now(UTC),
        indexed_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        extra_metadata=meta or {},
    )


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _FakeDB:
    def __init__(self, rows=None):
        self.rows = rows or []

    async def execute(self, _stmt):
        return _FakeResult(self.rows)


async def test_recall_hides_superseded_and_dirty(monkeypatch, barrier_outbox):
    # ``barrier_outbox``: the R14 freshness barrier runs first on every recall
    # and reads its own outbox (tests/retrieval/conftest.py) — even when the
    # retriever's DB is a fake.
    from app.retrieval.memory import retriever as R
    uid = uuid.uuid4()
    cur = _mem({}, uid=uid)
    old = _mem({"cm_superseded_by": str(cur.id)}, uid=uid)
    dirty = _mem({"cm_derived_from": [str(cur.id)], "cm_derived_dirty": True}, uid=uid)
    db = _FakeDB(rows=[cur, old, dirty])
    async def _rw(q, context=None): return {"rewritten_query": q, "entities": [], "_fallback_used": False, "reasoning": ""}
    async def _emb(q): return [0.1, 0.2]
    async def _search(emb, user_id=None, top_k=10):
        return [{"memory_id": str(old.id), "score": 0.99},
                {"memory_id": str(dirty.id), "score": 0.98},
                {"memory_id": str(cur.id), "score": 0.5}]
    monkeypatch.setattr(R, "rewrite_query", _rw)
    monkeypatch.setattr(R, "embed_query", _emb)
    monkeypatch.setattr(R, "search_memories", _search)
    # ponytail: FakeDB.execute ignores the statement; candidate ids come from
    # the fake Chroma above, filtering happens in Python like production.
    r = R.MemoryRetriever(db, uid, semantic_rerank=False)
    out = await r.recall("db prod", top_k=3, include_personal_context=False)
    assert [x.id for x in out.results] == [cur.id]
    assert out.trace.stage_ms.keys() >= {"rewrite_ms", "embed_ms", "search_ms", "hydrate_ms"}


async def test_recall_fast_path_skips_llm(monkeypatch, barrier_outbox):
    from app.retrieval.memory import retriever as R
    uid = uuid.uuid4()
    cur = _mem({}, uid=uid)
    db = _FakeDB(rows=[cur])
    called = []
    async def _rw(q, context=None):
        called.append(q)
        return {"rewritten_query": q, "entities": [], "_fallback_used": False, "reasoning": ""}
    async def _emb(q): return [0.1, 0.2]
    async def _search(emb, user_id=None, top_k=10):
        return [{"memory_id": str(cur.id), "score": 0.9}]
    monkeypatch.setattr(R, "rewrite_query", _rw)
    monkeypatch.setattr(R, "embed_query", _emb)
    monkeypatch.setattr(R, "search_memories", _search)
    r = R.MemoryRetriever(db, uid, semantic_rerank=False)
    out = await r.recall("postgres indexing", top_k=1, include_personal_context=False)
    assert called == [] and out.trace.rewrite_skipped is True
