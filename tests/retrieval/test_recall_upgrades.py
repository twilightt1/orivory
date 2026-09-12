"""Recall upgrades: lexical refinement, query routing, same-slot closure."""
from __future__ import annotations

# ── IDEA 1: lexical_bonus ─────────────────────────────────────────────

def test_lexical_bonus_exact_match():
    from app.retrieval.memory.scoring import lexical_bonus
    bonus, reasons = lexical_bonus('Hợp đồng với "Anh Tuấn" 2024', 'anh tuấn ký hợp đồng năm 2024')
    assert bonus == 0.15
    assert any("tuấn" in r.lower() for r in reasons)


def test_lexical_bonus_no_match_zero():
    from app.retrieval.memory.scoring import lexical_bonus
    bonus, reasons = lexical_bonus('"Anh Tuấn" Hà Nội', 'thời tiết hôm nay đẹp')
    assert bonus == 0.0
    assert reasons == []


def test_lexical_bonus_empty_query():
    from app.retrieval.memory.scoring import lexical_bonus
    assert lexical_bonus('', 'anything here') == (0.0, [])


# ── IDEA 2: route_query ───────────────────────────────────────────────

def test_route_relational_on_entities():
    from app.retrieval.memory.scoring import route_query
    assert route_query('hợp đồng đó thế nào', [{'name': 'Anh Tuấn', 'type': 'person'}]) == 'relational'


def test_route_relational_on_two_capitalized_tokens():
    from app.retrieval.memory.scoring import route_query
    assert route_query('Anh Tuấn Hà Nội hôm qua', []) == 'relational'


def test_route_local_on_time_words():
    from app.retrieval.memory.scoring import route_query
    assert route_query('tóm tắt quá trình làm việc', []) == 'local'
    assert route_query('when did we sign the contract', []) == 'local'


def test_route_general_default():
    from app.retrieval.memory.scoring import route_query
    assert route_query('thời tiết hôm nay', []) == 'general'


async def test_recall_trace_carries_route(monkeypatch):
    import uuid
    from datetime import UTC, datetime

    from app.models.memory import Memory
    from app.retrieval.memory import retriever as R

    uid = uuid.uuid4()
    mem = Memory(
        id=uuid.uuid4(), user_id=uid, title='t', content='x', tags=[],
        salience=0.5, pinned=False, source_type='manual_note', recall_count=0,
        captured_at=datetime.now(UTC), indexed_at=datetime.now(UTC),
        updated_at=datetime.now(UTC), extra_metadata={},
    )

    class _FakeResult:
        def scalars(self):
            return self

        def all(self):
            return [mem]

    class _FakeDB:
        async def execute(self, _stmt):
            return _FakeResult()

    async def _rw(q, context=None):
        return {'rewritten_query': q, 'entities': [], '_fallback_used': False, 'reasoning': ''}

    async def _emb(q):
        return [0.1, 0.2]

    async def _search(emb, user_id=None, top_k=10):
        return [{'memory_id': str(mem.id), 'score': 0.9}]

    monkeypatch.setattr(R, 'rewrite_query', _rw)
    monkeypatch.setattr(R, 'embed_query', _emb)
    monkeypatch.setattr(R, 'search_memories', _search)
    r = R.MemoryRetriever(_FakeDB(), uid, semantic_rerank=False)
    out = await r.recall('tóm tắt quá trình làm việc', top_k=1, include_personal_context=False)
    assert out.trace.route == 'local'
