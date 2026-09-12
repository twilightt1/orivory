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


# ── IDEA 3: apply_closure ───────────────────────────────────────────

def _slot_mem(subject, attribute, scope='default', **kw):
    from types import SimpleNamespace
    meta = {'cm_subject': subject, 'cm_attribute': attribute, 'cm_scope': scope}
    meta.update(kw)
    return SimpleNamespace(extra_metadata=meta)


def test_closure_swaps_same_slot_mate():
    from app.retrieval.memory.scoring import apply_closure
    a = _slot_mem('anh tuấn', 'số điện thoại')
    b = _slot_mem('anh tuấn', 'email')
    a2 = _slot_mem('anh tuấn', 'số điện thoại')
    scored = [(a, 0.9, []), (b, 0.8, []), (a2, 0.7, [])]
    top = apply_closure(scored, 2, cap=2)
    assert top[0][0] is a
    assert top[1][0] is a2
    assert 'closure:slot' in top[1][2]
    assert scored[2][2] == []  # input reasons not mutated


def test_closure_cap_respected():
    from app.retrieval.memory.scoring import apply_closure
    a = _slot_mem('anh tuấn', 'số điện thoại')
    b = _slot_mem('anh tuấn', 'email')
    a2 = _slot_mem('anh tuấn', 'số điện thoại')
    a3 = _slot_mem('anh tuấn', 'số điện thoại')
    scored = [(a, 0.9, []), (b, 0.8, []), (a2, 0.7, []), (a3, 0.65, [])]
    top = apply_closure(scored, 2, cap=1)
    assert top[0][0] is a
    assert top[1][0] is a2


def test_closure_skips_superseded_and_dirty():
    from app.retrieval.memory.scoring import apply_closure
    a = _slot_mem('anh tuấn', 'số điện thoại')
    b = _slot_mem('anh tuấn', 'email')
    stale = _slot_mem('anh tuấn', 'số điện thoại', cm_superseded_by='other-id')
    dirty = _slot_mem('anh tuấn', 'số điện thoại', cm_derived_dirty=True)
    scored = [(a, 0.9, []), (b, 0.8, []), (stale, 0.7, []), (dirty, 0.65, [])]
    top = apply_closure(scored, 2, cap=2)
    assert top[0][0] is a
    assert top[1][0] is b


def test_closure_no_slot_no_swap():
    from types import SimpleNamespace

    from app.retrieval.memory.scoring import apply_closure
    scored = [(SimpleNamespace(extra_metadata={}), s, []) for s in (0.9, 0.8, 0.7)]
    top = apply_closure(scored, 2, cap=2)
    assert [s for _, s, _ in top] == [0.9, 0.8]


def test_closure_cap_zero_is_off():
    from app.retrieval.memory.scoring import apply_closure
    a = _slot_mem('anh tuấn', 'số điện thoại')
    b = _slot_mem('anh tuấn', 'email')
    a2 = _slot_mem('anh tuấn', 'số điện thoại')
    scored = [(a, 0.9, []), (b, 0.8, []), (a2, 0.7, [])]
    top = apply_closure(scored, 2, cap=0)
    assert top[0][0] is a
    assert top[1][0] is b


async def test_recall_closure_swap(monkeypatch):
    import uuid
    from datetime import UTC, datetime

    from app.models.memory import Memory
    from app.retrieval.memory import retriever as R

    uid = uuid.uuid4()
    now = datetime.now(UTC)

    def _mem(score_slot, vector_score):
        return Memory(
            id=uuid.uuid4(), user_id=uid, title='ghi chú', content='nội dung ghi chú',
            tags=[], salience=0.5, pinned=False, source_type='manual_note', recall_count=0,
            captured_at=now, indexed_at=now, updated_at=now, extra_metadata=score_slot,
        )

    slot_x = {'cm_subject': 'anh tuấn', 'cm_attribute': 'số điện thoại', 'cm_scope': 'default'}
    slot_y = {'cm_subject': 'anh tuấn', 'cm_attribute': 'email', 'cm_scope': 'default'}
    mem_a, mem_b, mem_a2 = _mem(slot_x, 0.9), _mem(slot_y, 0.8), _mem(slot_x, 0.7)
    mems = {str(m.id): m for m in (mem_a, mem_b, mem_a2)}

    class _FakeResult:
        def scalars(self):
            return self

        def all(self):
            return list(mems.values())

    class _FakeDB:
        async def execute(self, _stmt):
            return _FakeResult()

    async def _rw(q, context=None):
        return {'rewritten_query': q, 'entities': [], '_fallback_used': False, 'reasoning': ''}

    async def _emb(q):
        return [0.1, 0.2]

    async def _search(emb, user_id=None, top_k=10):
        return [
            {'memory_id': str(mem_a.id), 'score': 0.9},
            {'memory_id': str(mem_b.id), 'score': 0.8},
            {'memory_id': str(mem_a2.id), 'score': 0.7},
        ]

    monkeypatch.setattr(R, 'rewrite_query', _rw)
    monkeypatch.setattr(R, 'embed_query', _emb)
    monkeypatch.setattr(R, 'search_memories', _search)

    out = await R.MemoryRetriever(_FakeDB(), uid, semantic_rerank=False).recall(
        'thời tiết hôm nay', top_k=2, include_personal_context=False)
    ids = {r.id for r in out.results}
    assert mem_a.id in ids and mem_a2.id in ids and mem_b.id not in ids
    mate = next(r for r in out.results if r.id == mem_a2.id)
    assert 'closure:slot' in mate.match_reasons

    off = await R.MemoryRetriever(
        _FakeDB(), uid, semantic_rerank=False, closure_cap=0).recall(
        'thời tiết hôm nay', top_k=2, include_personal_context=False)
    assert {r.id for r in off.results} == {mem_a.id, mem_b.id}
