"""Hybrid recall (P2/T5): UUID-keyed RRF fusion behind a flag + the lexical outage fallback.

Rulings pinned here:

- **R2(p2)** — the flag ships OFF and the OFF path is dense-only: the lexical
  index is not even probed, the trace counters gain no ``lexical``/``fused``
  key and ``stage_ms["lexical"]`` stays the schema's zero.
- **R11b(p2)** — fusion is keyed by canonical Memory UUID, never by content or
  ``parent_id`` (the legacy helper's rule is forbidden for memory), with
  zero-based ranks and ``sum(1/(k+rank+1))``; the legs' own scores (dense
  cosine, global BM25) travel separately on each candidate. Two rows with
  IDENTICAL text and different UUIDs both survive (§7.4).
- **R19(p2)** — a vector outage answers from the lexical leg when FTS exists
  (``count_fallback("retrieval.vector_unavailable")``, ``counts["lexical"]``,
  no ``dense`` key) and keeps the typed 503 when it does not (Postgres).

Also pinned here: the lexical leg's tenant + visibility clauses stay BEFORE
its LIMIT (a full page of better-matching foreign/superseded rows must not
starve the tenant's own row), and no candidate text reaches the reranker
before the SQL authorization replaced it with the canonical row.

The lexical leg is REAL in these tests — a temp-file SQLite FTS5 index with
T4's triggers — so the fusion is exercised against the actual BM25 ranking,
not a stub of it.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app import database, models  # noqa: F401 — register every table on Base
from app.database import Base
from app.models.memory import Memory
from app.models.user import User
from app.observability.fallbacks import fallback_counts, reset_fallback_counts
from app.retrieval import reranker as reranker_module
from app.retrieval.hybrid_retriever import fuse_by_uuid
from app.retrieval.memory import lexical_index
from app.retrieval.memory import retriever as rmod
from app.retrieval.memory.retriever import MemoryRetriever
from app.retrieval.vector_retriever import VectorUnavailableError

TENANT_A = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
TENANT_B = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

# One shared capture time, ten years back: EVERY row decays to the same floor
# (0.1) and carries the same salience, so the served score is `rrf x 0.1` for
# all of them and the served ORDER is the fusion's own — no clock noise.
CAPTURED = datetime.now(UTC) - timedelta(days=3650)
DECAY_MULT = 0.1
QUERY = "alpha charlie"
MATCHING = "alpha charlie"
UNMATCHED = "bravo delta"
VECTOR_COPY = "STALE VECTOR COPY — must never leave before SQL authorization"

# The UUIDs also pin the FTS tie-break: identical text scores identically in
# BM25, so the smaller canonical UUID wins the earlier rank.
A1 = uuid.UUID("11111111-1111-1111-1111-111111111111")
A2 = uuid.UUID("22222222-2222-2222-2222-222222222222")
A3 = uuid.UUID("33333333-3333-3333-3333-333333333333")
B1 = uuid.UUID("00000000-0000-0000-0000-000000000001")
B2 = uuid.UUID("00000000-0000-0000-0000-000000000002")
B3 = uuid.UUID("00000000-0000-0000-0000-000000000003")
SUPERSEDED = uuid.UUID("00000000-0000-0000-0000-000000000004")
CURRENT = uuid.UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")


def _mem(memory_id, user_id, content, *, superseded=False):
    meta: dict = {}
    if superseded:
        meta["cm_superseded_by"] = str(uuid.uuid4())
    return Memory(
        id=memory_id, user_id=user_id, title=None, content=content, tags=[],
        salience=0.5, pinned=False, source_type="manual_note", recall_count=0,
        captured_at=CAPTURED, indexed_at=CAPTURED, updated_at=CAPTURED,
        extra_metadata=meta,
    )


def _rrf(*ranks, k=60):
    return sum(1.0 / (k + rank + 1) for rank in ranks)


class _Store:
    """The dense leg: fixed rows ``(memory_id, score)`` best first — or an outage."""

    def __init__(self, rows, *, outage=False):
        self.rows = list(rows)
        self.outage = outage
        self.calls: list[int] = []

    async def __call__(self, _embedding, *, user_id, top_k=10, where=None):
        self.calls.append(top_k)
        if self.outage:
            raise VectorUnavailableError("vector store down")
        return [
            {"memory_id": str(memory_id), "content": VECTOR_COPY, "score": score}
            for memory_id, score in self.rows[:top_k]
        ]


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _PostgresishSession:
    """A non-SQLite deployment session: ``run_sync`` hands over a sync Session
    whose connection says postgres, so ``is_available`` answers False WITHOUT
    touching it, and hydrate answers with a fixed page."""

    def __init__(self, rows=()):
        self.rows = list(rows)

    async def execute(self, _statement):
        return _FakeResult(self.rows)

    async def run_sync(self, fn):
        conn = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))
        return fn(SimpleNamespace(connection=lambda: conn))


@pytest_asyncio.fixture
async def recall_db(tmp_path):
    """A real SQLite file carrying T4's FTS index (DDL + triggers)."""
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'hybrid-recall.db'}", poolclass=NullPool
    )
    event.listen(engine.sync_engine, "connect", database._configure_sqlite_connection)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(lexical_index.create_index)
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    try:
        yield engine, factory
    finally:
        await engine.dispose()


async def _seed(factory, *memories):
    """Insert every row (and its tenant) in one commit — the FTS triggers fire."""
    async with factory() as session:
        for user_id in {memory.user_id for memory in memories}:
            session.add(User(id=user_id, email=f"{user_id}@test.invalid",
                             onboarding_done=True, is_verified=True, is_active=True,
                             is_deleted=False))
        session.add_all(memories)
        await session.commit()


@pytest.fixture
def offline_env(monkeypatch, barrier_outbox):
    """Every out-of-process seam of recall faked except the LEXICAL leg, which
    stays real (SQLite FTS5) — see tests/retrieval/conftest.py for the outbox."""

    async def _rewrite(query, context=None):
        return {"rewritten_query": query, "entities": [], "reasoning": None,
                "_fallback_used": False}

    async def _embed(_query):
        return [0.1, 0.2]

    monkeypatch.setattr(rmod, "rewrite_query", _rewrite)
    monkeypatch.setattr(rmod, "embed_query", _embed)
    monkeypatch.setattr(rmod, "fetch_personal_context", AsyncMock(return_value=[]))


# ── the fusion itself (R11b) ────────────────────────────────────────────────


class TestUuidKeyedFusion:
    def test_zero_based_ranks_kept_scores_and_no_content_dedupe(self):
        dense = [{"memory_id": str(A1), "score": 0.91, "content": MATCHING},
                 {"memory_id": str(A2), "score": 0.72, "content": MATCHING}]
        lexical = [{"memory_id": str(A2), "score": -7.5, "rank": 0},
                   {"memory_id": str(A3), "score": -6.0, "rank": 1},
                   {"memory_id": str(A1), "score": -5.0, "rank": 2}]

        fused = fuse_by_uuid(dense, lexical, k=60)

        # A2 is dense rank 1 + lexical rank 0; A1 dense rank 0 + lexical rank 2;
        # A3 lexical rank 1 only. All sums distinct → a deterministic order.
        assert [row["memory_id"] for row in fused] == [str(A2), str(A1), str(A3)]
        assert fused[0]["score"] == pytest.approx(_rrf(1, 0))
        assert fused[1]["score"] == pytest.approx(_rrf(0, 2))
        assert fused[2]["score"] == pytest.approx(_rrf(1))
        # The legs' own scores stay separate — a fuse by rank, never by score.
        assert fused[0]["dense_score"] == 0.72 and fused[0]["lexical_score"] == -7.5
        assert fused[1]["dense_score"] == 0.91 and fused[1]["lexical_score"] == -5.0
        assert fused[2]["dense_score"] is None and fused[2]["lexical_score"] == -6.0

    def test_identical_text_with_different_uuids_both_survive(self):
        """§7.4: the legacy helper collapses these by md5(content) — forbidden
        for memory, where two facts can carry the same text."""
        dense = [{"memory_id": str(A1), "score": 0.9, "content": MATCHING},
                 {"memory_id": str(A2), "score": 0.8, "content": MATCHING}]

        fused = fuse_by_uuid(dense, [], k=60)

        assert [row["memory_id"] for row in fused] == [str(A1), str(A2)]

    def test_k_moves_the_sum_and_an_empty_leg_is_the_single_leg_case(self):
        row = {"memory_id": str(A1), "score": 0.5}
        assert fuse_by_uuid([row], [], k=60)[0]["score"] == pytest.approx(1 / 61)
        assert fuse_by_uuid([row], [], k=1)[0]["score"] == pytest.approx(1 / 2)
        # The rank IS the position in the leg: one row is rank 0 whatever its
        # own `rank` field says, so k=10 gives 1/11, not 1/13.
        lexical = [{"memory_id": str(A2), "score": -3.0, "rank": 2}]
        assert fuse_by_uuid([], lexical, k=10)[0]["score"] == pytest.approx(1 / 11)

    def test_a_repeated_row_votes_once_per_leg(self):
        """A drifted FTS index can repeat a memory_id (T4's DISTINCT-free
        join); its second appearance must not double-count the leg."""
        lexical = [{"memory_id": str(A1), "score": -3.0, "rank": 0},
                   {"memory_id": str(A1), "score": -3.0, "rank": 1}]

        fused = fuse_by_uuid([], lexical, k=60)

        assert len(fused) == 1
        assert fused[0]["score"] == pytest.approx(1 / 61)


# ── recall wiring (R2 / R19) ────────────────────────────────────────────────


async def _recall(factory, monkeypatch, store, *, top_k=3, **kwargs):
    """Run one recall against a real session with the fake dense leg."""
    monkeypatch.setattr(rmod, "search_memories", store)
    kwargs.setdefault("semantic_rerank", False)
    async with factory() as session:
        retriever = MemoryRetriever(session, TENANT_A, **kwargs)
        return await retriever.recall(QUERY, top_k=top_k, include_personal_context=False)


class TestHybridRecall:
    async def test_flag_on_fuses_dense_and_lexical_by_rrf(self, offline_env, recall_db, monkeypatch):
        _, factory = recall_db
        await _seed(factory,
                    _mem(A1, TENANT_A, MATCHING),   # lexical rank 0 (id asc)
                    _mem(A2, TENANT_A, MATCHING),   # dense rank 0 + lexical rank 1
                    _mem(A3, TENANT_A, UNMATCHED))  # dense rank 1, no lexical match
        store = _Store([(A2, 0.90), (A3, 0.80)])

        response = await _recall(factory, monkeypatch, store, top_k=3, hybrid=True)

        # A2 (both legs) > A1 (lexical only) > A3 (dense only).
        assert [result.id for result in response.results] == [A2, A1, A3]
        scores = {result.id: result.score for result in response.results}
        assert scores[A2] == pytest.approx(_rrf(0, 1) * DECAY_MULT, abs=1e-6)
        assert scores[A1] == pytest.approx(_rrf(0) * DECAY_MULT, abs=1e-6)
        assert scores[A3] == pytest.approx(_rrf(1) * DECAY_MULT, abs=1e-6)
        # C3: the three legs that ran report their own real numbers.
        counts = response.trace.counts
        assert counts["dense"] == 2 and counts["lexical"] == 2 and counts["fused"] == 3
        assert response.trace.stage_ms["lexical"] > 0.0
        assert response.trace.num_candidates == 3

    async def test_flag_off_is_dense_only_and_never_probes_the_index(
        self, offline_env, recall_db, monkeypatch
    ):
        _, factory = recall_db
        await _seed(factory,
                    _mem(A1, TENANT_A, MATCHING),
                    _mem(A2, TENANT_A, MATCHING),
                    _mem(A3, TENANT_A, UNMATCHED))
        store = _Store([(A2, 0.90), (A3, 0.80)])

        def _tripwire(*_args, **_kwargs):
            raise AssertionError("the OFF path must not touch the lexical index")

        monkeypatch.setattr(lexical_index, "search", _tripwire)
        response = await _recall(factory, monkeypatch, store, top_k=3)  # the shipped default

        assert [result.id for result in response.results] == [A2, A3]  # dense order, byte-for-byte
        scores = {result.id: result.score for result in response.results}
        assert scores[A2] == pytest.approx(0.90 * DECAY_MULT, abs=1e-6)
        assert scores[A3] == pytest.approx(0.80 * DECAY_MULT, abs=1e-6)
        assert "lexical" not in response.trace.counts
        assert "fused" not in response.trace.counts
        assert response.trace.stage_ms["lexical"] == 0.0
        assert response.trace.num_candidates == 2

    async def test_custom_rrf_k_moves_the_fused_scores(self, offline_env, recall_db, monkeypatch):
        _, factory = recall_db
        await _seed(factory,
                    _mem(A1, TENANT_A, MATCHING),
                    _mem(A2, TENANT_A, MATCHING),
                    _mem(A3, TENANT_A, UNMATCHED))
        store = _Store([(A2, 0.90), (A3, 0.80)])

        response = await _recall(factory, monkeypatch, store, top_k=3, hybrid=True, rrf_k=1)

        assert [result.id for result in response.results] == [A2, A1, A3]
        scores = {result.id: result.score for result in response.results}
        assert scores[A2] == pytest.approx(_rrf(0, 1, k=1) * DECAY_MULT, abs=1e-6)
        assert scores[A1] == pytest.approx(_rrf(0, k=1) * DECAY_MULT, abs=1e-6)
        assert scores[A3] == pytest.approx(_rrf(1, k=1) * DECAY_MULT, abs=1e-6)

    async def test_lexical_leg_filters_tenant_and_visibility_before_its_limit(
        self, offline_env, recall_db, monkeypatch
    ):
        """Every row has identical text, so BM25 ties and the FTS tie-break is
        the canonical id ascending: the foreign rows and the superseded row
        hold the smaller ids and would fill a post-LIMIT page of 2. Only a
        pre-LIMIT tenant + visibility filter leaves the tenant's current row
        in the leg — and `counts["lexical"] == 1` is what proves it."""
        _, factory = recall_db
        await _seed(factory,
                    _mem(B1, TENANT_B, MATCHING),
                    _mem(B2, TENANT_B, MATCHING),
                    _mem(B3, TENANT_B, MATCHING),
                    _mem(SUPERSEDED, TENANT_A, MATCHING, superseded=True),
                    _mem(CURRENT, TENANT_A, MATCHING))
        store = _Store([])

        response = await _recall(factory, monkeypatch, store, top_k=1, hybrid=True)

        assert [result.id for result in response.results] == [CURRENT]
        assert response.trace.counts["lexical"] == 1
        assert response.trace.counts["fused"] == 1

    async def test_reranker_sees_only_sql_authorized_text(
        self, offline_env, recall_db, monkeypatch
    ):
        _, factory = recall_db
        await _seed(factory,
                    _mem(A1, TENANT_A, MATCHING),
                    _mem(A2, TENANT_A, MATCHING),
                    _mem(A3, TENANT_A, UNMATCHED))
        store = _Store([(A2, 0.90), (A3, 0.80)])
        seen: list[dict] = []

        async def _rerank(_query, chunks, *, top_n=None):
            seen.extend(chunks)
            return [dict(chunk, rerank_score=1.0) for chunk in chunks]

        monkeypatch.setattr(reranker_module, "rerank", _rerank)

        response = await _recall(
            factory, monkeypatch, store, top_k=3, hybrid=True, semantic_rerank=True
        )

        assert len(seen) == 3, "the whole fused pool reaches the reranker"
        authorized = {str(A1): MATCHING, str(A2): MATCHING, str(A3): UNMATCHED}
        for chunk in seen:
            assert chunk["content"] == authorized[chunk["memory_id"]]
            assert VECTOR_COPY not in chunk["content"]
        assert {result.id for result in response.results} == {A1, A2, A3}

    async def test_vector_outage_answers_from_the_lexical_leg(self, offline_env, recall_db, monkeypatch):
        """R19: no flag here — the fallback only ever replaces a typed 503."""
        _, factory = recall_db
        await _seed(factory,
                    _mem(A1, TENANT_A, MATCHING),
                    _mem(A2, TENANT_A, MATCHING),
                    _mem(A3, TENANT_A, UNMATCHED))
        reset_fallback_counts()

        response = await _recall(factory, monkeypatch, _Store([], outage=True), top_k=3)

        assert [result.id for result in response.results] == [A1, A2]  # BM25 order
        counts = response.trace.counts
        assert counts["lexical"] == 2
        assert "dense" not in counts, "the dense leg never answered"
        assert "fused" not in counts, "a single leg is not a fusion"
        assert response.trace.stage_ms["lexical"] > 0.0
        assert fallback_counts()["retrieval.vector_unavailable"] == 1

    async def test_vector_outage_without_fts_keeps_the_typed_503(self, offline_env, monkeypatch):
        reset_fallback_counts()
        session = _PostgresishSession()
        retriever = MemoryRetriever(session, TENANT_A, hybrid=True)
        monkeypatch.setattr(rmod, "search_memories", _Store([], outage=True))
        with pytest.raises(VectorUnavailableError):
            await retriever.recall(QUERY, top_k=3, include_personal_context=False)
        assert "retrieval.vector_unavailable" not in fallback_counts(), (
            "no BM25-only answer happened, so the counter must stay silent"
        )

    async def test_hybrid_on_a_postgres_deployment_degrades_to_dense_only(
        self, offline_env, monkeypatch
    ):
        page = [_mem(A2, TENANT_A, MATCHING), _mem(A3, TENANT_A, UNMATCHED)]
        session = _PostgresishSession(page)
        monkeypatch.setattr(rmod, "search_memories", _Store([(A2, 0.90), (A3, 0.80)]))

        response = await MemoryRetriever(session, TENANT_A, hybrid=True).recall(
            QUERY, top_k=3, include_personal_context=False
        )

        assert [result.id for result in response.results] == [A2, A3]
        assert "lexical" not in response.trace.counts
        assert "fused" not in response.trace.counts
        assert response.trace.stage_ms["lexical"] == 0.0
