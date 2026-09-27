"""The recall candidate authorization seam, exercised at every checkpoint.

A candidate crosses the SQL authorization boundary at every place a pool can
change on its way to an answer: the first page, both branches of the bounded
refill, and once more after the network round to the reranker. The same
twelve lines used to be written out at each one, so the seam was four chances
to leak a row it should have dropped — and three of the four had no test that
reached them.

These tests state the rule at each checkpoint, not the shape of the code: drop
what SQL cannot authorize, serve SQL's text rather than the vector copy, and
count only the rows that survived. A fix that consolidates the checkpoints
passes; a fix that deletes one of them fails here.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app import database
from app.database import Base
from app.models.memory import Memory
from app.models.user import User
from app.retrieval import reranker as reranker_module
from app.retrieval.memory import lexical_index, namespaces
from app.retrieval.memory import retriever as rmod
from app.retrieval.memory.retriever import MemoryRetriever

TENANT_A = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
TEAM = "team"
CAPTURED = datetime.now(UTC) - timedelta(days=3650)
QUERY = "alpha charlie"
MATCHING = "alpha charlie"
VECTOR_COPY = "STALE VECTOR COPY — must never leave before SQL authorization"

# Smallest usable ids, so BM25 ties still order by canonical UUID.
A1 = uuid.UUID("11111111-1111-1111-1111-111111111111")
A2 = uuid.UUID("22222222-2222-2222-2222-222222222222")
SUPERSEDED = uuid.UUID("55555555-5555-5555-5555-555555555555")


def _mem(memory_id, content, *, superseded=False, namespace=namespaces.PERSONAL):
    meta: dict = {}
    if superseded:
        meta["cm_superseded_by"] = str(uuid.uuid4())
    return Memory(
        id=memory_id, user_id=TENANT_A, title=None, content=content, tags=[],
        salience=0.5, pinned=False, source_type="manual_note", recall_count=0,
        captured_at=CAPTURED, indexed_at=CAPTURED, updated_at=CAPTURED,
        extra_metadata=meta, namespace=namespace,
    )


class _Store:
    """The dense leg: fixed rows best first, or a different page per call."""

    def __init__(self, *pages, outage=False):
        self.pages = [list(page) for page in pages] or [[]]
        self.outage = outage
        self.calls: list[int] = []

    async def __call__(self, _embedding, *, user_id, top_k=10, where=None,
                       namespace=None):
        page = self.pages[min(len(self.calls), len(self.pages) - 1)]
        self.calls.append(top_k)
        return [
            {"memory_id": str(memory_id), "content": VECTOR_COPY, "score": score}
            for memory_id, score in page[:top_k]
        ]


@pytest_asyncio.fixture
async def recall_db(tmp_path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'authorize.db'}", poolclass=NullPool
    )
    event.listen(engine.sync_engine, "connect", database._configure_sqlite_connection)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(lexical_index.create_index)
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    try:
        yield factory
    finally:
        await engine.dispose()


async def _seed(factory, *memories):
    async with factory() as session:
        for user_id in {memory.user_id for memory in memories}:
            session.add(User(id=user_id, email=f"{user_id}@test.invalid",
                             onboarding_done=True, is_verified=True, is_active=True,
                             is_deleted=False))
        session.add_all(memories)
        await session.commit()


@pytest.fixture
def offline_env(monkeypatch, barrier_outbox):
    async def _rewrite(query, context=None):
        return {"rewritten_query": query, "entities": [], "reasoning": None,
                "_fallback_used": False}

    async def _embed(_query):
        return [0.1, 0.2]

    monkeypatch.setattr(rmod, "rewrite_query", _rewrite)
    monkeypatch.setattr(rmod, "embed_query", _embed)
    monkeypatch.setattr(rmod, "fetch_personal_context", AsyncMock(return_value=[]))


async def _recall(factory, monkeypatch, store, *, top_k, **kwargs):
    monkeypatch.setattr(rmod, "search_memories", store)
    kwargs.setdefault("semantic_rerank", False)
    async with factory() as session:
        retriever = MemoryRetriever(session, TENANT_A, **kwargs)
        return await retriever.recall(QUERY, top_k=top_k, include_personal_context=False)


# ── the checkpoints ────────────────────────────────────────────────────────


class TestAuthorizationCheckpoints:
    async def test_the_first_page_drops_a_superseded_row(
        self, offline_env, recall_db, monkeypatch
    ):
        await _seed(recall_db,
                    _mem(A1, MATCHING),
                    _mem(SUPERSEDED, MATCHING, superseded=True))
        store = _Store([(A1, 0.9), (SUPERSEDED, 0.8)])

        response = await _recall(recall_db, monkeypatch, store, top_k=3)

        assert [r.id for r in response.results] == [A1]

    async def test_the_refill_drops_a_superseded_row(
        self, offline_env, recall_db, monkeypatch
    ):
        """The refill is a checkpoint of its own, on BOTH of its branches: the
        dense-only path appends what it fetched, the hybrid path re-fuses and
        re-authorizes the widened pool. Either way a row the second fetch
        brought back must not join the pool unchecked.

        The refill only fires when the first page FILLED its pool and the SQL
        filter then starved it below top_k. Index-only ghosts do that starving
        — they occupy a page slot and have nothing in SQL to authorize.
        """
        await _seed(recall_db,
                    _mem(A1, MATCHING),
                    _mem(SUPERSEDED, MATCHING, superseded=True))
        ghosts = [uuid.uuid4() for _ in range(4)]
        first_page = [*[(ghost, 0.99 - i / 100.0) for i, ghost in enumerate(ghosts)],
                      (A1, 0.95), (SUPERSEDED, 0.94)]
        # The widened page's new row is the superseded one, at a cosine the
        # first page never offered.
        store = _Store(first_page,
                       [*first_page, (SUPERSEDED, 0.30)])

        dense_only = await _recall(recall_db, monkeypatch, store, top_k=3)
        assert store.calls == [6, 12], "the pool, then ONE refill at top_k * 4"
        # The refill's own counter is 0 here and must stay 0: the only row the
        # widened page added was the superseded one, and `added` counts what
        # actually joined the pool.
        assert dense_only.trace.counts["refill"] == 0
        assert SUPERSEDED not in {r.id for r in dense_only.results}

        store.calls.clear()
        hybrid = await _recall(recall_db, monkeypatch, store, top_k=3, hybrid=True)
        assert store.calls == [6, 12], "the pool, then ONE refill at top_k * 4"
        assert hybrid.trace.counts["refill"] == 0
        assert SUPERSEDED not in {r.id for r in hybrid.results}

    async def test_the_hydrated_counter_counts_rows_sql_resolved_not_rows_served(
        self, offline_env, recall_db, monkeypatch
    ):
        """`counts["hydrated"]` is the T3 trace number: how many rows SQL
        resolved for the pool. It is a count of the read, NOT of the answer —
        a row SQL resolved and then hid was still resolved, and the trace
        exists to show that gap.

        So the ghosts (nothing in SQL at all) are excluded while the superseded
        row is included, and `eligible` is what narrows afterwards.
        """
        await _seed(recall_db,
                    _mem(A1, MATCHING),
                    _mem(SUPERSEDED, MATCHING, superseded=True))
        ghosts = [uuid.uuid4() for _ in range(4)]
        first_page = [*[(ghost, 0.99 - i / 100.0) for i, ghost in enumerate(ghosts)],
                      (A1, 0.95), (SUPERSEDED, 0.94)]
        # The widened fetch adds only a ghost and a row SQL already had.
        store = _Store(first_page,
                       [*first_page, (uuid.uuid4(), 0.5), (SUPERSEDED, 0.3)])

        response = await _recall(recall_db, monkeypatch, store, top_k=3)

        assert store.calls == [6, 12], "the refill ran"
        # Two rows exist in SQL and both were resolved; only one is servable.
        assert response.trace.counts["hydrated"] == 2
        assert response.trace.counts["eligible"] == 1
        assert [r.id for r in response.results] == [A1]

    async def test_the_post_rerank_refresh_drops_a_row_superseded_mid_flight(
        self, offline_env, recall_db, monkeypatch
    ):
        """SQL moves while the reranker is on the network. The third checkpoint
        exists precisely for that, and it has to be the one that wins."""
        await _seed(recall_db,
                    _mem(A1, MATCHING),
                    _mem(A2, MATCHING))
        store = _Store([(A1, 0.9), (A2, 0.8)])

        async def _rerank(_query, chunks, *, top_n=None):
            # The reranker is the network round: supersede A2 while it is away.
            async with recall_db() as session:
                row = await session.get(Memory, A2)
                row.extra_metadata = {**(row.extra_metadata or {}),
                                     "cm_superseded_by": str(uuid.uuid4())}
                await session.commit()
            return [dict(chunk, rerank_score=1.0) for chunk in chunks]

        monkeypatch.setattr(reranker_module, "rerank", _rerank)

        response = await _recall(recall_db, monkeypatch, store, top_k=3,
                                 semantic_rerank=True)

        assert [r.id for r in response.results] == [A1]

    async def test_every_checkpoint_serves_sql_text_never_the_vector_copy(
        self, offline_env, recall_db, monkeypatch
    ):
        """One rule at all three: the row that leaves recall carries the
        document SQL owns, not the payload the index shipped."""
        await _seed(recall_db,
                    _mem(A1, MATCHING),
                    _mem(A2, MATCHING))
        store = _Store([(A1, 0.9)], [(A1, 0.9), (A2, 0.8)])
        seen: list[dict] = []

        async def _rerank(_query, chunks, *, top_n=None):
            seen.extend(chunks)
            return [dict(chunk, rerank_score=1.0) for chunk in chunks]

        monkeypatch.setattr(reranker_module, "rerank", _rerank)

        response = await _recall(recall_db, monkeypatch, store, top_k=2,
                                 semantic_rerank=True)

        for chunk in seen:
            assert chunk["content"] == MATCHING
            assert VECTOR_COPY not in chunk["content"]
        for result in response.results:
            assert VECTOR_COPY not in result.content

    def test_the_checkpoint_checks_the_namespace_itself_not_just_the_query(self):
        """The SQL read is tenant- and namespace-scoped, so this second check
        can only fire when that query is wrong. It stays anyway: it is the one
        rule that holds when the row arrived some other way, and a pool
        assembled from an index must not be one query away from a leak.

        A row the index proposes but SQL would never return is exactly what
        this guards — so the test hands the checkpoint one directly.
        """
        retriever = MemoryRetriever.__new__(MemoryRetriever)
        retriever.namespace = namespaces.PERSONAL

        assert retriever._authorize(
            [{"memory_id": str(A1), "content": VECTOR_COPY}],
            {str(A1): _mem(A1, MATCHING, namespace=TEAM)},
            include_superseded=False,
        ) == [], "a row from another namespace never leaves the checkpoint"

    def test_a_candidate_sql_cannot_find_is_dropped_not_served(self):
        """The index proposed an id SQL does not know — deleted, or another
        tenant's. There is nothing to authorize, so it does not leave."""
        retriever = MemoryRetriever.__new__(MemoryRetriever)
        retriever.namespace = namespaces.PERSONAL

        assert retriever._authorize(
            [{"memory_id": str(A1), "content": VECTOR_COPY}], {},
            include_superseded=False,
        ) == []
