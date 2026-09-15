"""Visibility: one SQL rule mirrors ``correction.state_of`` for every reader.

Isolated by construction: a private per-test SQLite file on ``tmp_path``, bound
in as the module engines / sessionmakers, so this suite can never read or write
whatever ``DATABASE_URL`` is ambient (pattern:
``tests/retrieval/test_index_outbox.py``). Real file, real rows, real SQL.

The cross-check is the point: the SQL predicate and the SELECT-side state label
must agree 1:1 with the Python authority ``state_of`` on fixture rows covering
all four states, a two-key precedence row, and an empty ``extra_metadata``.
"""
from __future__ import annotations

import json
import math
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import create_engine, event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from app import database
from app.api.v1.memories import list_memories
from app.config import settings
from app.database import Base
from app.mcp_hub import tools as hub_tools
from app.mcp_hub.identity import AgentPrincipal
from app.models.memory import Memory
from app.models.memory_access_log import MemoryAccessLog
from app.models.user import User
from app.observability.fallbacks import fallback_counts, reset_fallback_counts
from app.retrieval.memory import retriever as rmod
from app.retrieval.memory import vector_store
from app.retrieval.memory.context import fetch_personal_context
from app.retrieval.memory.correction import (
    CM_DERIVED_DIRTY,
    CM_NEEDS_CHECK,
    CM_SUPERSEDED_BY,
    state_of,
)
from app.retrieval.memory.outbox import IndexFreshnessTimeout
from app.retrieval.memory.reindex import reindex_user_memories_sync
from app.retrieval.memory.visibility import (
    current_memory_predicate,
    state_expression,
)
from app.retrieval.vector_retriever import VectorUnavailableError

VIS_DB = "visibility.sqlite"


@pytest_asyncio.fixture
async def db(tmp_path, monkeypatch):
    """A private per-test SQLite file, used by both the async and sync twins."""
    import app.models  # noqa: F401 — register every ORM table on Base

    url = f"sqlite+aiosqlite:///{tmp_path / VIS_DB}"
    eng = create_async_engine(
        url, connect_args={"check_same_thread": False}, poolclass=NullPool
    )
    event.listen(eng.sync_engine, "connect", database._configure_sqlite_connection)
    sync_eng = create_engine(
        url.replace("+aiosqlite", ""),
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )
    event.listen(sync_eng, "connect", database._configure_sqlite_connection)
    sessions = async_sessionmaker(eng, class_=AsyncSession, expire_on_commit=False, autoflush=False)

    monkeypatch.setattr(database, "engine", eng)
    monkeypatch.setattr(database, "AsyncSessionLocal", sessions)
    monkeypatch.setattr(
        database,
        "_get_sync_sessionmaker",
        lambda: sessionmaker(bind=sync_eng, expire_on_commit=False, autoflush=False),
    )

    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        async with sessions() as session:
            yield session
    finally:
        await eng.dispose()
        sync_eng.dispose()


async def _owner(db) -> uuid.UUID:
    uid = uuid.uuid4()
    db.add(User(id=uid, email=f"{uid.hex}@test.invalid", hashed_password="x",
                display_name="Owner", is_verified=True, is_active=True))
    await db.commit()
    return uid


def _mem(owner, title: str, *, meta: dict | None = None, minutes: int = 0,
         pinned: bool = False) -> Memory:
    return Memory(
        id=uuid.uuid4(), user_id=owner, title=title, content=f"{title} body",
        tags=[], salience=0.5, pinned=pinned,
        captured_at=datetime.now(UTC) - timedelta(minutes=minutes),
        extra_metadata={} if meta is None else meta,
    )


def _as_reader(monkeypatch, uid: uuid.UUID) -> AgentPrincipal:
    """Point the MCP tools at the temp file with a read-scoped principal."""
    principal = AgentPrincipal(user_id=uid, agent_client_id=uuid.uuid4(), name="Vis",
                               scopes=frozenset({"memory:read"}))
    monkeypatch.setattr(hub_tools, "_current_principal", lambda: principal)
    monkeypatch.setattr(hub_tools, "_session", database.AsyncSessionLocal)
    return principal


def _fixture_rows(owner) -> list[Memory]:
    """One row per state, plus the two-key precedence row and the empty-{} row."""
    return [
        _mem(owner, "plain"),                                                     # {} -> current
        _mem(owner, "annotated", meta={"cm_subject": "proj-x", "cm_attribute": "db"}),
        _mem(owner, "old", meta={CM_SUPERSEDED_BY: "successor"}),                 # superseded
        _mem(owner, "stale-view", meta={CM_DERIVED_DIRTY: True}),                 # dirty
        _mem(owner, "ask-me", meta={CM_NEEDS_CHECK: True}),                       # needs-check
        _mem(owner, "both", meta={CM_SUPERSEDED_BY: "successor",
                                  CM_DERIVED_DIRTY: True, CM_NEEDS_CHECK: True}),
    ]


# ── Task 6: MCP search through the shared recall seam ───────────────────────

# Ten years back: every row decays to the same floor (0.1), so the served order
# is the recall's own (dense rank x 0.5 + salience) and never clock noise.
AGED = datetime.now(UTC) - timedelta(days=3650)


def _aged(owner, title: str, *, salience: float, meta: dict | None = None) -> Memory:
    m = _mem(owner, title, meta=meta)
    m.salience = salience
    m.captured_at = AGED
    return m


class _Dense:
    """The dense leg: fixed ``(memory_id, score)`` rows best first — or an
    outage. ``user_id`` is deliberately IGNORED: the real store filters by
    tenant, so a page here can carry a foreign row and prove the SQL
    authorization (not the index) is what keeps it out."""

    def __init__(self, rows=(), *, outage: bool = False):
        self.rows = list(rows)
        self.outage = outage
        self.calls: list[int] = []

    async def __call__(self, _embedding, *, user_id, top_k=10, where=None):
        self.calls.append(top_k)
        if self.outage:
            raise VectorUnavailableError("vector store down")
        return [
            {"memory_id": str(mid), "content": "STALE VECTOR COPY", "score": score}
            for mid, score in self.rows[:top_k]
        ]


@pytest.fixture
def recall_seams(monkeypatch, barrier_outbox):
    """Every out-of-process seam of recall faked (rewrite, embed, personal
    context, vector search) so the MCP search path runs the REAL retriever —
    barrier included, through ``tests/retrieval/conftest.py``'s outbox."""

    async def _rewrite(query, context=None):
        return {"rewritten_query": query, "entities": [], "reasoning": None,
                "_fallback_used": False}

    async def _embed(_query):
        return [0.1, 0.2]

    monkeypatch.setattr(rmod, "rewrite_query", _rewrite)
    monkeypatch.setattr(rmod, "embed_query", _embed)
    monkeypatch.setattr(rmod, "fetch_personal_context", AsyncMock(return_value=[]))


def _serve(monkeypatch, store: _Dense) -> None:
    monkeypatch.setattr(rmod, "search_memories", store)


# ── cross-check: SQL predicate / label vs the Python authority ───────────────


async def test_sql_predicate_matches_state_of_on_fixtures(db):
    owner = await _owner(db)
    rows = _fixture_rows(owner)
    db.add_all(rows)
    await db.commit()

    # The fixture really covers all four states (a new state cannot slip by).
    assert {state_of(m) for m in rows} == {"current", "superseded", "dirty", "needs-check"}

    selected = set((await db.execute(
        select(Memory.id).where(Memory.user_id == owner, current_memory_predicate())
    )).scalars().all())

    # Visible iff state_of is neither superseded nor dirty — needs-check rows
    # stay visible (labeled), superseded rows are history, dirty rows are wrong.
    assert selected == {m.id for m in rows if state_of(m) not in ("superseded", "dirty")}
    assert rows[0].id in selected      # extra_metadata == {} reads current
    assert rows[5].id not in selected  # superseded outranks dirty


async def test_state_expression_matches_state_of_on_fixtures(db):
    owner = await _owner(db)
    rows = _fixture_rows(owner)
    db.add_all(rows)
    await db.commit()

    labeled = dict((await db.execute(
        select(Memory.id, state_expression()).where(Memory.user_id == owner)
    )).all())

    assert labeled == {m.id: state_of(m) for m in rows}
    # Precedence is the part a naive OR would get wrong.
    assert labeled[rows[5].id] == "superseded"
    assert labeled[rows[4].id] == "needs-check"


# ── readers ──────────────────────────────────────────────────────────────────


async def test_personal_context_excludes_stale_rows(db):
    """The cap applies AFTER the filter — stale rows must not eat the 30 slots."""
    owner = await _owner(db)
    keeper = _mem(owner, "keeper", minutes=90)
    pinned_stale = _mem(owner, "pinned stale", meta={CM_DERIVED_DIRTY: True},
                        minutes=120, pinned=True)
    stale = [
        _mem(owner, f"stale {i}",
             meta={CM_SUPERSEDED_BY: "s"} if i % 2 else {CM_DERIVED_DIRTY: True},
             minutes=i)
        for i in range(40)
    ]
    db.add_all([keeper, pinned_stale, *stale])
    await db.commit()

    out = await fetch_personal_context(db, owner)

    # 40 newer stale rows + a stale pinned row: the one current row must survive
    # the cap (the pre-fix query capped a superset and dropped it).
    assert [m.id for m in out] == [keeper.id]


async def test_rest_list_hides_dirty_keeps_superseded_labeled(db):
    owner = await _owner(db)
    current = _mem(owner, "current", minutes=1)
    superseded = _mem(owner, "superseded", meta={CM_SUPERSEDED_BY: "s"}, minutes=2)
    needs = _mem(owner, "needs-check", meta={CM_NEEDS_CHECK: True}, minutes=3)
    dirty = _mem(owner, "dirty", meta={CM_DERIVED_DIRTY: True}, minutes=4)
    db.add_all([current, superseded, needs, dirty])
    await db.commit()

    # FastAPI's Query() defaults are unresolved when the endpoint is called
    # directly, so every filter is passed explicitly.
    resp = await list_memories(
        SimpleNamespace(id=owner), db, source_type=None, tag=None, query=None,
        pinned=None, sort="newest", limit=50, offset=0,
    )

    assert [i.id for i in resp.items] == [current.id, superseded.id, needs.id]
    assert resp.total == 3  # the count hides dirty too, not just the page
    assert {i.id: i.state for i in resp.items} == {
        current.id: "current",
        superseded.id: "superseded",
        needs.id: "needs-check",
    }


async def test_mcp_list_recent_labels_state_and_hides_dirty(db, monkeypatch):
    owner = await _owner(db)
    current = _mem(owner, "current", minutes=1)
    superseded = _mem(owner, "superseded", meta={CM_SUPERSEDED_BY: "s"}, minutes=2)
    needs = _mem(owner, "needs-check", meta={CM_NEEDS_CHECK: True}, minutes=3)
    dirty = _mem(owner, "dirty", meta={CM_DERIVED_DIRTY: True}, minutes=4)
    db.add_all([current, superseded, needs, dirty])
    await db.commit()
    _as_reader(monkeypatch, owner)

    out = await hub_tools.list_recent(limit=10)

    assert [r["id"] for r in out["results"]] == [str(current.id), str(superseded.id),
                                                str(needs.id)]
    assert [r["state"] for r in out["results"]] == ["current", "superseded", "needs-check"]


async def test_mcp_search_hides_dirty_even_with_history(db, monkeypatch):
    owner = await _owner(db)
    current = _mem(owner, "current", minutes=1)
    superseded = _mem(owner, "superseded", meta={CM_SUPERSEDED_BY: "s"}, minutes=2)
    dirty = _mem(owner, "dirty", meta={CM_DERIVED_DIRTY: True}, minutes=3)
    db.add_all([current, superseded, dirty])
    await db.commit()
    _as_reader(monkeypatch, owner)

    async def _recall(_query, _limit):
        return [(m.id, 0.9) for m in (current, superseded, dirty)]

    monkeypatch.setattr(hub_tools, "_recall_memory_ids", _recall)

    out = await hub_tools.search_memory("body")
    assert [r["id"] for r in out["results"]] == [str(current.id)]

    # History widens to superseded, never to dirty.
    history = await hub_tools.search_memory("body", include_history=True)
    assert {r["id"] for r in history["results"]} == {str(current.id), str(superseded.id)}
    assert {r["state"] for r in history["results"]} == {"current", "superseded"}


async def test_mcp_search_dirty_rows_do_not_consume_candidate_slots(
    db, monkeypatch, recall_seams
):
    """The candidate cap applies AFTER the dirty filter — inside the shared
    recall's eligibility step (T6): a dirty row that ranks top cannot crowd a
    current row out of the capped slots (pre-fix this returned 1 row for
    ``limit=3``)."""
    owner = await _owner(db)
    current = [_aged(owner, f"current {i}", salience=0.5) for i in range(3)]
    dirty = [_aged(owner, f"dirty {i}", salience=0.9, meta={CM_DERIVED_DIRTY: True})
             for i in range(2)]
    db.add_all([*current, *dirty])
    await db.commit()
    _as_reader(monkeypatch, owner)
    _serve(monkeypatch, _Dense([(m.id, 0.95) for m in dirty]
                               + [(m.id, 0.5) for m in current]))  # dirty rank first

    out = await hub_tools.search_memory("body", limit=3)  # the real seam

    assert len(out["results"]) == 3  # honest count: no slot eaten by a dirty row
    assert [r["id"] for r in out["results"]] == [str(m.id) for m in current]
    assert all(r["state"] == "current" for r in out["results"])


async def test_mcp_search_recall_keeps_superseded_for_history(
    db, monkeypatch, recall_seams
):
    """Superseded rows stay ELIGIBLE in the recall ordering (it asks for them;
    R11(p2)); the ``include_history=False`` path still excludes them, dirty
    rows are never served either way."""
    owner = await _owner(db)
    current = _aged(owner, "current", salience=0.5)
    superseded = _aged(owner, "superseded", salience=0.9,
                       meta={CM_SUPERSEDED_BY: "s"})
    dirty = _aged(owner, "dirty", salience=0.99, meta={CM_DERIVED_DIRTY: True})
    db.add_all([current, superseded, dirty])
    await db.commit()
    _as_reader(monkeypatch, owner)
    _serve(monkeypatch, _Dense([(dirty.id, 0.99), (superseded.id, 0.90),
                                (current.id, 0.50)]))

    plain = await hub_tools.search_memory("body")  # the real seam
    assert [r["id"] for r in plain["results"]] == [str(current.id)]

    history = await hub_tools.search_memory("body", include_history=True)
    # Widened, in the recall's own rank order (superseded outscored current).
    assert [r["id"] for r in history["results"]] == [str(superseded.id), str(current.id)]
    assert [r["state"] for r in history["results"]] == ["superseded", "current"]


# ── Task 6: ranking + payload + ledger through the semantic path ────────────


async def test_mcp_search_ranks_by_the_shared_recall_semantics(
    db, monkeypatch, recall_seams
):
    """R22(p2): with a query, the SEMANTIC order wins over salience.

    The fixture makes the two orders disagree by construction — the dense rank
    0 row carries salience 0.05, the rank 1 row 0.99 — so a SQL (salience)
    ordering would serve the exact reverse.
    """
    owner = await _owner(db)
    semantic = _aged(owner, "semantic winner", salience=0.05)
    salient = _aged(owner, "salience winner", salience=0.99)
    db.add_all([semantic, salient])
    await db.commit()
    _as_reader(monkeypatch, owner)
    store = _Dense([(semantic.id, 0.90), (salient.id, 0.20)])
    _serve(monkeypatch, store)

    out = await hub_tools.search_memory("semantic query")

    assert [r["id"] for r in out["results"]] == [str(semantic.id), str(salient.id)]
    # The pool for the tool's default limit=8, through the retriever's own
    # rule — derived from settings, never hard-bound to 2.0 x 8.
    pool = max(8, math.ceil(8 * settings.RETRIEVAL_RERANK_POOL_MULTIPLIER))
    assert store.calls == [pool], "the retriever's pool, not a SQL select"


async def test_mcp_search_tenant_isolated_even_when_foreign_ranks_best(
    db, monkeypatch, recall_seams
):
    """A foreign row at the top of the page is never served: the recall's
    hydrate (and the tool's owner re-check) authorize from SQL."""
    owner = await _owner(db)
    other = await _owner(db)
    mine = _aged(owner, "mine", salience=0.5)
    foreign = _aged(other, "foreign", salience=0.5)
    db.add_all([mine, foreign])
    await db.commit()
    _as_reader(monkeypatch, owner)
    _serve(monkeypatch, _Dense([(foreign.id, 0.99), (mine.id, 0.50)]))

    out = await hub_tools.search_memory("body")

    assert [r["id"] for r in out["results"]] == [str(mine.id)]


async def test_mcp_search_payload_and_ledger_unchanged(db, monkeypatch, recall_seams):
    """R11(p2): index-only rows (160-char snippet, no body), one ledger row."""
    owner = await _owner(db)
    memory = _aged(owner, "shape", salience=0.5)
    memory.content = "x" * 400
    db.add(memory)
    await db.commit()
    _as_reader(monkeypatch, owner)
    _serve(monkeypatch, _Dense([(memory.id, 0.9)]))

    out = await hub_tools.search_memory("body")

    row = out["results"][0]
    assert set(row) == {"id", "title", "snippet", "tags", "salience",
                        "captured_at", "state"}
    assert len(row["snippet"]) == 161  # 160 + ellipsis
    assert "x" * 200 not in json.dumps(out)  # no full body anywhere in the payload

    async with database.AsyncSessionLocal() as fresh:
        ledger = (await fresh.execute(
            select(MemoryAccessLog).where(
                MemoryAccessLog.user_id == owner,
                MemoryAccessLog.action == "mcp_search",
            )
        )).scalars().all()
    assert len(ledger) == 1
    assert ledger[0].detail == {"query": "body", "returned": 1, "memory_ids": [str(memory.id)]}


async def test_mcp_search_limit_is_capped_at_the_tool_max(db, monkeypatch, recall_seams):
    """The ``MAX_SEARCH_LIMIT=20`` cap survives the wiring — and the ordering is
    the recall's down to the last served row."""
    owner = await _owner(db)
    rows = [_aged(owner, f"m{i}", salience=0.5) for i in range(25)]
    db.add_all(rows)
    await db.commit()
    _as_reader(monkeypatch, owner)
    _serve(monkeypatch, _Dense([(m.id, 1.0 - i / 100.0) for i, m in enumerate(rows)]))

    out = await hub_tools.search_memory("body", limit=999)

    assert len(out["results"]) == hub_tools.MAX_SEARCH_LIMIT
    assert [r["id"] for r in out["results"]] == [str(m.id) for m in rows[:20]]


# ── Task 6 / R23: the typed readiness errors fall back, never a tool error ──


async def test_mcp_search_barrier_timeout_falls_back_to_sql_ordering(
    db, monkeypatch, recall_seams
):
    """R23(p2): MCP gains no 503 semantics — a barrier timeout answers from the
    SQL ordering (salience desc), logged and counted, ledger unchanged."""
    owner = await _owner(db)
    salient = _aged(owner, "salient", salience=0.99)
    quiet = _aged(owner, "quiet", salience=0.10)
    db.add_all([salient, quiet])
    await db.commit()
    _as_reader(monkeypatch, owner)

    async def _timeout(**_kwargs):
        raise IndexFreshnessTimeout("pending writes")

    monkeypatch.setattr(rmod, "await_freshness", _timeout)
    _serve(monkeypatch, _Dense([]))  # never reached: the barrier raises first
    reset_fallback_counts()

    out = await hub_tools.search_memory("body")

    assert "error" not in out
    assert [r["id"] for r in out["results"]] == [str(salient.id), str(quiet.id)]
    assert fallback_counts()[hub_tools.SQL_FALLBACK_PATH] == 1


async def test_mcp_search_vector_outage_falls_back_to_sql_ordering(
    db, monkeypatch, recall_seams
):
    """R23(p2): this deployment has no lexical index (the temp DB never ran the
    v4 ladder), so the recall's vector outage is the API's typed 503 — at the
    MCP boundary it becomes the SQL ordering instead."""
    owner = await _owner(db)
    salient = _aged(owner, "salient", salience=0.90)
    quiet = _aged(owner, "quiet", salience=0.20)
    db.add_all([salient, quiet])
    await db.commit()
    _as_reader(monkeypatch, owner)
    _serve(monkeypatch, _Dense([], outage=True))
    reset_fallback_counts()

    out = await hub_tools.search_memory("body")

    assert "error" not in out
    assert [r["id"] for r in out["results"]] == [str(salient.id), str(quiet.id)]
    assert fallback_counts()[hub_tools.SQL_FALLBACK_PATH] == 1


async def test_mcp_search_fallback_serves_history_from_the_sql_order(
    db, monkeypatch, recall_seams
):
    """M1: the fallback x ``include_history=True`` interaction — the widened
    superseded row is served (history is the caller's ask) and it ranks by its
    SALIENCE, because the served order is the SQL one; dirty rows never widen
    in, on either call."""
    owner = await _owner(db)
    current = _aged(owner, "current", salience=0.30)
    superseded = _aged(owner, "superseded", salience=0.95, meta={CM_SUPERSEDED_BY: "s"})
    dirty = _aged(owner, "dirty", salience=0.99, meta={CM_DERIVED_DIRTY: True})
    db.add_all([current, superseded, dirty])
    await db.commit()
    _as_reader(monkeypatch, owner)

    async def _timeout(**_kwargs):
        raise IndexFreshnessTimeout("pending writes")

    monkeypatch.setattr(rmod, "await_freshness", _timeout)
    _serve(monkeypatch, _Dense([]))  # never reached: the barrier raises first
    reset_fallback_counts()

    plain = await hub_tools.search_memory("body")
    assert [r["id"] for r in plain["results"]] == [str(current.id)]

    history = await hub_tools.search_memory("body", include_history=True)
    assert [r["id"] for r in history["results"]] == [str(superseded.id), str(current.id)]
    assert [r["state"] for r in history["results"]] == ["superseded", "current"]
    assert fallback_counts()[hub_tools.SQL_FALLBACK_PATH] == 2


async def test_mcp_search_embed_outage_falls_back_to_sql_ordering(
    db, monkeypatch, recall_seams
):
    """R25(p2): the embed leg's outage is a degraded leg, not a no-match — the
    seam says WHY the order is empty and the tool answers from the SQL
    ordering instead of a confident ``results: []``."""
    owner = await _owner(db)
    salient = _aged(owner, "salient", salience=0.90)
    quiet = _aged(owner, "quiet", salience=0.20)
    db.add_all([salient, quiet])
    await db.commit()
    _as_reader(monkeypatch, owner)

    async def _embed_down(_query):
        raise ValueError("Failed to get embeddings: provider unreachable")

    monkeypatch.setattr(rmod, "embed_query", _embed_down)
    # Never reached: with no query vector the dense leg cannot run, and the
    # page below would serve quiet first if it somehow did.
    _serve(monkeypatch, _Dense([(quiet.id, 0.99)]))
    reset_fallback_counts()

    out = await hub_tools.search_memory("body")

    assert "error" not in out
    assert [r["id"] for r in out["results"]] == [str(salient.id), str(quiet.id)]
    assert fallback_counts()[hub_tools.SQL_FALLBACK_PATH] == 1


async def test_mcp_search_store_failure_falls_back_to_sql_ordering(
    db, monkeypatch, recall_seams
):
    """R25(p2): the untyped ``search_memories`` failure gets the same shape —
    an infrastructure failure must not be served as a no-match either."""
    owner = await _owner(db)
    salient = _aged(owner, "salient", salience=0.90)
    quiet = _aged(owner, "quiet", salience=0.20)
    db.add_all([salient, quiet])
    await db.commit()
    _as_reader(monkeypatch, owner)

    async def _boom(_embedding, *, user_id, top_k=10, where=None):
        raise RuntimeError("pgvector connection reset")

    monkeypatch.setattr(rmod, "search_memories", _boom)
    reset_fallback_counts()

    out = await hub_tools.search_memory("body")

    assert "error" not in out
    assert [r["id"] for r in out["results"]] == [str(salient.id), str(quiet.id)]
    assert fallback_counts()[hub_tools.SQL_FALLBACK_PATH] == 1


async def test_mcp_search_genuine_no_match_stays_empty(db, monkeypatch, recall_seams):
    """The discriminator is the leg state, never an empty list: the real
    retriever on a healthy dense leg that matched nothing returns ``[]`` and
    the SQL ordering is NOT consulted (the row below would win any SQL order)."""
    owner = await _owner(db)
    row = _aged(owner, "unrelated", salience=0.99)
    db.add(row)
    await db.commit()
    _as_reader(monkeypatch, owner)
    store = _Dense([])  # the dense leg ran and matched nothing
    _serve(monkeypatch, store)
    reset_fallback_counts()

    out = await hub_tools.search_memory("body")

    assert store.calls, "the healthy dense leg really ran"
    assert out["results"] == []
    assert fallback_counts().get(hub_tools.SQL_FALLBACK_PATH, 0) == 0


async def test_mcp_timeline_labels_every_row(db, monkeypatch):
    owner = await _owner(db)
    older = _mem(owner, "older", meta={CM_SUPERSEDED_BY: "s"}, minutes=2)
    anchor = _mem(owner, "anchor", minutes=1)
    newer = _mem(owner, "newer", meta={CM_DERIVED_DIRTY: True}, minutes=0)
    db.add_all([older, anchor, newer])
    await db.commit()
    _as_reader(monkeypatch, owner)

    out = await hub_tools.timeline(memory_id=str(anchor.id))

    # Timeline is history: it serves every row, each labeled with its state.
    assert out["anchor"]["state"] == "current"
    assert [r["state"] for r in out["before"]] == ["superseded"]
    assert [r["state"] for r in out["after"]] == ["dirty"]


async def test_reindex_indexes_current_rows_only(db, monkeypatch):
    owner = await _owner(db)
    current = _mem(owner, "current", minutes=1)
    needs = _mem(owner, "needs-check", meta={CM_NEEDS_CHECK: True}, minutes=2)
    superseded = _mem(owner, "superseded", meta={CM_SUPERSEDED_BY: "s"}, minutes=3)
    dirty = _mem(owner, "dirty", meta={CM_DERIVED_DIRTY: True}, minutes=4)
    db.add_all([current, needs, superseded, dirty])
    await db.commit()

    indexed: list[str] = []

    def _capture(rows):
        indexed.extend(str(m.id) for m in rows)
        return len(rows)

    monkeypatch.setattr(vector_store, "upsert_memories_sync", _capture)

    summary = reindex_user_memories_sync(str(owner), only_missing=False)

    # needs-check is visible (labeled), so it is indexed; superseded/dirty are not.
    assert set(indexed) == {str(current.id), str(needs.id)}
    assert summary["reindexed"] == 2
