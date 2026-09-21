"""Visibility: one SQL rule mirrors ``correction.state_of`` for every reader.

Isolated by construction: a private per-test SQLite file on ``tmp_path``, bound
in as the module engines / sessionmakers, so this suite can never read or write
whatever ``DATABASE_URL`` is ambient (pattern:
``tests/retrieval/test_index_outbox.py``). Real file, real rows, real SQL.

The cross-check is the point: the SQL predicate and the SELECT-side state label
must agree 1:1 with the Python authority ``state_of`` on fixture rows covering
all five states, a two-key precedence row, and an empty ``extra_metadata``.
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
from app.retrieval.memory import lexical_index, namespaces, vector_store
from app.retrieval.memory import retriever as rmod
from app.retrieval.memory.context import fetch_personal_context
from app.retrieval.memory.correction import (
    CM_DERIVED_DIRTY,
    CM_INVALIDATED,
    CM_NEEDS_CHECK,
    CM_SUPERSEDED_BY,
    Slot,
    collect_derived_ids,
    resolve_correction,
    state_of,
)
from app.retrieval.memory.outbox import IndexFreshnessTimeout
from app.retrieval.memory.reindex import reindex_user_memories_sync
from app.retrieval.memory.retriever import MemoryRetriever
from app.retrieval.memory.salience import bump_salience
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
         pinned: bool = False, namespace: str = namespaces.PERSONAL) -> Memory:
    return Memory(
        id=uuid.uuid4(), user_id=owner, title=title, content=f"{title} body",
        tags=[], salience=0.5, pinned=pinned,
        captured_at=datetime.now(UTC) - timedelta(minutes=minutes),
        extra_metadata={} if meta is None else meta,
        namespace=namespace,
    )


def _as_reader(monkeypatch, uid: uuid.UUID,
               scopes: frozenset[str] = frozenset({"memory:read"})) -> AgentPrincipal:
    """Point the MCP tools at the temp file with a read-scoped principal."""
    principal = AgentPrincipal(user_id=uid, agent_client_id=uuid.uuid4(), name="Vis",
                               scopes=scopes)
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
        _mem(owner, "invalidated", meta={CM_INVALIDATED: True}),
    ]


# ── Task 6: MCP search through the shared recall seam ───────────────────────

# Ten years back: every row decays to the same floor (0.1), so the served order
# is the recall's own (dense rank x 0.5 + salience) and never clock noise.
AGED = datetime.now(UTC) - timedelta(days=3650)


def _aged(owner, title: str, *, salience: float, meta: dict | None = None,
          namespace: str = namespaces.PERSONAL) -> Memory:
    m = _mem(owner, title, meta=meta, namespace=namespace)
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

    async def __call__(self, _embedding, *, user_id, top_k=10, where=None,
                       namespace=None):
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

    # The fixture really covers all five states (a new state cannot slip by).
    assert {state_of(m) for m in rows} == {"current", "superseded", "dirty", "needs-check", "invalidated"}

    selected = set((await db.execute(
        select(Memory.id).where(Memory.user_id == owner, current_memory_predicate())
    )).scalars().all())

    # Visible iff state_of is neither superseded nor dirty — needs-check rows
    # stay visible (labeled), superseded rows are history, dirty rows are wrong.
    assert selected == {m.id for m in rows if state_of(m) not in ("superseded", "dirty", "invalidated")}
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

    async def _boom(_embedding, *, user_id, top_k=10, where=None, namespace=None):
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


async def test_mcp_timeline_labels_superseded_history_and_hides_dirty(db, monkeypatch):
    """Timeline is a history view: superseded rows stay readable, labelled.
    Dirty rows are wrong data, not history — the neighbours' predicate hides
    them like every other reader's (T4 widened it; before, a dirty neighbour
    was served)."""
    owner = await _owner(db)
    older = _mem(owner, "older", meta={CM_SUPERSEDED_BY: "s"}, minutes=2)
    anchor = _mem(owner, "anchor", minutes=1)
    newer = _mem(owner, "newer", meta={CM_DERIVED_DIRTY: True}, minutes=0)
    db.add_all([older, anchor, newer])
    await db.commit()
    _as_reader(monkeypatch, owner)

    out = await hub_tools.timeline(memory_id=str(anchor.id))

    assert out["anchor"]["state"] == "current"
    assert [r["state"] for r in out["before"]] == ["superseded"]
    assert out["after"] == [], "a dirty row is never served as surrounding context"


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


# ── P4a/T4: the namespace through the MCP tools and the recall internals ─────

TEAM = "team"  # the second namespace P4b brings: seeded directly, never by a client


def _rank(ids):
    async def _recall(_query, _limit):
        return [(mid, 0.9) for mid in ids]

    return _recall


async def test_mcp_search_serves_only_the_callers_namespace(db, monkeypatch):
    """The tool's own hydration query is a namespaced read: the recall seam can
    rank another namespace's id, and the answer still cannot carry it."""
    owner = await _owner(db)
    mine = _mem(owner, "mine", minutes=1)
    theirs = _mem(owner, "theirs", namespace=TEAM, minutes=2)
    db.add_all([mine, theirs])
    await db.commit()
    _as_reader(monkeypatch, owner)
    monkeypatch.setattr(hub_tools, "_recall_memory_ids", _rank([theirs.id, mine.id]))

    out = await hub_tools.search_memory("body")

    assert [r["id"] for r in out["results"]] == [str(mine.id)]


async def test_mcp_search_recall_hydration_is_namespaced(db, monkeypatch, recall_seams):
    """Through the REAL recall seam: the retriever's hydrate authorizes from
    SQL, so a denser foreign-namespace row never reaches the tool."""
    owner = await _owner(db)
    mine = _aged(owner, "mine", salience=0.5)
    theirs = _aged(owner, "theirs", namespace=TEAM, salience=0.5)
    db.add_all([mine, theirs])
    await db.commit()
    _as_reader(monkeypatch, owner)
    _serve(monkeypatch, _Dense([(theirs.id, 0.99), (mine.id, 0.50)]))

    out = await hub_tools.search_memory("body")

    assert [r["id"] for r in out["results"]] == [str(mine.id)]


async def test_retriever_hydrate_drops_out_of_namespace_rows(db):
    """``_hydrate`` is the recall's SQL authorization: a foreign-namespace id
    in the candidate pool resolves to no row."""
    owner = await _owner(db)
    mine = _mem(owner, "mine", minutes=1)
    theirs = _mem(owner, "theirs", namespace=TEAM, minutes=2)
    db.add_all([mine, theirs])
    await db.commit()

    hydrated = await MemoryRetriever(db, owner)._hydrate([mine.id, theirs.id])

    assert set(hydrated) == {str(mine.id)}


async def test_recall_hands_the_dense_leg_the_namespace_explicitly(
    db, monkeypatch, recall_seams
):
    """T3 carried this: ``search_memories`` fails closed on a missing namespace,
    and the caller still passes the authorized one rather than trusting it."""
    owner = await _owner(db)
    row = _aged(owner, "mine", salience=0.5)
    db.add(row)
    await db.commit()
    _as_reader(monkeypatch, owner)
    seen: dict = {}

    async def _store(_embedding, *, user_id, top_k=10, where=None, namespace=None):
        seen["namespace"] = namespace
        return [{"memory_id": str(row.id), "content": "stale", "score": 0.9}]

    monkeypatch.setattr(rmod, "search_memories", _store)

    out = await hub_tools.search_memory("body")

    assert out["results"][0]["id"] == str(row.id)
    assert seen["namespace"] == namespaces.PERSONAL


async def test_context_fetch_gets_the_namespace_explicitly(
    db, monkeypatch, recall_seams
):
    """The personal-context read is a namespaced read, passed explicitly."""
    owner = await _owner(db)
    seen: dict = {}

    async def _context(_db, _user_id, *, namespace=None):
        seen["namespace"] = namespace
        return []

    monkeypatch.setattr(rmod, "fetch_personal_context", _context)
    _serve(monkeypatch, _Dense([]))

    await MemoryRetriever(db, owner).recall("body")

    assert seen["namespace"] == namespaces.PERSONAL


async def test_the_lexical_leg_gets_the_namespace_explicitly(db, monkeypatch):
    """The FTS leg's fail-closed default is never relied on either."""
    owner = await _owner(db)
    seen: dict = {}

    def _search(_conn, _query, *, user_id, limit, namespace=None):
        seen.update(namespace=namespace, limit=limit)
        return []

    monkeypatch.setattr(lexical_index, "is_available", lambda _conn: True)
    monkeypatch.setattr(lexical_index, "search", _search)

    out = await MemoryRetriever(db, owner)._lexical_leg("body", 5)

    assert out == [] and seen["namespace"] == namespaces.PERSONAL


async def test_mcp_sql_fallback_serves_only_the_callers_namespace(
    db, monkeypatch, recall_seams
):
    """R23's SQL ordering is a namespaced read too: a barrier timeout must not
    open the boundary."""
    owner = await _owner(db)
    team_row = _aged(owner, "team salient", namespace=TEAM, salience=0.99)
    mine = _aged(owner, "mine", salience=0.20)
    db.add_all([team_row, mine])
    await db.commit()
    _as_reader(monkeypatch, owner)

    async def _timeout(**_kwargs):
        raise IndexFreshnessTimeout("pending writes")

    monkeypatch.setattr(rmod, "await_freshness", _timeout)
    _serve(monkeypatch, _Dense([]))  # never reached: the barrier raises first
    reset_fallback_counts()

    out = await hub_tools.search_memory("body")

    assert [r["id"] for r in out["results"]] == [str(mine.id)]
    assert fallback_counts()[hub_tools.SQL_FALLBACK_PATH] == 1


async def test_mcp_list_serves_only_the_callers_namespace(db, monkeypatch):
    owner = await _owner(db)
    mine = _mem(owner, "mine", minutes=1)
    theirs = _mem(owner, "theirs", namespace=TEAM, minutes=2)
    db.add_all([mine, theirs])
    await db.commit()
    _as_reader(monkeypatch, owner)

    out = await hub_tools.list_recent(limit=10)

    assert [r["id"] for r in out["results"]] == [str(mine.id)]


async def test_mcp_get_refuses_a_row_outside_the_namespace(db, monkeypatch):
    owner = await _owner(db)
    theirs = _mem(owner, "theirs", namespace=TEAM)
    db.add(theirs)
    await db.commit()
    _as_reader(monkeypatch, owner)

    out = await hub_tools.get_memory(str(theirs.id))

    assert out == {"error": "memory not found"}


async def test_mcp_delete_refuses_a_row_outside_the_namespace(db, monkeypatch):
    owner = await _owner(db)
    theirs = _mem(owner, "theirs", namespace=TEAM)
    db.add(theirs)
    await db.commit()
    _as_reader(monkeypatch, owner, scopes=frozenset({"memory:read", "memory:write"}))
    erased: list = []

    async def _erase(_db, user_id, ids, *, requested_by):
        erased.append(list(ids))
        raise AssertionError("the erasure path is not the boundary check")

    monkeypatch.setattr(hub_tools, "erase_memories", _erase)

    out = await hub_tools.delete_memory(memory_id=str(theirs.id))

    assert out == {"error": "memory not found"}
    assert erased == []
    async with database.AsyncSessionLocal() as fresh:
        assert await fresh.get(Memory, theirs.id) is not None


async def test_mcp_correct_refuses_a_target_outside_the_namespace(db, monkeypatch):
    owner = await _owner(db)
    theirs = _mem(owner, "theirs", namespace=TEAM)
    db.add(theirs)
    await db.commit()
    _as_reader(monkeypatch, owner, scopes=frozenset({"memory:read", "memory:write"}))

    async def _no_index(_memory):
        return False

    monkeypatch.setattr(hub_tools, "index_new_memory", _no_index)

    out = await hub_tools.correct_memory(memory_id=str(theirs.id), content="x")

    assert out == {"error": "memory not found"}


async def test_mcp_timeline_neighbours_stay_in_the_namespace(db, monkeypatch):
    owner = await _owner(db)
    before_mine = _mem(owner, "before mine", minutes=4)
    anchor = _mem(owner, "anchor", minutes=2)
    db.add_all([before_mine,
                _mem(owner, "before team", namespace=TEAM, minutes=3),
                anchor,
                _mem(owner, "after team", namespace=TEAM, minutes=1)])
    await db.commit()
    _as_reader(monkeypatch, owner)

    out = await hub_tools.timeline(memory_id=str(anchor.id), window=4)

    assert [r["id"] for r in out["before"]] == [str(before_mine.id)]
    assert out["after"] == []


async def test_mcp_timeline_anchor_outside_the_namespace_refused(db, monkeypatch):
    owner = await _owner(db)
    theirs = _mem(owner, "theirs", namespace=TEAM, minutes=1)
    db.add(theirs)
    await db.commit()
    _as_reader(monkeypatch, owner)

    out = await hub_tools.timeline(memory_id=str(theirs.id))

    assert out == {"error": "memory not found"}


async def test_personal_context_excludes_other_namespace_rows(db):
    owner = await _owner(db)
    keeper = _mem(owner, "keeper", minutes=5)
    db.add_all([keeper, _mem(owner, "theirs", namespace=TEAM, minutes=1)])
    await db.commit()

    out = await fetch_personal_context(db, owner)

    assert [m.id for m in out] == [keeper.id]


async def test_bump_salience_skips_rows_outside_the_namespace(db):
    owner = await _owner(db)
    theirs = _mem(owner, "theirs", namespace=TEAM)
    db.add(theirs)
    await db.commit()

    updated = await bump_salience(db, owner, [theirs.id])

    assert updated == 0
    await db.refresh(theirs)
    assert theirs.recall_count == 0 and theirs.salience == 0.5


async def test_a_correction_never_reaches_another_namespace(db):
    """The correction path's candidate read carries the boundary too (the 7th
    file of this class, T4): a corrected fact supersedes/dirties rows of its OWN
    namespace only — the decision is made over rows it may write."""
    owner = await _owner(db)
    stale = _mem(owner, "team fact", namespace=TEAM,
                 meta={"cm_subject": "proj-x", "cm_attribute": "db",
                       "cm_scope": "prod"})
    db.add(stale)
    await db.commit()

    out = await resolve_correction(db, user_id=owner, title="DB", content="SQLite",
                                   slot=Slot.of("proj-x", "db", "prod"))

    assert out["status"] != "superseded"
    assert out["superseded"] == []
    await db.refresh(stale)
    assert stale.extra_metadata.get(CM_SUPERSEDED_BY) is None


async def test_the_derived_closure_is_namespaced(db):
    """The erasure closure walks one namespace (the same 7th file, T4): a team
    row deriving from an erased personal row is not collected — it belongs to
    another boundary."""
    owner = await _owner(db)
    source = _mem(owner, "source", minutes=1)
    mine = _mem(owner, "mine", meta={"cm_derived_from": [str(source.id)]}, minutes=2)
    theirs = _mem(owner, "theirs", namespace=TEAM,
                  meta={"cm_derived_from": [str(source.id)]}, minutes=3)
    db.add_all([source, mine, theirs])
    await db.commit()

    out = await collect_derived_ids(db, owner, [source.id])

    assert [str(mid) for mid in out] == [str(mine.id)]


async def test_reindex_indexes_only_the_callers_namespace(db, monkeypatch):
    owner = await _owner(db)
    mine = _mem(owner, "mine", minutes=1)
    theirs = _mem(owner, "theirs", namespace=TEAM, minutes=2)
    db.add_all([mine, theirs])
    await db.commit()
    indexed: list[str] = []

    def _capture(rows):
        indexed.extend(str(m.id) for m in rows)
        return len(rows)

    monkeypatch.setattr(vector_store, "upsert_memories_sync", _capture)

    summary = reindex_user_memories_sync(str(owner), only_missing=False)

    assert set(indexed) == {str(mine.id)}
    assert summary["scanned"] == 1


# ── P4a Task 5: the erasure tool, the fallback order, the row-in-hand mirror ──


async def test_mcp_forget_refuses_a_row_outside_the_namespace(db, monkeypatch):
    """R35: ``forget_memory`` is namespace-bounded like get/list/search.

    Reviewer's reproduce: a same-account row in ``team`` was forgotten by a
    ``memory:write`` agent while every other tool answered "not found" for it.
    It is skipped here, and the row, its children and its vector stay put.
    """
    owner = await _owner(db)
    theirs = _mem(owner, "theirs", namespace=TEAM)
    db.add(theirs)
    await db.commit()

    # The tool boundary is pinned on its own: the id must never REACH the
    # forget service. The service refuses the row too (defence in depth), so a
    # pin that only looks at the outcome cannot see this guard — it would stay
    # green with the tool's pre-read deleted.
    reached: list[list[str]] = []
    real_forget = hub_tools.soft_forget

    async def _spy(db_, user_id, memory_ids, *, requested_by):
        reached.append([str(m) for m in memory_ids])
        return await real_forget(db_, user_id, memory_ids, requested_by=requested_by)

    monkeypatch.setattr(hub_tools, "soft_forget", _spy)
    _as_reader(monkeypatch, owner, frozenset({"memory:write"}))

    out = await hub_tools.forget_memory(memory_ids=[str(theirs.id)])

    assert reached == [[]], "the id outside the namespace never reaches the forget service"
    assert out["invalidated"] == 0 and out["skipped"] == 1
    async with database.AsyncSessionLocal() as session:
        row = await session.get(Memory, theirs.id)
        assert row is not None and state_of(row) == "current", "cross-namespace forget (R35)"


async def test_mcp_forget_still_invalidates_the_callers_own_row(db, monkeypatch):
    """The same call on the caller's own namespace is the SOFT forget (P4b).

    The row is kept (provenance survives) and leaves serving; no vector is
    purged here (R37: the payload refresh rides the outbox).
    """
    owner = await _owner(db)
    mine = _mem(owner, "mine")
    db.add(mine)
    await db.commit()

    _as_reader(monkeypatch, owner, frozenset({"memory:write"}))
    out = await hub_tools.forget_memory(memory_ids=[str(mine.id)])

    assert out["invalidated"] == 1 and out["skipped"] == 0
    async with database.AsyncSessionLocal() as session:
        row = await session.get(Memory, mine.id)
        assert row is not None, "soft forget keeps the row"
        assert state_of(row) == "invalidated"


async def test_the_sql_fallback_order_is_namespaced(db, monkeypatch):
    """I2: ``_sql_recall_ids`` pinned DIRECTLY, order included.

    Through ``search_memory`` the hydration re-check hid this predicate: dropping
    it left every test green while the fallback ordering itself served another
    namespace's row first.
    """
    owner = await _owner(db)
    team_row = _aged(owner, "team salient", namespace=TEAM, salience=0.99)
    mine = _aged(owner, "mine", salience=0.20)
    db.add_all([team_row, mine])
    await db.commit()
    principal = _as_reader(monkeypatch, owner)

    out = await hub_tools._sql_recall_ids(principal, 10)

    assert [str(mid) for mid, _score in out] == [str(mine.id)]


def test_the_hidden_mirror_rejects_another_namespace():
    """M1: ``_hidden`` is the second layer (a pool assembled from the index), and
    it is pinned on a row in hand — no query, no hydration, just the mirror."""
    mine = _mem(uuid.uuid4(), "mine")
    theirs = _mem(mine.user_id, "theirs", namespace=TEAM)

    assert rmod._hidden(theirs, namespace=namespaces.PERSONAL) is True
    assert rmod._hidden(theirs, namespace=TEAM) is False
    assert rmod._hidden(mine, namespace=namespaces.PERSONAL) is False
    assert rmod._hidden(mine) is False, (
        "no namespace handed: the lifecycle rule alone (unchanged)")



# ── OCR review fix: falsy markers read the same on both sides ───────────────


async def test_a_falsy_marker_reads_the_same_on_both_sides(db):
    """OCR fix (P4b review): ``state_of`` read truthiness while every SQL
    predicate reads PRESENCE (``_has``): a stored ``{"cm_invalidated": false}``
    made Python answer "current" about a row SQL hid. Both sides read the
    marker's presence now — one row, one verdict."""
    owner = await _owner(db)
    falsy = _mem(owner, "falsy invalidated", meta={"cm_invalidated": False})
    db.add(falsy)
    falsy_id = falsy.id  # capture before commit: the instance expires on commit
    await db.commit()

    served = list((await db.execute(
        select(Memory.id).where(Memory.id == falsy_id,
                                current_memory_predicate()))).scalars().all())
    assert served == [], "SQL hides the row: any non-NULL marker value counts"

    fresh = (await db.execute(select(Memory).where(Memory.id == falsy_id))).scalar_one()
    assert state_of(fresh) == "invalidated", (
        "the Python authority must agree with the SQL side: present, not truthy")
