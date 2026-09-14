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

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest_asyncio
from sqlalchemy import create_engine, event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from app import database
from app.api.v1.memories import list_memories
from app.database import Base
from app.mcp_hub import tools as hub_tools
from app.mcp_hub.identity import AgentPrincipal
from app.models.memory import Memory
from app.models.user import User
from app.retrieval.memory import vector_store
from app.retrieval.memory.context import fetch_personal_context
from app.retrieval.memory.correction import (
    CM_DERIVED_DIRTY,
    CM_NEEDS_CHECK,
    CM_SUPERSEDED_BY,
    state_of,
)
from app.retrieval.memory.reindex import reindex_user_memories_sync
from app.retrieval.memory.visibility import (
    current_memory_predicate,
    state_expression,
)

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


async def test_mcp_search_dirty_rows_do_not_consume_candidate_slots(db, monkeypatch):
    """The recall LIMIT applies AFTER the dirty filter: a dirty row that ranks
    top cannot crowd a current row out of the capped candidate slots (pre-fix
    this returned 1 row for ``limit=3``)."""
    owner = await _owner(db)
    current = [_mem(owner, f"current {i}", minutes=10 - i) for i in range(3)]
    dirty = [_mem(owner, f"dirty {i}", meta={CM_DERIVED_DIRTY: True}, minutes=i)
             for i in range(2)]
    for stale in dirty:
        stale.salience = 0.99  # dirty rows outrank every current row
    db.add_all([*current, *dirty])
    await db.commit()
    _as_reader(monkeypatch, owner)

    out = await hub_tools.search_memory("body", limit=3)  # real _recall_memory_ids

    assert len(out["results"]) == 3  # honest count: no slot eaten by a dirty row
    assert [r["id"] for r in out["results"]] == [str(m.id) for m in reversed(current)]
    assert all(r["state"] == "current" for r in out["results"])


async def test_mcp_search_recall_keeps_superseded_for_history(db, monkeypatch):
    """Superseded rows stay eligible in recall (history widening happens at
    hydration); the ``include_history=False`` path still excludes them, dirty
    rows are never served either way."""
    owner = await _owner(db)
    current = _mem(owner, "current", minutes=0)
    superseded = _mem(owner, "superseded", meta={CM_SUPERSEDED_BY: "s"}, minutes=1)
    dirty = _mem(owner, "dirty", meta={CM_DERIVED_DIRTY: True}, minutes=2)
    dirty.salience = 0.99
    db.add_all([current, superseded, dirty])
    await db.commit()
    _as_reader(monkeypatch, owner)

    plain = await hub_tools.search_memory("body")  # real _recall_memory_ids
    assert [r["id"] for r in plain["results"]] == [str(current.id)]

    history = await hub_tools.search_memory("body", include_history=True)
    assert [r["id"] for r in history["results"]] == [str(current.id), str(superseded.id)]
    assert [r["state"] for r in history["results"]] == ["current", "superseded"]


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
