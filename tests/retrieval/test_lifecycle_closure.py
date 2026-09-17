"""P4b lifecycle pins on the existing private SQLite harness."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.database import get_db
from app.main import app
from app.mcp_hub import tools
from app.models.memory import Memory
from app.retrieval.memory import correction as C
from app.retrieval.memory import retriever as R
from app.retrieval.memory.visibility import (
    current_memory_predicate,
    not_dirty_predicate,
    state_expression,
)
from app.services.digest_service import build_digest
from tests.retrieval.test_visibility import (
    _as_reader,
    _mem,
    _owner,
)
from tests.retrieval.test_visibility import (
    db as db,
)


async def test_invalidated_state_mirrors_and_history(db, monkeypatch):
    owner = await _owner(db)
    anchor = _mem(owner, "anchor", minutes=2)
    invalid = _mem(owner, "invalid", minutes=1, meta={"cm_invalidated": True})
    both = _mem(owner, "all markers", minutes=3, meta={
        "cm_invalidated": True, C.CM_SUPERSEDED_BY: "new",
        C.CM_DERIVED_DIRTY: True, C.CM_NEEDS_CHECK: True,
    })
    db.add_all([anchor, invalid, both])
    await db.commit()
    assert C.state_of(invalid) == C.state_of(both) == "invalidated"
    assert "invalidated" in C.MEMORY_STATES
    labels = dict((await db.execute(select(Memory.id, state_expression()))).all())
    assert labels[invalid.id] == labels[both.id] == "invalidated"
    for predicate in (current_memory_predicate(), not_dirty_predicate()):
        assert list((await db.execute(select(Memory.id).where(predicate))).scalars()) == [anchor.id]
    assert R._hidden(invalid, include_superseded=True)
    assert R._memory_response(invalid).state == "invalidated"
    _as_reader(monkeypatch, owner)
    timeline = await tools.timeline(str(anchor.id))
    assert any(row["id"] == str(both.id) and row["state"] == "invalidated"
               for row in timeline["before"])
    assert any(row["id"] == str(invalid.id) and row["state"] == "invalidated"
               for row in timeline["after"])


async def test_retriever_context_drops_invalidated(db, monkeypatch):
    owner = await _owner(db)
    invalid = _mem(owner, "not context", meta={"cm_invalidated": True})
    db.add(invalid)
    await db.commit()
    monkeypatch.setattr(R, "await_freshness", AsyncMock(return_value=0))
    monkeypatch.setattr(R, "fetch_personal_context", AsyncMock(return_value=[invalid]))
    rewrite = AsyncMock(return_value={"rewritten_query": "hello", "entities": [],
                                     "reasoning": None, "_fallback_used": False})
    monkeypatch.setattr(R, "rewrite_query", rewrite)
    monkeypatch.setattr(R, "embed_query", AsyncMock(side_effect=RuntimeError("offline")))
    await R.MemoryRetriever(db, owner).recall("hello", include_personal_context=True)
    assert rewrite.call_args.kwargs["context"] == []


async def test_shared_memory_hides_invalidated_not_current(db, monkeypatch):
    owner = await _owner(db)
    current = _mem(owner, "shared current")
    invalid = _mem(owner, "shared invalid", meta={C.CM_INVALIDATED: True})
    current.is_shared = invalid.is_shared = True
    db.add_all([current, invalid])
    await db.commit()

    async def session():
        yield db

    monkeypatch.setitem(app.dependency_overrides, get_db, session)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        visible = await client.get(f"/api/v1/memories/{current.id}/share")
        hidden = await client.get(f"/api/v1/memories/{invalid.id}/share")
    assert visible.status_code == 200
    assert visible.json()["content"] == current.content
    assert hidden.status_code == 404
    assert invalid.content not in hidden.text


async def test_digest_excludes_invalidated_from_recent_and_resurfaced(db):
    owner = await _owner(db)
    now = datetime(2026, 9, 17, tzinfo=UTC)
    current = _mem(owner, "recent current")
    invalid = _mem(owner, "recent invalid", meta={C.CM_INVALIDATED: True})
    old_current = _mem(owner, "old current")
    old_invalid = _mem(owner, "old invalid", meta={C.CM_INVALIDATED: True})
    current.captured_at = invalid.captured_at = now - timedelta(days=1)
    old_current.captured_at = old_invalid.captured_at = now.replace(year=2025)
    current.tags = ["valid"]
    invalid.tags = ["invalid"]
    db.add_all([current, invalid, old_current, old_invalid])
    await db.commit()

    digest = await build_digest(db, owner, now=now)
    assert [m.id for m in digest.recent_memories] == [current.id]
    assert [m.memory.id for m in digest.resurfaced] == [old_current.id]
    assert digest.recent_count == 1
    assert [(theme.theme, theme.count) for theme in digest.top_themes] == [("valid", 1)]


async def _chain(db):
    owner = await _owner(db)
    a = _mem(owner, "source", meta={C.CM_SUBJECT: "project", C.CM_ATTRIBUTE: "db"})
    b = _mem(owner, "derived", meta={C.CM_DERIVED_FROM: [str(a.id)]})
    c = _mem(owner, "derived twice", meta={C.CM_DERIVED_FROM: [str(b.id)]})
    child = _mem(owner, "parent edge")
    child.parent_id = c.id
    db.add_all([a, b, c])
    await db.commit()
    db.add(child)
    await db.commit()
    return owner, a, b, c, child


async def test_dependency_closure_transitive_kinds_and_boundary(db):
    owner, a, b, c, child = await _chain(db)
    other_owner = await _owner(db)
    foreign = _mem(other_owner, "foreign", meta={C.CM_DERIVED_FROM: [str(b.id)]})
    team = _mem(owner, "team", namespace="team", meta={C.CM_DERIVED_FROM: [str(a.id)]})
    behind_team = _mem(owner, "not reachable", meta={C.CM_DERIVED_FROM: [str(team.id)]})
    db.add_all([foreign, team, behind_team])
    await db.commit()
    result = await C.collect_dependency_closure(db, [a.id, a.id])
    assert result.affected == [a.id, b.id, c.id, child.id]
    assert result.visited == {a.id, b.id, c.id, child.id}
    assert not result.truncated
    derived = await C.collect_dependency_closure(db, [a.id], kinds=("derived",))
    assert derived.affected == [a.id, b.id, c.id]
    parents = await C.collect_dependency_closure(db, [c.id], kinds=("parent",))
    assert parents.affected == [c.id, child.id]
    with pytest.raises(ValueError):
        await C.collect_dependency_closure(db, [team.id])
    with pytest.raises(ValueError):
        await C.collect_dependency_closure(db, [a.id, foreign.id])
    with pytest.raises(ValueError):
        await C.collect_dependency_closure(db, [a.id], kinds=("typo",))
    assert (await C.collect_dependency_closure(db, [])).affected == []


async def test_dependency_cycle_terminates_without_truncating(db, monkeypatch):
    _, a, b, c, child = await _chain(db)
    C.set_cm(a, {C.CM_DERIVED_FROM: [str(c.id)]})
    await db.commit()
    # Removing the visited guard hits this cap instead of hanging the test.
    monkeypatch.setattr(C, "_MAX_CLOSURE_IDS", 6)
    result = await C.collect_dependency_closure(db, [a.id])
    assert result.affected == [a.id, b.id, c.id, child.id]
    assert not result.truncated


async def test_correction_dirties_transitively_through_dirty_node(db):
    owner, a, b, c, child = await _chain(db)
    C.set_cm(b, {C.CM_DERIVED_DIRTY: True})
    await db.commit()
    out = await C.resolve_correction(db, user_id=owner, title="new", content="postgres",
                                     slot=C.Slot.of("project", "db"))
    assert out["status"] == "superseded"
    assert set(out["dirtied"]) == {str(c.id), str(child.id)}
    assert C.state_of(c) == C.state_of(child) == "dirty"
    assert C.state_of(a) == "superseded"


async def test_truncated_closure_refuses_correction_before_any_write(db, monkeypatch):
    owner, a, b, c, child = await _chain(db)
    monkeypatch.setattr(C, "_MAX_CLOSURE_IDS", 2)
    result = await C.collect_dependency_closure(db, [a.id])
    assert result.truncated
    with pytest.raises(C.DerivedClosureError, match="truncated"):
        await C.resolve_correction(db, user_id=owner, title="new", content="postgres",
                                   slot=C.Slot.of("project", "db"))
    # Even committing after the refusal must not publish a half-correction.
    await db.commit()
    assert set((await db.execute(select(Memory.id))).scalars()) == {a.id, b.id, c.id, child.id}
    assert all(C.state_of(m) == "current" for m in (a, b, c, child))


async def test_erasure_counts_cascade_subtree_below_derived(db, monkeypatch):
    from app.services import erasure_service as E

    owner, a, b, c, child = await _chain(db)
    # R36's erase union stays root-parent closure + direct derived ids.
    # B is deleted; its same-namespace child and deeper foreign child are
    # DB-cascaded without vector intents, so completion MUST report residual.
    child.parent_id = b.id
    team = _mem(owner, "team grandchild", namespace="team")
    team.parent_id = child.id
    db.add(team)
    await db.commit()
    purged = []

    async def purge(mid):
        purged.append(mid)
        return True

    monkeypatch.setattr(E, "safe_delete_from_index", purge)
    monkeypatch.setattr(E, "_vector_present_ids", AsyncMock(return_value=set()))
    receipt = await E.erase_memories(db, owner, [a.id], requested_by="test")
    assert receipt.status == E.ERASURE_STATUS_RESIDUAL
    target = receipt.detail["targets"][0]
    assert target["db_residual"]["cascaded_out_of_namespace"] == 2
    assert set(purged) == {a.id, b.id}  # R36 union unchanged
    remaining = set((await db.execute(select(Memory.id))).scalars())
    assert child.id not in remaining and team.id not in remaining  # actual FK effect
    assert c.id in remaining  # derived-of-derived is not silently added to erase union
