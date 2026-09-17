"""P4b T5 — the consolidation producer (spec §8.1/§8.2, plan Task 5).

Real SQLite, real rows, real SQL (``tests.retrieval.test_visibility``'s private
per-test file). The LLM is a seam (``consolidation._complete``) and EVERY test
here patches it — no test in this file may reach a provider.

What these pin (spec §8.1): a summary is a VIEW of its sources at their
revisions. It may never be served as a fresher fact (serving label ``derived``),
a re-run must not publish a second copy of the same view, a source that moved
while the summary was being generated may not be published as if it had not
moved, and the run stays inside its budget.
"""
from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace

from sqlalchemy import select

from app import database
from app.config import settings
from app.mcp_hub import tools
from app.models.index_outbox import IndexOutbox
from app.models.memory import Memory, MemorySuppression
from app.retrieval.memory import consolidation, drain_loop
from app.retrieval.memory import correction as C
from app.retrieval.memory.consolidation import (
    RULE_VERSION,
    ConsolidationReport,
    run_consolidation,
)
from tests.retrieval.test_visibility import _as_reader, _mem, _owner
from tests.retrieval.test_visibility import db as db


def _tagged(owner, title: str, *, tag: str = "atlas", meta: dict | None = None) -> Memory:
    row = _mem(owner, title, meta=meta)
    row.tags = [tag]
    return row


def _fake_llm(monkeypatch, *, text: str = "atlas rolled up", within=None):
    """The LLM seam, deterministic. ``within`` runs INSIDE the call (a race)."""
    calls: list[str] = []

    async def _complete(*, agent: str, **kwargs):
        calls.append(agent)
        if within is not None:
            await within()
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
        )

    monkeypatch.setattr(consolidation, "_complete", _complete)
    return calls


def _expected_key(source_ids, revisions) -> str:
    """The plan's contract, spelled independently here: the dedupe key is a
    sha256 over the sorted source set, their revisions and the rule version."""
    ids = sorted(source_ids)
    payload = (",".join(ids) + "|"
               + ",".join(f"{sid}:{revisions[sid]}" for sid in ids) + "|" + RULE_VERSION)
    return hashlib.sha256(payload.encode()).hexdigest()


async def _derived_rows(db, owner) -> list[Memory]:
    """The owner's derived summaries, read FRESH (never from the identity map)."""
    rows = (await db.execute(
        select(Memory).where(Memory.user_id == owner)
        .execution_options(populate_existing=True)
    )).scalars().all()
    return [row for row in rows if C.get_cm(row).get(C.CM_DERIVED_FROM)]


async def _group(db, owner, *, tag: str = "atlas"):
    """Two servable sources under one tag — the smallest consolidation group."""
    a, b = _tagged(owner, f"{tag} one", tag=tag), _tagged(owner, f"{tag} two", tag=tag)
    db.add_all([a, b])
    await db.commit()
    return a, b


# ── the publish: provenance, serving label, index intent ────────────────────


async def test_publishes_the_group_with_full_provenance(db, monkeypatch):
    owner = await _owner(db)
    a, b = await _group(db, owner)
    lone = _tagged(owner, "solo", tag="solo")
    db.add(lone)
    await db.commit()
    a_id, b_id = a.id, b.id
    calls = _fake_llm(monkeypatch)

    report = await run_consolidation(db, owner, budget=10)

    assert len(report.published) == 1
    derived = (await _derived_rows(db, owner))[0]
    assert report.published == [str(derived.id)]
    meta = C.get_cm(derived)
    assert meta[C.CM_DERIVED_FROM] == sorted([str(a_id), str(b_id)])
    assert meta[C.CM_SOURCE_REVISIONS] == {str(a_id): 1, str(b_id): 1}
    assert meta[C.CM_RULE_VERSION] == RULE_VERSION
    assert meta[C.CM_EVIDENCE_IDS] == sorted([str(a_id), str(b_id)])
    assert meta[C.CM_ASSERTION] == "derived"
    assert meta[C.CM_DERIVED_KEY] == _expected_key(
        [str(a_id), str(b_id)], {str(a_id): 1, str(b_id): 1})
    assert C.state_of(derived) == "current"
    assert derived.source_type == "consolidation"
    assert derived.tags == ["atlas"] and derived.content == "atlas rolled up"
    assert calls == ["consolidation"]
    # The publish carries its own durable index intent (bump_revision + enqueue).
    assert derived.revision == 1
    intents = (await db.execute(select(IndexOutbox))).scalars().all()
    assert [(i.kind, i.entity_id, i.revision, i.operation, i.status) for i in intents] == [
        ("memory", derived.id.hex, 1, "upsert", "pending")]
    assert C.state_of(lone) == "current", "a group of one is not consolidation"


async def test_only_servable_non_suppressed_raw_rows_are_sources(db, monkeypatch):
    """A summary's evidence is rows a reader could serve: the group query carries
    the serving predicate, the ledger, and the non-recursion (rule v1)."""
    owner = await _owner(db)
    rows = [
        _tagged(owner, "dirty pair", tag="pair", meta={C.CM_DERIVED_DIRTY: True}),
        _tagged(owner, "current pair", tag="pair"),
        _tagged(owner, "invalid pair", tag="gone", meta={C.CM_INVALIDATED: True}),
        _tagged(owner, "superseded pair", tag="gone", meta={C.CM_SUPERSEDED_BY: "x"}),
        _tagged(owner, "summary one", tag="views", meta={C.CM_DERIVED_FROM: ["a"]}),
        _tagged(owner, "summary two", tag="views", meta={C.CM_DERIVED_FROM: ["b"]}),
    ]
    lifted = [_tagged(owner, "lifted one", tag="lift"), _tagged(owner, "lifted two", tag="lift")]
    for row in lifted:
        row.source_ref = "file:forgotten.pdf"
    db.add_all(rows + lifted)
    db.add(MemorySuppression(user_id=owner, source_ref="file:forgotten.pdf",
                             reason="soft_forget"))
    await db.commit()
    calls = _fake_llm(monkeypatch)

    report = await run_consolidation(db, owner)

    assert (report.published, calls) == ([], []), "nothing servable to consolidate"
    views = await _derived_rows(db, owner)
    assert len(views) == 2, "the two fixture summaries are the only derived rows"
    assert {C.state_of(row) for row in views} == {"current"}, "and they were not used as sources"


async def test_rerun_is_idempotent_by_dedupe_key(db, monkeypatch):
    owner = await _owner(db)
    await _group(db, owner)
    calls = _fake_llm(monkeypatch)

    first = await run_consolidation(db, owner)
    second = await run_consolidation(db, owner)

    assert len(first.published) == 1
    assert (second.published, second.skipped) == ([], 1)
    assert calls == ["consolidation"], "a keyed re-run never re-asks the LLM"
    assert len(await _derived_rows(db, owner)) == 1


# ── the stale guard: a source that moves while the summary runs ─────────────


async def _another_writer_moves(db, memory_id, *, revision: int) -> None:
    """A second connection's committed write (the real race shape)."""
    async with database.AsyncSessionLocal() as other:
        row = await other.get(Memory, memory_id)
        assert row is not None
        row.revision = revision
        await other.commit()


async def test_a_source_that_moves_mid_run_is_refused_and_the_view_goes_dirty(db, monkeypatch):
    owner = await _owner(db)
    a, _b = await _group(db, owner)
    a_id = a.id
    _fake_llm(monkeypatch)
    await run_consolidation(db, owner)  # S1 over (a@1, b@1)
    published = (await _derived_rows(db, owner))[0]
    published_id = published.id

    # A correction landed BEFORE the pass: the key moves, so the group is work
    # again. (The pass runs on its own session, the drain loop's shape.)
    await _another_writer_moves(db, a_id, revision=2)

    async def _move_again():
        await _another_writer_moves(db, a_id, revision=3)

    calls = _fake_llm(monkeypatch, text="stale rollup", within=_move_again)
    async with database.AsyncSessionLocal() as pass_db:
        report = await run_consolidation(pass_db, owner)

    assert calls == ["consolidation"], "the summary really was generated first"
    assert (report.published, report.refused) == ([], 1)
    assert report.dirtied == 1
    rows = await _derived_rows(db, owner)
    assert [str(row.id) for row in rows] == [str(published_id)], "nothing new published"
    assert C.state_of(rows[0]) == "dirty", "the view of a moved source is marked"
    assert "stale rollup" not in (rows[0].content or "")


async def test_missing_evidence_is_never_published(db, monkeypatch):
    """A source erased while the summary runs: the output has no evidence."""
    owner = await _owner(db)
    _a, b = await _group(db, owner)
    b_id = b.id

    async def _erase_b():
        async with database.AsyncSessionLocal() as other:
            await other.delete(await other.get(Memory, b_id))
            await other.commit()

    _fake_llm(monkeypatch, within=_erase_b)
    report = await run_consolidation(db, owner)

    assert (report.published, report.refused) == ([], 1)
    assert await _derived_rows(db, owner) == []


async def test_a_cycle_in_the_dependency_graph_does_not_hang_the_guard(db, monkeypatch):
    """The dirty-propagation walk is cycle-defended (spec §8.1: cycle defense)."""
    owner = await _owner(db)
    a, _b = await _group(db, owner)
    a_id = a.id
    p = _tagged(owner, "view p", tag="other")
    q = _tagged(owner, "view q", tag="other")
    db.add_all([p, q])
    await db.commit()
    # p and q depend on each other (a stored cycle) and p depends on the source.
    C.set_cm(p, {C.CM_DERIVED_FROM: [str(a_id), str(q.id)]})
    C.set_cm(q, {C.CM_DERIVED_FROM: [str(p.id)]})
    await db.commit()
    # A missing visited-guard would walk the cycle until this cap, not hang.
    monkeypatch.setattr(C, "_MAX_CLOSURE_IDS", 6)

    async def _race():
        await _another_writer_moves(db, a_id, revision=2)

    _fake_llm(monkeypatch, within=_race)
    report = await run_consolidation(db, owner)

    assert report.published == [] and report.refused == 1
    assert report.truncated is False, "the walk terminated, it did not cap out"
    assert report.dirtied == 2


# ── budget (soft: ≤ budget memories per run) ────────────────────────────────


async def test_budget_caps_a_run_and_a_later_run_makes_progress(db, monkeypatch):
    owner = await _owner(db)
    for tag in ("alpha", "beta", "gamma"):
        await _group(db, owner, tag=tag)
    calls = _fake_llm(monkeypatch)

    first = await run_consolidation(db, owner, budget=2)
    assert len(first.published) == 2
    assert (first.skipped, first.deferred, calls) == (0, 1, ["consolidation"] * 2)
    assert len(await _derived_rows(db, owner)) == 2

    second = await run_consolidation(db, owner, budget=2)
    assert (len(second.published), second.skipped, second.deferred) == (1, 2, 0)
    assert len(await _derived_rows(db, owner)) == 3

    zero = await run_consolidation(db, owner, budget=0)
    assert (zero.published, len(calls)) == ([], 3), "budget 0 spends no LLM work"


async def test_an_llm_failure_or_empty_summary_never_publishes(db, monkeypatch):
    owner = await _owner(db)
    await _group(db, owner)

    async def _boom(**kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(consolidation, "_complete", _boom)
    report = await run_consolidation(db, owner)
    assert (report.published, report.errors) == ([], 1)

    _fake_llm(monkeypatch, text="   ")
    empty = await run_consolidation(db, owner)
    assert (empty.published, empty.errors) == ([], 1)
    assert await _derived_rows(db, owner) == []


# ── the serving label (spec §8.2: never stronger evidence than raw) ─────────


async def test_mcp_provenance_labels_a_summary_as_derived(db, monkeypatch):
    owner = await _owner(db)
    a, b = await _group(db, owner)
    a_id, b_id = a.id, b.id
    _fake_llm(monkeypatch)
    await run_consolidation(db, owner)
    derived = (await _derived_rows(db, owner))[0]
    derived_id = derived.id
    _as_reader(monkeypatch, owner)

    summary = await tools.get_memory(str(derived_id))
    assert summary["derived"] is True
    assert summary["derived_from"] == sorted([str(a_id), str(b_id)])
    assert summary["rule_version"] == RULE_VERSION
    assert summary["source_revisions"] == {str(a_id): 1, str(b_id): 1}
    assert summary["assertion"] == "derived"
    assert summary["state"] == "current"

    raw = await tools.get_memory(str(a_id))
    assert raw["derived"] is False and raw["derived_from"] == []


# ── R39: the drain loop's hook (after applied>0 AND on an idle tick) ────────


async def _drive_loop(monkeypatch, *, user_id, applied_rounds: int, hook, markers: list[str]):
    """Run the drain loop for a few rounds; ``markers`` records which round the
    hook saw (an applied round or an idle tick)."""
    state = {"rounds": 0, "landed": False}

    async def _drain(*, batch_size):
        state["rounds"] += 1
        applied = 1 if state["rounds"] <= applied_rounds else 0
        state["landed"] = bool(applied)
        return {"applied": applied}

    async def _reconcile():
        return None

    async def _users(db_, **kwargs):
        return [user_id]

    async def _run(db_, user, budget=10):
        markers.append("applied" if state["landed"] else "idle")
        return await hook(db_, user, budget)

    monkeypatch.setattr(drain_loop, "drain_once", _drain)
    monkeypatch.setattr(drain_loop, "_reconcile_after_drain", _reconcile)
    monkeypatch.setattr(drain_loop, "users_with_servable_memories", _users)
    monkeypatch.setattr(drain_loop, "run_consolidation", _run)

    stop = asyncio.Event()
    task = asyncio.create_task(
        drain_loop.run_drain_loop(interval=0.02, batch_size=5, stop=stop))
    deadline = asyncio.get_running_loop().time() + 5.0
    while len(markers) < 3:
        assert asyncio.get_running_loop().time() < deadline, "the hook never ran"
        await asyncio.sleep(0.01)
    stop.set()
    await asyncio.wait_for(task, timeout=5.0)


async def test_the_loop_hooks_consolidation_after_applied_and_on_idle(db, monkeypatch):
    owner = await _owner(db)
    seen: list[tuple] = []

    async def _hook(db_, user_id, budget):
        seen.append((user_id, budget))
        return ConsolidationReport(published=[], skipped=0, refused=0, dirtied=0,
                                   errors=0, deferred=0, truncated=False)

    markers: list[str] = []
    await _drive_loop(monkeypatch, user_id=owner, applied_rounds=2, hook=_hook,
                      markers=markers)

    assert markers[:2] == ["applied", "applied"], "a landed round picks the work up"
    assert "idle" in markers, "an idle tick is free time for the producer (R39)"
    assert {uid for uid, _ in seen} == {owner}
    assert {budget for _, budget in seen} == {settings.CONSOLIDATION_BUDGET_PER_RUN}


async def test_a_consolidation_failure_does_not_stop_the_drain(db, monkeypatch):
    owner = await _owner(db)
    calls: list[tuple] = []

    async def _hook(db_, user_id, budget):
        calls.append((user_id, budget))
        raise RuntimeError("producer exploded")

    markers: list[str] = []
    await _drive_loop(monkeypatch, user_id=owner, applied_rounds=0, hook=_hook,
                      markers=markers)

    assert len(calls) >= 3, "the loop kept ticking after every producer failure"
    assert not markers == [], "the hook ran at all"
