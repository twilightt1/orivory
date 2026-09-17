"""P4b acceptance gate — the §9 P4 row (lifecycle part), over the REAL stores.

Everything here runs against a real SQLite file and a real embedded Qdrant
folder, through the P1b gate's own harness (``pytest_plugins`` below, reused by
name): a private per-test database, a private embedded-Qdrant folder, the real
cutover install (both manifests ACTIVE) and deterministic unit vectors for the
embedding CONTRACT. Nothing about SQL or the vector store is mocked.

The only substitutions are the seams that are out of process in production, and
they are the P1b/P3/P4a gates' own seams rather than new ones: the embedder
(deterministic unit vectors — no claim here is about embedding QUALITY), the
recall path's LLM query rewrite, and — in the consolidation bullet only — the
shared LLM completion seam (an LLM call, never the claim).

The §9 P4 row, lifecycle bullet by bullet — one test each:

* transitive invalidation + closure (plan §8.1): a correction dirties the
  derived chain AND the parent chain transitively, a cycle terminates, and a
  truncated closure refuses the correction whole → test 1;
* concurrent same-slot corrections (CAS): the MCP boundary carries the
  revision it read, so a rival correction that lands mid-call answers
  ``conflict`` instead of a silent supersede → test 2;
* ... and the CAS never moves a candidate the caller never named: a slot with
  two exact candidates refuses rather than superseding the one it was not
  asked about → test 3;
* soft forget + suppression + provenance: MCP ``forget_memory`` invalidates
  the closure, keeps the rows, pins EVERY affected source, refreshes the
  vector payload (R37) and blocks a re-import — while an explicit forget still
  wins over a pin → test 4;
* budgeted consolidation with provenance: the budget defers, the dedupe key
  makes a re-run idempotent, the publish-time guard stands down a moved source
  and marks the stale view dirty, and the MCP provenance carries the
  ``derived`` label → test 5;
* retention opt-in: default OFF sweeps nobody, an opted-in user's old rows are
  invalidated per row (audit row, pin exempt, re-run a no-op) → test 6;
* ... behind the REAL HTTP boundary: ``PATCH /users/me/settings`` is a full
  replace and refuses "enabled without a window" → test 7;
* serving exclusions: dirty and invalidated rows are never served (REST list,
  recall, MCP search), while direct reads and timeline still show them
  labelled — and ``include_history`` widens to superseded only → test 8;
* the CI pin: the workflow runs this file, wires the P4b suites that had no CI
  step (T1–T6), names no path that does not exist, and cannot be stood down by
  ``--ignore``/``if:``/``continue-on-error`` → test 9.

The gate is FALLIBLE by mutation (recorded in the task report): capping the
closure at one hop, dropping the MCP snapshot, hard-erasing in the forget
tool, dropping the budget, or flipping retention's default ON each turns its
bullet red on a copy of the tree.
"""
from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi import Depends
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, update

import app.models  # noqa: F401 — register every ORM table on Base
from app import database
from app.main import app as asgi_app
from app.mcp_hub import tools as hub_tools
from app.mcp_hub.identity import AgentPrincipal
from app.models.memory import Memory, MemorySuppression
from app.models.memory_access_log import MemoryAccessLog
from app.models.user import User
from app.retrieval.embedding_fingerprint import generation_name
from app.retrieval.memory import consolidation, outbox, vector_store
from app.retrieval.memory import correction as C
from app.retrieval.memory import retriever as retriever_module
from app.retrieval.memory.correction import (
    DerivedClosureError,
    Slot,
    collect_dependency_closure,
    resolve_correction,
    state_of,
)
from app.retrieval.memory.namespaces import PERSONAL
from app.retrieval.memory.retriever import MemoryRetriever
from app.services import import_service
from app.services.retention_service import (
    RETENTION_REASON,
    RetentionReport,
    run_retention,
)
from app.utils.dependencies import get_current_verified_user
from tests.retrieval.test_p1b_gate import (  # noqa: F401 — the harness, by name
    _intents,
    _memory,
    _payloads,
    _vector_for,
)

# The P1b gate's real-store fixtures (``env`` / ``world`` / ``live``), reused
# rather than copied: this gate has to prove the SAME store every earlier gate
# proved, not a second harness that happens to look like it.
pytest_plugins = ["tests.retrieval.test_p1b_gate"]

REPO = Path(__file__).resolve().parents[2]
CI_YML = REPO / ".github" / "workflows" / "ci.yml"
CI_STEP_NAME = "Run P4b lifecycle gate suites (temp SQLite, no services)"
GATE_MODULE = "tests/retrieval/test_p4b_gate.py"
SLOT = Slot.of("proj-x", "db", "prod")

# The P4b suites the new CI step must wire. Explicit paths on purpose: before
# this step, none of them was named in ANY CI step — the whole P4b phase (T1
# through T6) was green locally and invisible to CI.
P4B_CI_SUITES = (
    "tests/retrieval/test_p4b_gate.py",
    "tests/retrieval/test_lifecycle_closure.py",
    "tests/retrieval/test_correction_concurrency.py",
    "tests/retrieval/test_consolidation.py",
    "tests/services/test_soft_forget.py",
    "tests/services/test_suppression_guards.py",
    "tests/services/test_retention.py",
    "tests/lite/test_sqlite_schema_v6.py",
    "tests/lite/test_sqlite_schema_v7.py",
    # DB-free, so the step can gate it without Postgres: the T4 shape pin lived
    # in a PG-probed module before, i.e. it skipped in exactly the CI run that
    # was supposed to gate it.
    "tests/api/test_import_summary_shape.py",
)
# Wired by other steps — the pin only requires they stay wired SOMEWHERE (a
# suite that silently loses its step is the same hole as one that never had
# one). The mcp_hub dir carries the forget-tool unit pins; the v5 ladder and
# the AST fence belong to the P4a step; test_visibility to the P1a step.
P4B_WIRED_ELSEWHERE = (
    "tests/mcp_hub/test_forget_tool.py",
    "tests/lite/test_sqlite_schema_v5.py",
    "tests/api/test_dormant_router_acl.py",
    "tests/retrieval/test_visibility.py",
)
P4B_DOCS = (
    REPO / "docs" / "ARCHITECTURE.md",
    REPO / "docs" / "OPERATIONS_RUNBOOK.md",
    REPO / "docs" / "API.md",
    REPO / "skills" / "orivory" / "references" / "tool-catalog.md",
)


# ── helpers ─────────────────────────────────────────────────────────────────


def _slotted(user_id, content: str, *, slot: Slot = SLOT, **kwargs) -> Memory:
    """A current memory occupying ``slot`` (subject/attribute/scope markers)."""
    extra = kwargs.pop("extra_metadata", {})
    return _memory(user_id, content, extra_metadata={
        "cm_subject": slot.subject, "cm_attribute": slot.attribute,
        "cm_scope": slot.scope, **extra,
    }, **kwargs)


def _derived(user_id, content: str, *, sources: list, **kwargs) -> Memory:
    """A (hand-written) derived view over ``sources`` — the T5 shape, no LLM."""
    return _memory(user_id, content, extra_metadata={
        C.CM_DERIVED_FROM: [str(s) for s in sources], C.CM_ASSERTION: "derived",
    }, **kwargs)


async def _fresh(db, user_id) -> dict:
    """The user's rows keyed by id, read FRESH (never from the identity map)."""
    db.expire_all()
    return {row.id: row for row in (await db.execute(
        select(Memory).where(Memory.user_id == user_id))).scalars().all()}


def _mcp_seams(monkeypatch, principal) -> None:
    """Point the MCP tools at the private stores with ``principal`` in place."""
    monkeypatch.setattr(hub_tools, "_current_principal", lambda: principal)
    monkeypatch.setattr(hub_tools, "_session", database.AsyncSessionLocal)


def _writer_principal(live) -> AgentPrincipal:
    return AgentPrincipal(user_id=live.alice.id, agent_client_id=uuid.uuid4(),
                          name="gate", scopes=frozenset({"memory:read", "memory:write"}))


def _auth_client(user) -> AsyncClient:
    """An ASGI client over the real app, authenticated as ``user``.

    The override loads the user IN the request's own session (``get_db`` is
    cached per request, so this is the same session the route writes with): a
    settings PATCH mutates and commits the row, which a detached instance
    cannot do.
    """
    async def _current_user(db=Depends(database.get_db)):
        return await db.get(User, user.id)

    asgi_app.dependency_overrides[get_current_verified_user] = _current_user
    return AsyncClient(transport=ASGITransport(app=asgi_app), base_url="http://gate")


@pytest.fixture
def llm_seam(monkeypatch):
    """The recall path's LLM query rewrite — out of process, never the claim."""
    async def _identity(query, context=None):
        return {"rewritten_query": query, "entities": [], "reasoning": None,
                "_fallback_used": False}

    monkeypatch.setattr(retriever_module, "rewrite_query", _identity)


@pytest.fixture(autouse=True)
def _clear_app_overrides():
    """The ASGI app is module-global: never leave an auth override behind."""
    yield
    asgi_app.dependency_overrides.clear()


# ── 1. transitive invalidation + closure ────────────────────────────────────


async def test_a_correction_dirties_the_derived_chain_transitively_cycle_defends_and_refuses_truncation(live):
    """§8.1: dirty propagation and forgetting walk the dependency DAG.

    Measured, not asserted by reading code: a correction of the root dirties a
    DERIVED chain three hops long AND a ``parent_id`` descendant — one hop
    would leave ``m2``/``m3`` serving as current. A derived cycle terminates
    (the visited set is the defense) and a closure that would be capped is
    REFUSED before any write instead of half-applied.
    """
    alice = live.alice
    async with live.sessions() as db:
        root = _slotted(alice.id, "db v1")
        m1 = _derived(alice.id, "m1", sources=[root.id])
        m2 = _derived(alice.id, "m2", sources=[m1.id])
        m3 = _derived(alice.id, "m3", sources=[m2.id])
        child = _memory(alice.id, "child of root")
        child.parent_id = root.id
        # A cycle the walk must survive; it is NOT reachable from ``root``, so
        # the closure assertions below stay unambiguous.
        c1, c2 = _memory(alice.id, "cycle one"), _memory(alice.id, "cycle two")
        db.add_all([root, m1, m2, m3, child, c1, c2])
        await db.commit()
        c1.extra_metadata = {C.CM_DERIVED_FROM: [str(c2.id)]}
        c2.extra_metadata = {C.CM_DERIVED_FROM: [str(c1.id)]}
        await db.commit()
        ids = SimpleNamespace(root=root.id, m1=m1.id, m2=m2.id, m3=m3.id,
                              child=child.id, c1=c1.id, c2=c2.id)

    async with live.sessions() as db:
        cycle = await collect_dependency_closure(db, [ids.c1])
    assert set(cycle.affected) == {ids.c1, ids.c2}, "the cycle walk must terminate and cover both nodes"
    assert not cycle.truncated

    async with live.sessions() as db:
        out = await resolve_correction(db, user_id=alice.id, title="DB", content="Postgres",
                                       slot=SLOT)
    assert out["status"] == "superseded"

    async with live.sessions() as db:
        rows = await _fresh(db, alice.id)
    assert state_of(rows[ids.root]) == "superseded"
    for name in ("m1", "m2", "m3"):
        assert state_of(rows[getattr(ids, name)]) == "dirty", (
            f"{name} is {state_of(rows[getattr(ids, name)])}: a one-hop closure would do exactly this")
    assert state_of(rows[ids.child]) == "dirty", "a parent_id descendant is part of the closure"
    assert set(out["dirtied"]) == {str(ids.m1), str(ids.m2), str(ids.m3), str(ids.child)}, (
        f"the report names what it dirtied: {out['dirtied']}")
    assert state_of(rows[ids.c1]) == "current" and state_of(rows[ids.c2]) == "current", (
        "an unrelated cycle is not part of this closure")

    # The cap: a closure the walk cannot enumerate completely refuses the
    # WHOLE correction (no partial supersede, no half-dirty chain).
    before = len(rows)
    other_slot = Slot.of("proj-x", "db2", "prod")
    async with live.sessions() as db:
        root2 = _slotted(alice.id, "db2 v1", slot=other_slot)
        d1 = _derived(alice.id, "d1", sources=[root2.id])
        d2 = _derived(alice.id, "d2", sources=[d1.id])
        db.add_all([root2, d1, d2])
        await db.commit()
        root2_id = root2.id
    with pytest.MonkeyPatch.context() as cap:
        cap.setattr(C, "_MAX_CLOSURE_IDS", 2)
        async with live.sessions() as db:
            with pytest.raises(DerivedClosureError):
                await resolve_correction(db, user_id=alice.id, title="DB2", content="Postgres",
                                         slot=other_slot)
    async with live.sessions() as db:
        rows = await _fresh(db, alice.id)
    assert state_of(rows[root2_id]) == "current", "the refused correction superseded nothing"
    assert len(rows) == before + 3, "the refused correction wrote no new memory"


# ── 2. concurrent same-slot corrections (CAS), through the MCP boundary ─────


async def test_a_concurrent_correction_through_mcp_conflicts_and_leaves_one_current_fact(live, monkeypatch):
    """§8.2: two writers on one slot never leave two current facts.

    The snapshot is the MCP tool's own pre-read. A rival correction that lands
    between that read and the apply is REFUSED — and the refusal is the tool's
    ``conflict`` status, not a silent supersede: the loser's new row lands
    flagged ``needs-check`` and the winner's pointer stands. The rival call
    runs the real tool on its own session, so this is the production shape
    (agent reads, agent corrects, a second agent corrects in the same instant).
    """
    alice = live.alice
    _mcp_seams(monkeypatch, _writer_principal(live))
    async with live.sessions() as db:
        target = _slotted(alice.id, "db v1")
        db.add(target)
        await db.commit()
        target_id = target.id

    real = C._cas_supersede
    state = {"engaged": False, "winner": None}

    async def _racing(db_, memory, successor_id, expected):
        if not state["engaged"]:
            # Set first: the rival's own correction re-enters this hook.
            state["engaged"] = True
            rival = await hub_tools.correct_memory(
                memory_id=str(target_id), subject=SLOT.subject,
                attribute=SLOT.attribute, scope=SLOT.scope,
                title="DB", content="Postgres")
            assert rival["status"] == "superseded", rival
            state["winner"] = rival["id"]
        return await real(db_, memory, successor_id, expected)

    monkeypatch.setattr(C, "_cas_supersede", _racing)

    out = await hub_tools.correct_memory(memory_id=str(target_id), subject=SLOT.subject,
                                         attribute=SLOT.attribute, scope=SLOT.scope,
                                         title="DB", content="SQLite")

    assert state["engaged"], "the interleave really happened"
    assert out["status"] == "conflict", "a stale snapshot never supersedes silently"
    assert (out["superseded"], out["dirtied"]) == ([], [])
    assert out["state"] == "needs-check"

    async with live.sessions() as db:
        rows = await _fresh(db, alice.id)
    slotted = [row for row in rows.values() if row.extra_metadata.get("cm_subject") == SLOT.subject]
    assert rows[target_id].extra_metadata[C.CM_SUPERSEDED_BY] == state["winner"]
    current = [row.id for row in slotted if state_of(row) == "current"]
    assert current == [uuid.UUID(state["winner"])], (
        f"exactly one current fact on the slot, found {[str(i) for i in current]}")
    loser = [row for row in slotted if state_of(row) == "needs-check"]
    assert [str(row.id) for row in loser] == [out["id"]]


# ── 3. the CAS never moves a candidate the caller never named ───────────────


async def test_the_cas_refuses_a_slot_with_a_candidate_the_caller_never_named(live, monkeypatch):
    """The caller vetted the target it named; the tool never moves a second one.

    A slot that already holds TWO current facts is ambiguous by construction.
    Before the CAS wire this correction superseded both silently; now the
    unnamed candidate refuses the whole apply (savepoint stand-down), which is
    the fail-closed direction §8.1 asks for: nothing is superseded, the new row
    is flagged, and the ambiguity stays visible.
    """
    alice = live.alice
    _mcp_seams(monkeypatch, _writer_principal(live))
    async with live.sessions() as db:
        first, second = _slotted(alice.id, "db v1"), _slotted(alice.id, "db v2")
        db.add_all([first, second])
        await db.commit()
        first_id, second_id = first.id, second.id

    out = await hub_tools.correct_memory(memory_id=str(first_id), subject=SLOT.subject,
                                         attribute=SLOT.attribute, scope=SLOT.scope,
                                         title="DB", content="SQLite")

    assert out["status"] == "conflict", "an un-named candidate refuses the whole apply"
    assert out["superseded"] == [] and out["dirtied"] == []
    async with live.sessions() as db:
        rows = await _fresh(db, alice.id)
    assert rows[first_id].extra_metadata.get(C.CM_SUPERSEDED_BY) is None, (
        "the named candidate's own CAS write was stood down with the refusal")
    assert rows[second_id].extra_metadata.get(C.CM_SUPERSEDED_BY) is None
    assert {state_of(rows[first_id]), state_of(rows[second_id])} == {"current"}


# ── 4. soft forget: invalidation + suppression + provenance + re-import ─────


async def test_mcp_forget_is_soft_suppresses_every_source_and_blocks_reimport(live, monkeypatch):
    """§12/§5.4: forget invalidates and pins; it never shreds.

    Through the real tool: the closure is invalidated (root AND derived), the
    rows keep their content and provenance, every affected ``source_ref`` is
    pinned, the vector payload is refreshed to ``invalidated`` by the REAL
    applier (R37 — the point still exists, it just stops serving), and a
    re-import of the forgotten source is refused. The explicit forget wins over
    a pin.
    """
    alice = live.alice
    _mcp_seams(monkeypatch, _writer_principal(live))
    root_ref, derived_ref = "drive://p4b-forget-root", "drive://p4b-forget-derived"
    async with live.sessions() as db:
        root = _memory(alice.id, "forget root", source_type="file_upload",
                       source_ref=root_ref, pinned=True)
        derived = _derived(alice.id, "forget derived", sources=[root.id],
                           source_type="file_upload", source_ref=derived_ref)
        db.add_all([root, derived])
        await db.commit()
        root_id, derived_id = root.id, derived.id

    out = await hub_tools.forget_memory([str(root_id)])

    assert set(out) == {"receipt_id", "status", "invalidated", "suppressed", "skipped", "invalid"}, (
        f"the response keys are the soft contract: {sorted(out)}")
    assert "erased" not in out, "a soft forget never claims an erase"
    assert out["status"] == "completed", out
    # ``invalidated`` counts the REQUESTED targets this call invalidated (one
    # per id that reached the service); the closure it invalidated with them is
    # asserted on the rows below, and ``suppressed`` counts every affected
    # source — more than one per target is the closure working.
    assert (out["invalidated"], out["suppressed"], out["skipped"], out["invalid"]) == (1, 2, 0, [])

    async with live.sessions() as db:
        rows = await _fresh(db, alice.id)
        suppressions = (await db.execute(select(MemorySuppression).where(
            MemorySuppression.user_id == alice.id))).scalars().all()
    assert rows[root_id].content == "forget root", "the row (and its provenance) stays"
    assert rows[derived_id].extra_metadata[C.CM_DERIVED_FROM] == [str(root_id)]
    assert state_of(rows[root_id]) == state_of(rows[derived_id]) == "invalidated"
    assert rows[root_id].pinned is True, "pinning does not block an explicit forget"
    pinned_refs = {s.source_ref for s in suppressions}
    assert {root_ref, derived_ref} <= pinned_refs, "EVERY affected source is suppressed, not only the root"

    # R37 through the real applier: the point survives, its payload state moves.
    assert (await outbox.drain_pending())["applied"] >= 2
    payload = _payloads(generation_name("memory"))[str(root_id)]
    assert payload["visibility_state"] == "invalidated", (
        "the vector kept serving `current` — the refresh intent never landed")

    # The re-import guard reads the same ledger, through the real import path.
    async with live.sessions() as db:
        summary = await import_service.run_import(
            db, alice.id,
            b'[{"ref": "drive://p4b-forget-root", "title": "again", "content": "again"}]',
            "auto", requested_by="gate")
    assert summary.suppressed_skipped == 1 and summary.created == 0, (
        f"the forgotten source came back: {summary}")


# ── 5. budgeted consolidation with provenance and the publish-time guard ────


async def test_consolidation_is_budgeted_idempotent_guarded_and_labelled(live, monkeypatch):
    """§8.1: a derived summary has provenance, a budget, and a stale guard.

    The LLM seam is the only substitution — and it is also the interleave: the
    fake completion bumps a source's revision, which is exactly the window the
    publish-time guard protects (a source that moves while the summary is
    being generated). Every guard is measured on the real store: the budget
    defers the overflow instead of dropping it, a re-run recognises its own
    output by the dedupe key (no duplicate), the moved source stands the
    publish down AND the guard's walk marks the stale view dirty, and the MCP
    provenance carries the ``derived`` label with the source revisions the
    summary was published with.
    """
    alice = live.alice
    calls: list[str] = []
    race: dict = {"pending": None}

    async def _fake_complete(*, agent, messages, **kwargs):
        calls.append(agent)
        if race["pending"] is not None:
            # The interleave: another writer lands while the LLM "runs".
            source_id = race["pending"]
            race["pending"] = None
            async with database.AsyncSessionLocal() as other:
                await other.execute(update(Memory).where(Memory.id == source_id)
                                    .values(revision=Memory.revision + 1))
                await other.commit()
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="summary of the notes"))])

    monkeypatch.setattr(consolidation, "_complete", _fake_complete)
    _mcp_seams(monkeypatch, _writer_principal(live))
    async with live.sessions() as db:
        alpha = [_memory(alice.id, f"alpha {i}") for i in range(2)]
        beta = [_memory(alice.id, f"beta {i}") for i in range(2)]
        for row in alpha:
            row.tags = ["p4b-alpha"]
        for row in beta:
            row.tags = ["p4b-beta"]
        db.add_all(alpha + beta)
        await db.commit()
        alpha_ids, beta_ids = [str(m.id) for m in alpha], [str(m.id) for m in beta]

    async with live.sessions() as db:
        first = await consolidation.run_consolidation(db, alice.id, budget=1)
    assert len(first.published) == 1 and first.deferred == 1 and len(calls) == 1, first

    async with live.sessions() as db:
        zero = await consolidation.run_consolidation(db, alice.id, budget=0)
    assert zero.published == [] and zero.deferred == 1 and len(calls) == 1, (
        "a spent budget buys no LLM call")

    async with live.sessions() as db:
        again = await consolidation.run_consolidation(db, alice.id, budget=10)
    assert again.skipped == 1 and len(again.published) == 1, (
        f"the re-run republished its own output: {again}")
    assert len(calls) == 2

    beta_summary = again.published[0]
    got = await hub_tools.get_memory(beta_summary)
    assert got["derived"] is True and got["assertion"] == "derived", (
        "the serving label a consumer switches on")
    assert got["derived_from"] == sorted(beta_ids)
    assert got["source_revisions"] == {sid: 1 for sid in sorted(beta_ids)}
    assert got["rule_version"] == consolidation.RULE_VERSION
    assert got["state"] == "current"

    # The stale guard, measured inside the window it protects. A source write
    # first makes the group's key NEW (so the run does not skip it) — and the
    # old view stays current: a source change alone does not dirty it. The run
    # then re-summarizes, and a SECOND write lands while the summary is being
    # generated: that is the move the publish-time guard refuses.
    async with database.AsyncSessionLocal() as other:
        await other.execute(update(Memory).where(Memory.id == uuid.UUID(alpha_ids[0]))
                            .values(revision=Memory.revision + 1))
        await other.commit()
    async with live.sessions() as db:
        rows = await _fresh(db, alice.id)
    alpha_summary = [row for row in rows.values()
                     if row.extra_metadata.get(C.CM_DERIVED_KEY)
                     and alpha_ids[0] in row.extra_metadata.get(C.CM_DERIVED_FROM, [])]
    assert len(alpha_summary) == 1
    alpha_summary_id = alpha_summary[0].id
    assert state_of(alpha_summary[0]) == "current", (
        "a source change alone does not dirty the view — the next pass's guard walk does")

    race["pending"] = uuid.UUID(alpha_ids[0])
    async with live.sessions() as db:
        stale = await consolidation.run_consolidation(db, alice.id, budget=10)
    assert race["pending"] is None, "the interleave really happened"
    assert stale.refused == 1 and stale.dirtied == 1 and stale.published == [] and stale.truncated is False, stale
    async with live.sessions() as db:
        rows = await _fresh(db, alice.id)
    assert state_of(rows[alpha_summary_id]) == "dirty", (
        "the pass refused the publish but left the stale view serving")
    assert state_of(rows[uuid.UUID(beta_summary)]) == "current", "an unrelated summary was dirtied"


# ── 6. retention opt-in, per row ────────────────────────────────────────────


async def test_retention_is_opt_in_per_row_and_pin_exempt(live):
    """§8.1: auto expiration is opt-in, audited, and pin-protected.

    Default OFF is measured, not assumed: a sweep with nobody opted in
    invalidates nothing. An opted-in user's rows past their own window are
    invalidated one by one — the pinned row is exempt, the fresh one is not
    expired, the audit row carries the reason, the payload-refresh intent is
    queued, and a second sweep is a no-op.
    """
    alice = live.alice
    now = datetime.now(UTC)
    async with live.sessions() as db:
        old = _memory(alice.id, "retention old")
        old.indexed_at = now - timedelta(days=40)
        old_pinned = _memory(alice.id, "retention old pinned", pinned=True)
        old_pinned.indexed_at = now - timedelta(days=40)
        fresh = _memory(alice.id, "retention fresh")
        db.add_all([old, old_pinned, fresh])
        await db.commit()
        old_id, pinned_id, fresh_id = old.id, old_pinned.id, fresh.id

    async with live.sessions() as db:
        idle = await run_retention(db)
    assert idle == RetentionReport(0, 0), "no user opted in: the sweep must visit nobody"
    async with live.sessions() as db:
        rows = await _fresh(db, alice.id)
    assert state_of(rows[old_id]) == "current"

    async with live.sessions() as db:
        user = await db.get(User, alice.id)
        user.retention_enabled, user.retention_days = True, 30
        await db.commit()

    async with live.sessions() as db:
        swept = await run_retention(db)
    assert swept == RetentionReport(1, 1), swept
    async with live.sessions() as db:
        rows = await _fresh(db, alice.id)
        audits = (await db.execute(select(MemoryAccessLog).where(
            MemoryAccessLog.user_id == alice.id,
            MemoryAccessLog.action == RETENTION_REASON))).scalars().all()
        intents = await _intents(live)
    assert state_of(rows[old_id]) == "invalidated"
    assert state_of(rows[pinned_id]) == "current", "pin protects against auto-retention"
    assert state_of(rows[fresh_id]) == "current", "a fresh row is not expired"
    assert rows[old_id].content == "retention old", "retention is a serving decision, not a shredder"
    assert [a.memory_id for a in audits] == [old_id], f"one audit row per expired memory: {audits}"
    assert audits[0].detail["reason"] == RETENTION_REASON and audits[0].detail["retention_days"] == 30
    pending = [i for i in intents if i.entity_id == old_id.hex and i.status == "pending"]
    assert pending, "the payload refresh intent is what reaches the vector (R37)"

    async with live.sessions() as db:
        second = await run_retention(db)
    assert second == RetentionReport(1, 0), "an already-invalidated row is never selected twice"


# ── 7. the opt-in behind the REAL HTTP boundary ─────────────────────────────


async def test_patch_settings_is_a_full_replace_and_refuses_enable_without_a_window(live):
    """The setting is written through the real API surface (F1-T6 semantics).

    Full replace: a body that omits a field writes that field's default — a
    ``PATCH`` with ``{}`` turns retention OFF (the fail-safe direction), it is
    never a merge. "Enabled without a window" is refused at the boundary
    because the sweep could never run for it.
    """
    async with _auth_client(live.alice) as client:
        on = await client.patch("/api/v1/users/me/settings",
                                json={"retention_enabled": True, "retention_days": 30})
        assert on.status_code == 200 and on.json() == {"retention_enabled": True, "retention_days": 30}

        off = await client.patch("/api/v1/users/me/settings", json={})
        assert off.status_code == 200 and off.json() == {"retention_enabled": False, "retention_days": None}, (
            "a partial body is a full replace: the omitted fields took their defaults")

        refused = await client.patch("/api/v1/users/me/settings", json={"retention_enabled": True})
        assert refused.status_code == 422, refused.text

        window_only = await client.patch("/api/v1/users/me/settings", json={"retention_days": 7})
        assert window_only.json() == {"retention_enabled": False, "retention_days": 7}

        me = await client.get("/api/v1/users/me")
        assert me.status_code == 200 and me.json()["retention_days"] == 7


# ── 8. serving exclusions ───────────────────────────────────────────────────


async def test_serving_surfaces_exclude_dirty_and_invalidated_rows(live, monkeypatch, llm_seam):
    """Dirty and invalidated rows are never served — but they are not gone.

    The rows get REAL vectors, so a reader that forgot the rule would rank
    them. The REST list, the shared recall and the MCP search all exclude
    them; the direct reads still answer, labelled, and the timeline keeps the
    invalidated row as history while dropping the dirty one; MCP
    ``include_history`` widens to superseded ONLY.
    """
    alice = live.alice
    async with live.sessions() as db:
        dirty = _memory(alice.id, "p4b dirty row", extra_metadata={C.CM_DERIVED_DIRTY: True})
        invalidated = _memory(alice.id, "p4b invalidated row", extra_metadata={C.CM_INVALIDATED: True})
        db.add_all([dirty, invalidated])
        await db.commit()
        dirty_id, invalidated_id = dirty.id, invalidated.id
    vector_store.upsert_memories_sync([dirty, invalidated])  # the REAL writer
    assert await vector_store.get_memory_ids_present([str(dirty_id), str(invalidated_id)]) == {
        str(dirty_id), str(invalidated_id)}, "the fixture must put pieces a reader could leak"

    async with _auth_client(alice) as client:
        listing = (await client.get("/api/v1/memories")).json()
    served = {item["id"] for item in listing["items"]}
    assert str(dirty_id) not in served and str(invalidated_id) not in served
    assert str(live.alice_superseded.id) in served, "superseded rows are history: the list serves them"

    async with live.sessions() as db:
        ranked, _why = await MemoryRetriever(db, alice.id).recall_ids("p4b dirty row", top_k=10)
    ranked_ids = {str(memory_id) for memory_id, _score in ranked}
    assert str(dirty_id) not in ranked_ids and str(invalidated_id) not in ranked_ids

    _mcp_seams(monkeypatch, _writer_principal(live))
    for query in ("p4b dirty row", "p4b invalidated row"):
        search = await hub_tools.search_memory(query)
        assert {row["id"] for row in search["results"]} & {str(dirty_id), str(invalidated_id)} == set()
    history = await hub_tools.search_memory("p4b invalidated row", include_history=True)
    assert str(invalidated_id) not in {row["id"] for row in history["results"]}, (
        "history widens to superseded, never to invalidated")

    assert (await hub_tools.get_memory(str(dirty_id)))["state"] == "dirty"
    got = await hub_tools.get_memory(str(invalidated_id))
    assert got["state"] == "invalidated" and got["content"] == "p4b invalidated row"

    timeline = await hub_tools.timeline(str(live.alice_current.id), window=10)
    neighbours = {row["id"]: row for row in timeline["before"] + timeline["after"]}
    assert str(invalidated_id) in neighbours, "an invalidated neighbour is labelled history"
    assert str(dirty_id) not in neighbours, "a dirty neighbour is wrong data, not history"


# ── 9. the CI pin (parse the workflow, not a copy of it) ────────────────────


def _workflow() -> dict:
    return yaml.safe_load(CI_YML.read_text())


def _step(job: dict, name: str) -> dict:
    for step in job["steps"]:
        if step.get("name") == name:
            return step
    raise AssertionError(f"CI step {name!r} is gone: {[s.get('name') for s in job['steps']]}")


def test_the_workflow_runs_this_gate_and_names_only_paths_that_exist():
    """R30-style: dropping this gate from CI (or narrowing its env, or wiring a
    path that no longer exists) fails HERE instead of silently.

    pytest exits 4 on a path that does not exist, so every ``tests/...`` token
    the workflow, the Makefile or the four P4b docs name is checked to exist.
    """
    workflow = _workflow()
    step = _step(workflow["jobs"]["test"], CI_STEP_NAME)
    run = step["run"]

    assert step["env"]["DATABASE_URL"].startswith("sqlite+aiosqlite:////tmp/"), (
        "the P4b step must pin its own disposable SQLite file, like every step above it")
    assert "continue-on-error" not in step and "if" not in step, (
        f"the {CI_STEP_NAME!r} step can be stood down without touching its run command: {sorted(step)}")
    for suite in P4B_CI_SUITES:
        assert suite in run, f"{suite} is not wired into the {CI_STEP_NAME!r} step"
    assert "--ignore" not in run, (
        f"an --ignore in the {CI_STEP_NAME!r} step skips a suite it claims to run: {run!r}")
    all_runs = "\n".join(s.get("run", "") for job in workflow["jobs"].values() for s in job["steps"])
    for suite in P4B_WIRED_ELSEWHERE:
        # A directory-scoped step (`tests/mcp_hub`) wires every module under it.
        assert suite in all_runs or suite in run or suite.rsplit("/", 1)[0] in all_runs, (
            f"{suite} is wired into no CI step at all")
    assert GATE_MODULE in run, "the acceptance gate itself must be in the step"

    # No dead path: pytest exits 4 on a missing target, and a stale mention in
    # the docs is the same lie told to a reader.
    sources = [CI_YML, REPO / "Makefile", *P4B_DOCS]
    pattern = re.compile(r"tests/[A-Za-z0-9_./*-]+")
    missing: dict[str, str] = {}
    for source in sources:
        for token in pattern.findall(source.read_text()):
            token = token.rstrip(".,;:)`'\"")
            if not list(REPO.glob(token)):
                missing[token] = source.relative_to(REPO).as_posix()
    assert missing == {}, f"cite tests/ paths that do not exist: {missing}"


# ── hygiene: the vocabulary the bullets are written against ─────────────────


def test_the_gate_pins_the_lifecycle_vocabulary_and_the_ladder_stamp():
    """A silent drift in either constant would leave these bullets checking nothing."""
    assert database.SQLITE_SCHEMA_VERSION == 7, "the gate is written for the terminal SQLite stamp"
    assert C.MEMORY_STATES == ("current", "superseded", "dirty", "needs-check", "invalidated")
    assert PERSONAL == "personal"
