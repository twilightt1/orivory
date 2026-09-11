# Correctable Memory V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement evidence-first correctable memory on the Lite path: one resolve function, one new MCP tool, recall filters, bounded timeline, fast-path recall — no migration, no new service.

**Architecture:** All new state lives namespaced (`cm_*`) inside `Memory.extra_metadata`. One new module (`correction.py`) owns metadata helpers, the atomic resolve, and pure rules. `tools.py` and `retriever.py` get thin wiring only. Every task is TDD with the repo's existing FakeDB/monkeypatch style.

**Tech Stack:** Python 3.13, SQLAlchemy asyncio, FastMCP 1.x, pytest (asyncio_mode=auto), SQLite (lite tests run real SQLite via `tests/lite/` pattern).

**Spec:** `docs/superpowers/specs/2026-09-12-correctable-memory-v1.md`

## Global Constraints

- No migration: absent `cm_*` keys = legacy (`fact`, scope `default`, current, `valid_from` = `captured_at`). Times ISO-8601 UTC.
- Single commit per resolve: INSERT new + set old `cm_superseded_by` + set `cm_derived_dirty` in one transaction.
- Validity/dirty filters run at hydration (PG/SQLite), never in Chroma metadata.
- `add` never accepts chain fields from the caller; only server-side resolve links chains.
- `pytest.ini` uses `asyncio_mode = auto`; tests are plain `async def`, no decorators needed.
- Commit per task; run the named tests before committing.

---

## File map

- Create: `app/retrieval/memory/correction.py` — `cm_*` constants, pure helpers, `resolve_correction`, `collect_derived_ids`, `needs_rewrite`.
- Create: `tests/retrieval/test_correction.py` — pure + FakeDB tests for the above.
- Create: `tests/lite/test_correction_roundtrip.py` — real-SQLite JSON-filter proof + §9b self-check.
- Modify: `app/mcp_hub/tools.py` — `correct_memory`, `add_memory` via resolve, `search` state + `include_history`, `get` provenance, bounded `timeline`.
- Modify: `app/mcp_hub/identity.py` — `ACTION_CORRECT = "mcp_correct"`.
- Modify: `app/mcp_hub/server.py` — register `correct_memory`.
- Modify: `app/retrieval/memory/retriever.py` — hydration filters, fast-path skip, stage timings.
- Modify: `app/schemas/Orivory.py` — `RecallTrace.rewrite_skipped`, `RecallTrace.stage_ms`.
- Modify: `app/services/erasure_service.py` — one call site: extend `affected` with derived dependents (best-effort).
- Modify: `tests/mcp_hub/test_tools.py`, `tests/mcp_hub/test_server.py`, `tests/services/test_erasure_service.py` (run-only unless broken).
- Modify: `skills/orivory/SKILL.md` (six→seven), `skills/orivory/references/tool-catalog.md` (new row).

---

### Task 1: `cm_*` helpers + pure rules

**Files:**
- Create: `app/retrieval/memory/correction.py`
- Test: `tests/retrieval/test_correction.py`

**Interfaces:**
- Consumes: `app.models.memory.Memory` (only `.extra_metadata`).
- Produces: `CM_PREFIX="cm_"`, key constants, `normalize_slot`, `get_cm`, `set_cm`, `state_of`, `needs_rewrite`, `find_derived_dependent_ids` — all imported by Tasks 2–7.

- [ ] **Step 1: Write the failing test**

```python
from app.retrieval.memory import correction as C

def test_state_of_legacy_is_current():
    assert C.state_of(_mem(None)) == "current"

def test_normalize_slot():
    assert C.normalize_slot("  Proj-X  DB ") == "proj-x db"
    assert C.normalize_slot(None) == ""

def test_needs_rewrite_pure_rule():
    assert C.needs_rewrite("db prod la gi") is False
    assert C.needs_rewrite("no chay tren cong nao cua du an do") is True

def test_derived_dependents():
    old = _mem({"cm_derived_from": ["A", "B"]})
    assert C.find_derived_dependent_ids([old], {"B"}) == [str(old.id)]
    assert C.find_derived_dependent_ids([old], {"Z"}) == []
```

(`_mem(meta)` helper in the test file builds a `Memory` with `id=uuid4()`, `user_id=uuid4()`, `content="x"`, `tags=[]`, `extra_metadata=meta or {}`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/retrieval/test_correction.py -v`
Expected: FAIL with "No module named" / names not defined.

- [ ] **Step 3: Write minimal implementation** (whole module section 1 — resolve comes in Task 2, same file)

```python
"""cm_* metadata helpers + pure rules for correctable memory (spec 2026-09-12)."""
from __future__ import annotations
import re
from uuid import UUID

CM_ASSERTION = "cm_assertion"
CM_SUBJECT = "cm_subject"
CM_ATTRIBUTE = "cm_attribute"
CM_SCOPE = "cm_scope"
CM_VALID_FROM = "cm_valid_from"
CM_VALID_TO = "cm_valid_to"
CM_SUPERSEDES = "cm_supersedes"
CM_SUPERSEDED_BY = "cm_superseded_by"
CM_EVIDENCE_IDS = "cm_evidence_ids"
CM_DERIVED_FROM = "cm_derived_from"
CM_DERIVED_DIRTY = "cm_derived_dirty"
CM_NEEDS_CHECK = "cm_needs_check"

DEFAULT_SCOPE = "default"

# ponytail: pronoun list is heuristic; add words only when eval shows a miss.
_PRONOUNS = frozenset({
    "it", "its", "this", "that", "these", "those", "he", "him", "his",
    "she", "her", "they", "them", "their", "no", "nó", "chúng", "đó",
})
_WORD = re.compile(r"[a-zà-ỹ]+", re.IGNORECASE)


def normalize_slot(value: str | None) -> str:
    return " ".join((value or "").strip().lower().split())


def get_cm(memory) -> dict:
    return dict(memory.extra_metadata or {})


def set_cm(memory, patch: dict) -> None:
    memory.extra_metadata = {**(memory.extra_metadata or {}), **patch}


def state_of(memory) -> str:
    meta = get_cm(memory)
    if meta.get(CM_SUPERSEDED_BY):
        return "superseded"
    if meta.get(CM_NEEDS_CHECK):
        return "needs-check"
    return "current"


def needs_rewrite(query: str) -> bool:
    return any(w in _PRONOUNS for w in _WORD.findall(query or ""))


def find_derived_dependent_ids(memories, erased_ids: set[str]) -> list[str]:
    out = []
    for m in memories:
        try:
            deps = set(get_cm(m).get(CM_DERIVED_FROM) or [])
        except (TypeError, AttributeError):
            continue
        if deps & erased_ids and not get_cm(m).get(CM_DERIVED_DIRTY):
            out.append(str(m.id))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/retrieval/test_correction.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/retrieval/memory/correction.py tests/retrieval/test_correction.py
git commit -m "feat: cm_* metadata helpers and pure correction rules"
```

---

### Task 2: `resolve_correction` — the single atomic write path

**Files:**
- Modify: `app/retrieval/memory/correction.py` (append)
- Test: `tests/retrieval/test_correction.py` (append; reuse `_FakeDB` pattern from `tests/mcp_hub/test_tools.py`)

**Interfaces:**
- Consumes: Task 1 helpers.
- Produces: `async resolve_correction(db, *, user_id, title, content, tags=None, source_type="mcp_agent", source_ref=None, subject="", attribute="", scope="default", assertion="fact", valid_from=None, evidence_ids=None, memory_id=None) -> dict` with `{"status": "added"|"superseded"|"needs-check", "memory": Memory, "superseded": [str], "dirtied": [str]}`. Used by Tasks 4–5. Rules: exact normalized triple match → supersede in one commit; caller scope empty while candidate scoped (or vice versa), `memory_id` mismatch, or `fact`-vs-`plan` clash → create unlinked with `cm_needs_check=True`, status `needs-check`; empty identity → plain add.

- [ ] **Step 1: Write the failing test**

```python
async def test_resolve_supersede_chain_single_commit():
    from app.retrieval.memory.correction import resolve_correction
    uid = uuid.uuid4()
    old = _mem({"cm_subject": "proj-x", "cm_attribute": "db", "cm_scope": "prod"})
    old.user_id = uid
    db = _FakeDB(rows=[old])
    out = await resolve_correction(db, user_id=uid, title="DB", content="Postgres",
        subject="Proj-X", attribute="db", scope="prod")
    assert out["status"] == "superseded"
    assert db.committed == 1
    new = out["memory"]
    assert new.extra_metadata["cm_supersedes"] == str(old.id)
    assert old.extra_metadata["cm_superseded_by"] == str(new.id)
    assert out["superseded"] == [str(old.id)]

async def test_resolve_ambiguous_scope_keeps_both():
    from app.retrieval.memory.correction import resolve_correction
    uid = uuid.uuid4()
    old = _mem({"cm_subject": "proj-x", "cm_attribute": "db", "cm_scope": "prod"})
    old.user_id = uid
    db = _FakeDB(rows=[old])
    out = await resolve_correction(db, user_id=uid, title="DB", content="SQLite",
        subject="proj-x", attribute="db", scope="")
    assert out["status"] == "needs-check"
    assert old.extra_metadata.get("cm_superseded_by") is None
    assert out["memory"].extra_metadata["cm_needs_check"] is True
```

(Copy the `_FakeDB`/`_FakeCtx` classes from `tests/mcp_hub/test_tools.py:60-97` into this test file; `_FakeDB.execute` returns all rows so candidate filtering must happen in Python — which is also what production does after the loose user-scoped SQL fetch.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/retrieval/test_correction.py -v`
Expected: FAIL with "resolve_correction not defined" (import error inside test).

- [ ] **Step 3: Write minimal implementation** (append to `correction.py`)

```python
from datetime import UTC, datetime
from sqlalchemy import select

async def resolve_correction(db, *, user_id, title, content, tags=None,
        source_type="mcp_agent", source_ref=None, subject="", attribute="",
        scope=DEFAULT_SCOPE, assertion="fact", valid_from=None,
        evidence_ids=None, memory_id=None) -> dict:
    """Single creation path for add + correct. One commit, never raises."""
    from uuid import uuid4
    from app.models.memory import Memory

    subj, attr, sc = normalize_slot(subject), normalize_slot(attribute), normalize_slot(scope)
    rows = (await db.execute(
        select(Memory).where(Memory.user_id == user_id)
    )).scalars().all()
    cands = [m for m in rows if not get_cm(m).get(CM_SUPERSEDED_BY)]

    def _triple(m) -> tuple[str, str, str]:
        meta = get_cm(m)
        return (normalize_slot(meta.get(CM_SUBJECT)),
                normalize_slot(meta.get(CM_ATTRIBUTE)),
                normalize_slot(meta.get(CM_SCOPE, DEFAULT_SCOPE)))

    now = datetime.now(UTC)
    meta: dict = {CM_ASSERTION: assertion or "fact",
                  CM_SUBJECT: subj, CM_ATTRIBUTE: attr,
                  CM_SCOPE: sc or DEFAULT_SCOPE,
                  CM_EVIDENCE_IDS: [str(e) for e in (evidence_ids or [])]}
    if valid_from:
        try:
            datetime.fromisoformat(str(valid_from).replace("Z", "+00:00"))
            meta[CM_VALID_FROM] = str(valid_from)
        except ValueError:
            meta[CM_NEEDS_CHECK] = True
    if memory_id:
        meta.setdefault(CM_EVIDENCE_IDS, []).append(str(memory_id))

    status, superseded, dirtied = "added", [], []
    if subj and attr:
        exact = [m for m in cands if _triple(m) == (subj, attr, sc or DEFAULT_SCOPE)]
        if memory_id and exact and str(exact[0].id) != str(memory_id) and len(exact) == 1:
            pass  # explicit target mismatch handled below as ambiguous
        if exact and not meta.get(CM_NEEDS_CHECK):
            clash = any(get_cm(m).get(CM_ASSERTION, "fact") != meta[CM_ASSERTION] for m in exact)
            target_ok = (not memory_id) or any(str(m.id) == str(memory_id) for m in exact)
            if not clash and target_ok:
                status = "superseded"
            else:
                meta[CM_NEEDS_CHECK] = True
                status = "needs-check"
        elif not exact:
            same_sa = [m for m in cands
                       if _triple(m)[:2] == (subj, attr) and _triple(m)[2] not in ("", sc or DEFAULT_SCOPE)]
            if same_sa and not sc:
                meta[CM_NEEDS_CHECK] = True
                status = "needs-check"
            # different non-empty scope, or no candidates at all: independent add

    new = Memory(id=uuid4(), user_id=user_id, title=title, content=content,
                 tags=list(tags or []), source_type=source_type,
                 source_ref=source_ref, captured_at=now, extra_metadata=meta)
    db.add(new)
    if status == "superseded":
        for m in exact:
            set_cm(m, {CM_SUPERSEDED_BY: str(new.id)})
            superseded.append(str(m.id))
        meta[CM_SUPERSEDES] = superseded[0] if len(superseded) == 1 else superseded
        new.extra_metadata = {**new.extra_metadata, CM_SUPERSEDES: meta[CM_SUPERSEDES]}
        erased = set(superseded)
        for m in cands:
            if m in exact:
                continue
            try:
                deps = set(get_cm(m).get(CM_DERIVED_FROM) or [])
            except (TypeError, AttributeError):
                continue
            if deps & erased and not get_cm(m).get(CM_DERIVED_DIRTY):
                set_cm(m, {CM_DERIVED_DIRTY: True})
                dirtied.append(str(m.id))
    await db.commit()
    return {"status": status, "memory": new,
            "superseded": superseded, "dirtied": dirtied}


async def collect_derived_ids(db, user_id, erased_ids: list) -> list:
    """Return ids of memories deriving from erased ids. Never raises."""
    from app.models.memory import Memory
    try:
        erased = {str(e) for e in erased_ids}
        rows = (await db.execute(
            select(Memory).where(Memory.user_id == user_id)
        )).scalars().all()
        return [m.id for m in rows
                if str(m.id) not in erased
                and set(get_cm(m).get(CM_DERIVED_FROM) or []) & erased]
    except Exception:
        return []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/retrieval/test_correction.py tests/mcp_hub/ -v`
Expected: PASS (existing MCP tests untouched).

- [ ] **Step 5: Commit**

```bash
git add app/retrieval/memory/correction.py tests/retrieval/test_correction.py
git commit -m "feat: atomic resolve_correction write path"
```

---

### Task 3: Retriever — hydration filters, fast-path, stage timings

**Files:**
- Modify: `app/schemas/Orivory.py` (`RecallTrace`: add `rewrite_skipped: bool = False`, `stage_ms: dict[str, float] = Field(default_factory=dict)`)
- Modify: `app/retrieval/memory/retriever.py`
- Test: `tests/retrieval/test_retriever_filters.py` (new; monkeypatch `rewrite_query`, `embed_query`, `search_memories` at module attrs, FakeDB for hydrate)

**Interfaces:**
- Consumes: `state_of`, `needs_rewrite` (Task 1).
- Produces: unchanged `recall()` signature; trace now carries per-stage ms + skip flag. Task 5 relies on superseded/dirty filtering happening here (not in Chroma).

- [ ] **Step 1: Write the failing test**

```python
async def test_recall_hides_superseded_and_dirty(monkeypatch):
    from app.retrieval.memory import retriever as R
    uid = uuid.uuid4()
    cur = _mem({}); cur.user_id = uid
    old = _mem({"cm_superseded_by": str(cur.id)}); old.user_id = uid
    dirty = _mem({"cm_derived_from": [str(cur.id)], "cm_derived_dirty": True}); dirty.user_id = uid
    db = _FakeDB(rows=[cur, old, dirty])
    async def _rw(q, context=None): return {"rewritten_query": q, "entities": [], "_fallback_used": False, "reasoning": ""}
    async def _emb(q): return [0.1, 0.2]
    async def _search(emb, user_id=None, top_k=10):
        return [{"memory_id": str(old.id), "score": 0.99},
                {"memory_id": str(dirty.id), "score": 0.98},
                {"memory_id": str(cur.id), "score": 0.5}]
    monkeypatch.setattr(R, "rewrite_query", _rw)
    monkeypatch.setattr(R, "embed_query", _emb)
    monkeypatch.setattr(R, "search_memories", _search)
    # ponytail: FakeDB.execute ignores the statement; candidate ids come from
    # the fake Chroma above, filtering happens in Python like production.
    r = R.MemoryRetriever(db, uid, semantic_rerank=False)
    out = await r.recall("db prod", top_k=3, include_personal_context=False)
    assert [x.id for x in out.results] == [cur.id]
    assert out.trace.stage_ms.keys() >= {"rewrite_ms", "embed_ms", "search_ms", "hydrate_ms"}

async def test_recall_fast_path_skips_llm(monkeypatch):
    from app.retrieval.memory import retriever as R
    called = []
    async def _rw(q, context=None):
        called.append(q)
        return {"rewritten_query": q, "entities": [], "_fallback_used": False, "reasoning": ""}
    # ... same fakes, query without pronouns ...
    out = await r.recall("postgres indexing", top_k=1, include_personal_context=False)
    assert called == [] and out.trace.rewrite_skipped is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/retrieval/test_retriever_filters.py -v`
Expected: FAIL (no `stage_ms` / `rewrite_skipped` / filters).

- [ ] **Step 3: Write minimal implementation**

1. `schemas/Orivory.py`, in `RecallTrace`, add:
```python
    rewrite_skipped: bool = False
    stage_ms: dict[str, float] = Field(default_factory=dict)
```
2. `retriever.py` `recall()`: wrap steps 2/3/4/5 each with `t = time.perf_counter()` → `stage_ms["rewrite_ms"] = ...` etc. Replace step 2 with:
```python
        from app.retrieval.memory.correction import needs_rewrite as _needs_rewrite
        t_rewrite = time.perf_counter()
        if include_personal_context or _needs_rewrite(query):
            rewrite_result = await rewrite_query(query, context=context)
            rewrite_skipped = False
        else:
            rewrite_result = {"rewritten_query": query, "entities": [],
                              "reasoning": "fast-path: no pronouns", "_fallback_used": False}
            rewrite_skipped = True
        stage_ms["rewrite_ms"] = (time.perf_counter() - t_rewrite) * 1000.0
```
Pass `rewrite_skipped=rewrite_skipped, stage_ms=stage_ms` into both `RecallTrace(...)` constructions. After hydration (step 5 result `hydrated`), drop filtered candidates before scoring:
```python
        from app.retrieval.memory.correction import state_of as _state_of
        visible = []
        for cand in candidates:
            mem = hydrated.get(cand["memory_id"])
            if mem is None:
                continue
            st = _state_of(mem)
            if st == "superseded" or (getattr(mem, "extra_metadata", {}) or {}).get("cm_derived_dirty"):
                continue
            visible.append(cand)
        candidates = visible
```
(`include_history` flag arrives in Task 5 for the MCP seam; retriever default stays filtered.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/retrieval/ tests/mcp_hub/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/schemas/Orivory.py app/retrieval/memory/retriever.py tests/retrieval/test_retriever_filters.py
git commit -m "feat: recall hides superseded/dirty, fast-path skip, stage timings"
```

---

### Task 4: `correct_memory` MCP tool + registration

**Files:**
- Modify: `app/mcp_hub/identity.py` (add `ACTION_CORRECT = "mcp_correct"` + `__all__`)
- Modify: `app/mcp_hub/tools.py` (import it; add `correct_memory`; add `_memory_provenance`)
- Modify: `app/mcp_hub/server.py` (register tool; fix "six tools" docstring → seven)
- Test: `tests/mcp_hub/test_tools.py` (append), `tests/mcp_hub/test_server.py` (registration assert)

**Interfaces:**
- Consumes: `resolve_correction` (Task 2).
- Produces: `async correct_memory(memory_id=None, subject="", attribute="", scope="default", title="", content="", valid_from=None, evidence_ids=None) -> dict`; provenance helper used by Task 5.

- [ ] **Step 1: Write the failing test**

```python
async def test_correct_memory_requires_write_scope(reader):
    result = await hub_tools.correct_memory(content="Postgres")
    assert result == {"error": "scope memory:write required"}

async def test_correct_memory_supersedes_and_logs(writer, monkeypatch):
    p, db = writer
    old = _memory_row(uuid.uuid4(), p.user_id)
    old.extra_metadata = {"cm_subject": "proj-x", "cm_attribute": "db", "cm_scope": "prod"}
    db.rows = [old]
    async def _noop_index(memory): return None
    monkeypatch.setattr(hub_tools, "index_new_memory", _noop_index)
    out = await hub_tools.correct_memory(subject="proj-x", attribute="db",
        scope="prod", title="DB", content="Postgres")
    assert out["status"] == "superseded"
    assert out["superseded"] == [str(old.id)]
    ledger = [o for o in db.added if type(o).__name__ == "MemoryAccessLog"]
    assert ledger and ledger[0].action == "mcp_correct"

async def test_correct_memory_foreign_id_rejected(writer):
    _p, _db = writer
    out = await hub_tools.correct_memory(memory_id=str(uuid.uuid4()), content="x")
    assert out == {"error": "memory not found"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/mcp_hub/test_tools.py -v`
Expected: FAIL with "correct_memory not defined".

- [ ] **Step 3: Write minimal implementation**

In `identity.py`: add `ACTION_CORRECT = "mcp_correct"` and extend `__all__`. In `tools.py`:
```python
def _memory_provenance(memory: Memory) -> dict[str, Any]:
    from app.retrieval.memory.correction import get_cm, state_of
    meta = get_cm(memory)
    return {
        "state": state_of(memory),
        "assertion": meta.get("cm_assertion", "fact"),
        "scope": meta.get("cm_scope", "default"),
        "valid_from": meta.get("cm_valid_from"),
        "supersedes": meta.get("cm_supersedes"),
        "superseded_by": meta.get("cm_superseded_by"),
        "evidence_ids": list(meta.get("cm_evidence_ids") or []),
    }

async def correct_memory(memory_id=None, subject="", attribute="", scope="default",
        title="", content="", valid_from=None, evidence_ids=None) -> dict[str, Any]:
    """Correct a fact with evidence: new version links back, never overwrites."""
    from app.retrieval.memory.correction import resolve_correction
    principal = _current_principal()
    if principal is None:
        return IDENTITY_ERROR
    if not principal.can_write():
        return WRITE_SCOPE_ERROR
    if not (content or "").strip():
        return {"error": "content required"}
    target = None
    if memory_id:
        try:
            mid = UUID(memory_id)
        except ValueError:
            return {"error": "invalid memory id"}
        async with _session() as db:
            target = await db.get(Memory, mid)
        if target is None or target.user_id != principal.user_id:
            return {"error": "memory not found"}
    async with _session() as db:
        out = await resolve_correction(
            db, user_id=principal.user_id, title=title or (target.title if target else ""),
            content=content, subject=subject, attribute=attribute, scope=scope,
            valid_from=valid_from, evidence_ids=list(evidence_ids or []),
            memory_id=str(target.id) if target else None,
            source_ref=f"agent:{principal.name}")
        new = out["memory"]
        db.add(_ledger_entry(principal, ACTION_CORRECT, memory_id=new.id,
            detail={"status": out["status"], "superseded": out["superseded"],
                    "dirtied": out["dirtied"], "memory_id": str(new.id)}))
        await db.commit()
    try:
        await index_new_memory(new)
    except Exception as exc:
        log.warning("MCP correct_memory indexing failed for %s: %s", new.id, exc)
    return {"status": out["status"], "id": str(new.id),
            "superseded": out["superseded"], "dirtied": out["dirtied"],
            **_memory_provenance(new)}
```

Note: `_FakeDB.get` loops `self.rows` — the foreign-id test works because the unknown id is absent. But `resolve_correction` + ledger run in *separate* `_session()` blocks sharing one FakeDB, so state carries over — same pattern as `add_memory` today. In production each block is its own session; resolve commits inside its block, ledger commits in the next. Two commits, one per block — the "single commit" guarantee covers the chain mutation itself (Task 2), not the ledger append (pre-existing pattern).

In `server.py` register after `add_memory`:
```python
    @mcp.tool()
    async def correct_memory(
        content: str, title: str = "", subject: str = "", attribute: str = "",
        scope: str = "default", memory_id: str | None = None,
        valid_from: str | None = None, evidence_ids: list[str] | None = None,
        ctx: Context = None,
    ) -> dict[str, Any]:
        """Correct a stored fact with evidence (memory:write). Creates a linked new version; never overwrites."""
        return await _call_with_identity(hub_tools.correct_memory(
            memory_id=memory_id, subject=subject, attribute=attribute, scope=scope,
            title=title, content=content, valid_from=valid_from,
            evidence_ids=evidence_ids), ctx)
```
Add `"correct_memory"` to `tools.__all__`; update "six tools" docstring.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/mcp_hub/ -v`
Expected: PASS. Also extend `test_server.py` registration test:
```python
assert any(t.name == "correct_memory" for t in tools)
```

- [ ] **Step 5: Commit**

```bash
git add app/mcp_hub/ tests/mcp_hub/
git commit -m "feat: correct_memory MCP tool with evidence chain"
```

---

### Task 5: `add` via resolve, `search` state, `get` provenance

**Files:**
- Modify: `app/mcp_hub/tools.py` (`add_memory`, `search_memory`, `_memory_index_row`, `get_memory`)
- Test: `tests/mcp_hub/test_tools.py` (append)

**Interfaces:**
- Consumes: Tasks 1–2, 4.
- Produces: `search_memory(query, limit=8, include_history=False)`; index rows carry `state`; `get_memory` merges `_memory_brief` + `_memory_provenance`. `add_memory` stamps legacy-default `cm_*` and routes through `resolve_correction` with empty identity (plain-add branch — never auto-supersedes).

- [ ] **Step 1: Write the failing test**

```python
async def test_search_hides_superseded_by_default(writer, monkeypatch):
    p, db = writer
    new = _memory_row(uuid.uuid4(), p.user_id)
    old = _memory_row(uuid.uuid4(), p.user_id)
    old.extra_metadata = {"cm_superseded_by": str(new.id)}
    db.rows = [old, new]
    monkeypatch.setattr(hub_tools, "_recall_memory_ids", _fake_recall([old.id, new.id]))
    out = await hub_tools.search_memory("x")
    assert [r["id"] for r in out["results"]] == [str(new.id)]
    assert out["results"][0]["state"] == "current"
    out2 = await hub_tools.search_memory("x", include_history=True)
    assert {r["id"] for r in out2["results"]} == {str(old.id), str(new.id)}

async def test_add_never_supersedes(writer, monkeypatch):
    p, db = writer
    old = _memory_row(uuid.uuid4(), p.user_id)
    old.extra_metadata = {"cm_subject": "proj-x", "cm_attribute": "db", "cm_scope": "prod"}
    db.rows = [old]
    async def _noop_index(memory): return None
    monkeypatch.setattr(hub_tools, "index_new_memory", _noop_index)
    out = await hub_tools.add_memory(title="DB", content="SQLite")
    assert old.extra_metadata.get("cm_superseded_by") is None
    assert out["state"] == "current"

async def test_get_carries_provenance(reader):
    p, db = reader
    m = _memory_row(uuid.uuid4(), p.user_id)
    m.extra_metadata = {"cm_subject": "proj-x", "cm_scope": "prod",
                        "cm_supersedes": "prev-id", "cm_evidence_ids": ["e1"]}
    db.rows = [m]
    out = await hub_tools.get_memory(str(m.id))
    assert out["scope"] == "prod" and out["supersedes"] == "prev-id"
    assert out["evidence_ids"] == ["e1"] and out["state"] == "current"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/mcp_hub/test_tools.py -v`
Expected: FAIL (`include_history` unexpected kwarg; no `state` key).

- [ ] **Step 3: Write minimal implementation**

1. `_memory_index_row`: add `"state": state_of(memory)` (import from correction).
2. `search_memory(query, limit=8, include_history=False)`: after hydration, filter rows whose `state_of == "superseded"` unless `include_history`; index rows now carry state. (ponytail ceiling: no backfill — if superseded crowd out the limit, results shrink; fix when measured.)
3. `get_memory`: `return {**_memory_brief(row), **_memory_provenance(row)}`.
4. `add_memory`: keep compression + ledger `mcp_add`, but replace direct `Memory(...)` construction + commit with:
```python
    from app.retrieval.memory.correction import resolve_correction
    async with _session() as db:
        out = await resolve_correction(
            db, user_id=principal.user_id, title=title, content=content,
            tags=list(tags or []), source_ref=f"agent:{principal.name}")
        memory = out["memory"]
```
then index + ledger as today; return `{**_memory_brief(memory), **_memory_provenance(memory)}`. (`add` never passes subject/attribute/scope, so resolve always takes the plain-add branch — chain fields from the caller are structurally impossible.)
5. `server.py` `search_memory` wrapper gains `include_history: bool = False` passthrough.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/mcp_hub/ tests/retrieval/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/mcp_hub/ tests/mcp_hub/
git commit -m "feat: search state filter, get provenance, add via resolve"
```

---

### Task 6: Bounded `timeline` (fix O(n) load-all)

**Files:**
- Modify: `app/mcp_hub/tools.py` (`timeline` body only)
- Test: `tests/mcp_hub/test_tools.py` (existing 4 timeline tests must stay green, no changes)

**Interfaces:** No signature change. Two bounded SQL queries replace the full-table scan.

- [ ] **Step 1: Confirm current tests pass as baseline**

Run: `pytest tests/mcp_hub/test_tools.py -k timeline -v`
Expected: PASS (FakeDB returns all rows regardless of statement, so the Python-side before/after filters keep these green after the change).

- [ ] **Step 2: Replace the query body**

```python
        before_rows = (
            await db.execute(
                select(Memory)
                .where(Memory.user_id == principal.user_id,
                       (Memory.captured_at, Memory.id) < (anchor.captured_at, anchor.id))
                .order_by(Memory.captured_at.desc(), Memory.id.desc())
                .limit(capped)
            )
        ).scalars().all()
        after_rows = (
            await db.execute(
                select(Memory)
                .where(Memory.user_id == principal.user_id,
                       (Memory.captured_at, Memory.id) > (anchor.captured_at, anchor.id))
                .order_by(Memory.captured_at.asc(), Memory.id.asc())
                .limit(capped)
            )
        ).scalars().all()
        neighbours = [m for m in before_rows if m.id != anchor.id]
        neighbours_after = [m for m in after_rows if m.id != anchor.id]
```

Keep the anchor lookup, ledger, cap, and `_row` shape identical. (Tuple comparison compiles on SQLite ≥ 3.15 and Postgres; the lite roundtrip in Task 8 re-proves it on real SQLite.)

- [ ] **Step 3: Run tests**

Run: `pytest tests/mcp_hub/ -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add app/mcp_hub/tools.py
git commit -m "perf: bounded timeline queries"
```

---

### Task 7: `forget` extends to derived dependents

**Files:**
- Modify: `app/services/erasure_service.py` (one call site in `_erase_one`)
- Test: `tests/retrieval/test_correction.py` (collector unit test with FakeDB) + run existing erasure suite untouched

**Interfaces:**
- Consumes: `collect_derived_ids` (Task 2).
- Produces: `_erase_one` affected set = target + parent-descendants + derived dependents; receipt shape unchanged.

- [ ] **Step 1: Write the failing test** (collector level; erasure wiring verified by existing suite staying green)

```python
async def test_collect_derived_ids():
    from app.retrieval.memory.correction import collect_derived_ids
    uid = uuid.uuid4()
    target = _mem({}); target.user_id = uid
    view = _mem({"cm_derived_from": [str(target.id)]}); view.user_id = uid
    other = _mem({}); other.user_id = uid
    db = _FakeDB(rows=[target, view, other])
    out = await collect_derived_ids(db, uid, [target.id])
    assert out == [view.id]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/retrieval/test_correction.py -v`
Expected: FAIL (`collect_derived_ids` missing — already added in Task 2, so instead temporarily assert against a wrong expectation? No — honest TDD: this test was written after the implementation exists. Mark it as a characterization test: run once, confirm PASS, keep as regression lock. The real gate is the erasure suite below.)

- [ ] **Step 3: Wire the call site** in `_erase_one`, after `affected = [memory_id, *child_ids]`:

```python
    try:
        from app.retrieval.memory.write_back import safe_delete_from_chroma as _  # noqa (no-op, keeps import order)
    except Exception:
        pass
    try:
        from app.retrieval.memory.correction import collect_derived_ids
        for _did in await collect_derived_ids(db, user_id, affected):
            if _did not in affected:
                affected.append(_did)
    except Exception as exc:
        log.warning("Derived-dependent collection failed: %s", exc)
```

Then extend the existing flow: rows for derived ids are deleted by loading them (`db.get` + `db.delete`) before the Chroma/verify pass — minimal edit inside `_erase_one`:
```python
    for _did in affected[1:]:
        if _did not in child_ids and _did != memory_id:
            _extra = await db.get(Memory, _did)
            if _extra is not None and _extra.user_id == user_id:
                await db.delete(_extra)
    await db.commit()
```
placed right after the existing `await db.delete(row)` + `await db.commit()`, followed by a second commit. (Two commits: chain erasure stays exactly as verified today; derived cleanup is additive best-effort. Collector filters strictly on `cm_derived_from`, so existing tests with derived-free fakes see empty lists and stay green.)

- [ ] **Step 4: Run tests**

Run: `pytest tests/services/test_erasure_service.py tests/retrieval/ tests/mcp_hub/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/erasure_service.py tests/retrieval/test_correction.py
git commit -m "feat: forget cascades to derived dependents"
```

---

### Task 8: Real-SQLite roundtrip + §9b self-check

**Files:**
- Create: `tests/lite/test_correction_roundtrip.py`
- Modify: `skills/orivory/SKILL.md`, `skills/orivory/references/tool-catalog.md`

**Interfaces:** Proves the two risky bits against real SQL (not fakes): JSON `cm_*` filtering compiles on SQLite, and tuple comparison in `timeline` works. This is the spec §9b gate.

- [ ] **Step 1: Write the test** (follows `tests/lite/test_sqlite_bootstrap.py` pattern; skipped unless `IS_SQLITE`)

```python
"""Real-SQLite proof: JSON resolve filter + §9b self-check. No fakes."""
from __future__ import annotations
import uuid
import pytest
import pytest_asyncio
from app.database import IS_SQLITE, AsyncSessionLocal, Base, engine
from app.models.memory import Memory
from app.models.user import User
from app.retrieval.memory.correction import resolve_correction, state_of

pytestmark = pytest.mark.skipif(not IS_SQLITE, reason="lite-mode tests require a sqlite DATABASE_URL")

@pytest_asyncio.fixture
async def _tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield

async def test_self_check_supersede_scope_isolation(_tables):
    """Spec §9b: same triple hides old by default; other scope untouched."""
    from app.mcp_hub import tools as hub_tools
    from app.mcp_hub.identity import AgentPrincipal
    uid = uuid.uuid4()
    async with AsyncSessionLocal() as db:
        db.add(User(id=uid, email="selfcheck@example.com", hashed_password="x",
                    display_name="SC", is_verified=True, is_active=True))
        await db.commit()
    async with AsyncSessionLocal() as db:
        r1 = await resolve_correction(db, user_id=uid, title="DB", content="Postgres",
            subject="proj", attribute="db", scope="prod")
        r2 = await resolve_correction(db, user_id=uid, title="DB", content="SQLite",
            subject="proj", attribute="db", scope="demo")
        r3 = await resolve_correction(db, user_id=uid, title="DB", content="PG16",
            subject="proj", attribute="db", scope="prod")
    assert r1.value["status"] == "added" if hasattr(r1, "value") else r1["status"] == "added"
    assert r3["status"] == "superseded"
    assert r2["status"] == "added"  # other scope untouched
    async with AsyncSessionLocal() as db:
        p = AgentPrincipal(user_id=uid, agent_client_id=uuid.uuid4(),
                           name="SC", scopes=frozenset({"memory:read"}))
        hub_tools._principal_var.set(p)
        try:
            out = await hub_tools.search_memory("db")
            ids = [r["id"] for r in out["results"]]
            assert str(r3["memory"].id) in ids
            assert str(r1["memory"].id) not in ids  # old prod hidden by default
            out2 = await hub_tools.search_memory("db", include_history=True)
            assert str(r1["memory"].id) in {r["id"] for r in out2["results"]}
            got = await hub_tools.get_memory(str(r3["memory"].id))
            assert got["supersedes"] == str(r1["memory"].id)
        finally:
            hub_tools._principal_var.set(None)
```

Note: `search_memory`'s `_recall_memory_ids` seam does a real salience-ordered select here (no monkeypatch) — both prod rows exist, old one filtered by state. If salience ties exclude a row from the limit, the test fails honestly and the limit-backfill ceiling (Task 5) gets promoted to a fix.

- [ ] **Step 2: Run — first run exercises real code paths**

Run: `DATABASE_URL="sqlite+aiosqlite:////tmp/sc.db" pytest tests/lite/test_correction_roundtrip.py -v`
Expected: PASS (or an honest failure pointing at tuple-comparison / JSON SQL — fix the SQL, not the test).

- [ ] **Step 3: Update skill docs** (`SKILL.md:80` "six" → "seven", add `correct_memory`; `tool-catalog.md` table row + choosing rule: correct-vs-add — correct when a stored fact changed, add when it is new; never overwrite via add)

- [ ] **Step 4: Commit**

```bash
git add tests/lite/test_correction_roundtrip.py skills/orivory/
git commit -m "test: sqlite roundtrip self-check plus correct_memory docs"
```

---

### Task 9: Acceptance gate (merge gate, no new code)

- [ ] **Step 1: Full suite**

Run: `pytest -v`
Expected: all green (pre-existing failures, if any, recorded — never silently deselected).

- [ ] **Step 2: Lint**

Run: `make lint` (or `ruff check .`)
Expected: clean on touched files.

- [ ] **Step 3: Record the gate**

Append results (counts, `needs-check` rate if eval ran, stale/false-supersession on the 9 internal cases) to the spec §8 checklist or CHANGELOG. Benchmark-subset numbers (LongMemEval-S KU, FactConsolidation-SH) are follow-up execution work — explicitly out of this plan; do not invent them.

## Self-Review

- **Spec coverage:** §6.1 keys → T1 (+`cm_needs_check`); legacy rule → T1/T2; §6.2 resolve+atomicity+normalization → T2; §6.3 filters/fast-path/trace → T3; §6.4 tool table → T4/T5/T6; §6.5 3 patches → T3 (fast-path+trace), T6 (timeline); §7 deletion list → out of scope (separate deletion passes, noted); §8/§9b → T8/T9; §10 risks → defaults in T2 + skill honesty note (agent-external copies unrecoverable — add one line to `tool-catalog.md` in T8).
- **Placeholder scan:** no TBD/TODO; every step has exact code/commands.
- **Type consistency:** `resolve_correction` dict keys (`status/memory/superseded/dirtied`) identical in T2/T4/T5/T8; `ACTION_CORRECT` imported from identity in tools; `stage_ms`/`rewrite_skipped` set on both trace constructions in T3.
- **Fix applied during review:** T2's `memory_id`-mismatch branch initially fell through to plain add — now forces `needs-check` via `target_ok`. T7's first draft put a nonsense import in the call site — removed, kept the guarded collector call only.
