"""P2 acceptance gate — the plan's §9 P2 row, end to end, over REAL stores.

Everything here runs against the P1b gate's real install (a private SQLite file
+ a private embedded-Qdrant folder, its fixtures registered as a plugin and
reused by name): the SQL layer, the FTS5 ladder, the vector store and the recall
pipeline are exercised whole. Nothing about the store, the index or the ladder is
mocked. The only substitutions are the seams that are out of process in
production:

* the embedder — deterministic unit vectors, because no claim in this file is
  about embedding QUALITY (the T7 ablation artifact and the P0 parity gate own
  that). The embedding CONTRACT (dim 384, the cutover generation name) is the
  real one. The heartbeat test is the exception: it restores the REAL arctic
  session, because there the embed cost is the claim;
* the LLM query rewrite (an out-of-process call, never the claim);
* the reranker TRANSPORT — a recording stub where the claim is what the
  reranker is HANDED, or what happens when it fails;
* the vector outage, which is REAL: the client is pointed at a port nothing
  listens on (the P3 gate's ``_unreachable_store``), so the store really refuses
  the connection.

The §9 P2 row, bullet by bullet (one test each, marked with the bullet text):

* ``top_k`` 1/5/10/15 with a sufficient / stale-foreign-heavy / empty candidate
  pool — the returned count is the count invariant from T2;
* rerank zero / timeout / invalid response / refill;
* no pre-ACL outbound text — the reranker receives only current SQL text, pinned
  by recording what it was handed while the point's payload text is stale;
* heartbeat/event-loop responsiveness under concurrent ingest + recall, with
  T1's own ticker; both the lag bound and the answers are asserted;
* FTS insert/update/rollback/delete/rebuild parity, plus the ladder's published
  rollback drill (docs/ROLLBACK_P1B.md §6) run on a copy and rolled forward;
* duplicate text with different UUIDs is never merged;
* exact-ID / Vietnamese (with and without diacritics) / English slices;
* vector unavailable → the explicit lexical fallback (and the typed 503 where
  the deployment has no lexical index);
* MCP and API share auth + the semantic fixture, and the MCP payload stays
  index-only;
* the ablation + late/full hydration parity verdicts are present in the artifact;
* the CI wiring itself is read (YAML), so narrowing it fails here.

Three pins the plan attached to this task are code rather than behaviour, and
live here too: the lifespan warms the embedder BEFORE the boot drain, the
example env files carry the shipped P2 retrieval defaults (CI copies
``.env.test.example`` to ``.env``, so a stale example is a stale deployment),
and the trace may only carry DECLARED keys (the P3 gate's leaked-key pin,
re-asserted on every trace this file builds — the API's serialized body
included).

Two further pins the final fix wave attached: the erasure path leaves no FTS5
row behind (through the real ``erase_memories``), and the signed §12.2 RSS
budget is MEASURED here in a FRESH SUBPROCESS that boots the real stack alone
(R30): an in-process peak is contaminated by whatever the suite loaded before
it, which is exactly what made the first version of this pin load-dependent.
"""
from __future__ import annotations

import ast
import asyncio
import json
import math
import os
import sqlite3
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from dotenv import dotenv_values
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event, select, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app import database
from app.config import Settings, settings
from app.main import app
from app.mcp_hub import tools as hub_tools
from app.mcp_hub.identity import ACTION_SEARCH, AgentPrincipal
from app.models.index_outbox import IndexOutbox
from app.models.memory import Memory
from app.models.memory_access_log import MemoryAccessLog
from app.observability.fallbacks import fallback_counts, reset_fallback_counts
from app.retrieval import e5_local
from app.retrieval import reranker as reranker_module
from app.retrieval.embedder import embed_query as real_embed_query
from app.retrieval.embedder import embed_texts as real_embed_texts
from app.retrieval.embedder import embed_texts_sync as real_embed_texts_sync
from app.retrieval.embedder import warmup_embedder
from app.retrieval.embedding_fingerprint import generation_name
from app.retrieval.memory import lexical_index, vector_store
from app.retrieval.memory import retriever as retriever_module
from app.retrieval.memory.retriever import MemoryRetriever
from app.retrieval.reranker import RerankInvalidResponse, RerankUnavailable
from app.retrieval.vector_retriever import VectorUnavailableError
from app.schemas.Orivory import RECALL_TRACE_COUNTER_KEYS, RECALL_TRACE_STAGE_KEYS
from app.utils.dependencies import enforce_llm_quota, get_current_user

# The P1b gate's real-store fixtures (``env`` / ``world`` / ``live``), reused
# rather than copied — this gate proves the SAME store the migration installs.
from tests.retrieval.test_event_loop_responsiveness import MAX_LAG_MS, P99_LAG_MS, LoopTicker
from tests.retrieval.test_event_loop_responsiveness import _ingest as t1_ingest
from tests.retrieval.test_event_loop_responsiveness import _seed_pending as t1_seed_pending
from tests.retrieval.test_event_loop_responsiveness import _text as t1_text
from tests.retrieval.test_p1b_gate import _memory, _payloads, _user, _vector_for

pytest_plugins = ["tests.retrieval.test_p1b_gate"]

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_YML = REPO_ROOT / ".github" / "workflows" / "ci.yml"
ARTIFACT_PATH = REPO_ROOT / "eval" / "ablation_retrieval_p2.json"
MAIN_PY = REPO_ROOT / "app" / "main.py"
ENV_EXAMPLES = (REPO_ROOT / ".env.example", REPO_ROOT / ".env.test.example")

CI_STEP_NAME = "Run P2 retrieval/hybrid suites (temp SQLite, no services)"
CI_ABLATION_STEP_NAME = "Run P2 retrieval ablation + its contract test"
GATE_MODULE = "tests/retrieval/test_p2_gate.py"

# The P2 knobs the plan added (R13(p2) signed the cap, not the window). CI
# copies ``.env.test.example`` to ``.env`` for the compose jobs, so what these
# files say IS what CI and an example-seeded deployment run. The pin below
# PARSES the files and compares a mapping against the live settings defaults:
# a commented-out line, a wrong value and a missing key all fail — a substring
# match passes on the first two.
ENV_EXAMPLE_KNOBS = (
    "RERANK_TOP_N",
    "RETRIEVAL_RERANK_POOL_MULTIPLIER",
    "RETRIEVAL_HYBRID_ENABLED",
    "RETRIEVAL_RRF_K",
    "EMBED_EXECUTOR_WORKERS",
    "EMBED_ORT_INTRA_OP_THREADS",
    "EMBED_WARMUP_ON_BOOT",
)


def _example_value(value) -> str:
    """How a dotenv file spells a settings default (bools are lowercase)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _shipped_default(knob: str):
    """The value the code SHIPS — not the ambient one a local ``.env`` can set.

    These assertions state the release contract ("the flag ships OFF", "the cap
    is 20"). Reading the live ``settings`` made a developer's ``.env`` turn them
    red while CI (no ``.env``) stayed green — a false red that reads exactly
    like a real regression.
    """
    return Settings.model_fields[knob].default


# The group marker: rows whose content carries it embed to ONE shared vector, so
# a test can put a group ahead of another in the store's own cosine order
# without depending on the fake embedder's luck.
BASE_VECTOR_MARKER = "P2GATE-GROUP"


# ── the harness: real rows, real points, controlled out-of-process seams ────


def _shaped_embedder(monkeypatch, base: list[float]) -> None:
    """Row vectors by group: marked rows embed to ``base``, the rest normally."""
    def _vector_for_text(text_value: str) -> list[float]:
        return base if BASE_VECTOR_MARKER in text_value else _vector_for(text_value)

    async def _embed(texts: list[str]) -> list[list[float]]:
        return [_vector_for_text(value) for value in texts]

    monkeypatch.setattr(vector_store, "embed_texts", _embed)
    monkeypatch.setattr(vector_store, "embed_texts_sync",
                        lambda texts: [_vector_for_text(value) for value in texts])


async def _seed(env, owner, contents: list[str], *, superseded: bool = False) -> list[Memory]:
    """Rows + their points, straight into the REAL store (the operator's own seed).

    No API path: the write-through is not the claim here, the store state is.
    """
    rows: list[Memory] = []
    async with env.sessions() as db:
        for content in contents:
            row = _memory(
                owner,
                content,
                extra_metadata={"cm_superseded_by": "p2gate"} if superseded else {},
            )
            db.add(row)
            rows.append(row)
        await db.commit()
    for row in rows:
        await vector_store.upsert_memory(row)
    return rows


async def _new_user(env, email: str) -> uuid.UUID:
    """A fresh tenant, so a shape can own a pool without another shape's rows."""
    async with env.sessions() as db:
        user = _user(db, email)
        db.add(user)
        await db.commit()
        return user.id


def _seams(monkeypatch) -> dict:
    """The out-of-process seams: the query embedding and the LLM rewrite.

    Returns a holder the test writes the next query vector into — the same
    vector goes to every caller, which is what makes the API and the MCP paths
    answer off one fixture.
    """
    holder: dict = {"vector": None}

    async def _embed_query(_query: str) -> list[float]:
        return holder["vector"]

    async def _rewrite(query, context=None, **_kwargs):
        return {"rewritten_query": query, "entities": [], "reasoning": None,
                "_fallback_used": False}

    monkeypatch.setattr(retriever_module, "embed_query", _embed_query)
    monkeypatch.setattr(retriever_module, "rewrite_query", _rewrite)
    return holder


def _counting_store(monkeypatch) -> list[int]:
    """The REAL store call, counted: the fetch sequence is part of the contract."""
    calls: list[int] = []
    real_search = retriever_module.search_memories

    async def _search(*args, **kwargs):
        calls.append(int(kwargs["top_k"]))
        return await real_search(*args, **kwargs)

    monkeypatch.setattr(retriever_module, "search_memories", _search)
    return calls


async def _recall(env, owner, query: str, *, top_k: int = 10,
                  semantic_rerank: bool = False, hybrid: bool | None = None):
    async with env.sessions() as db:
        retriever = MemoryRetriever(
            db, owner, semantic_rerank=semantic_rerank, hybrid=hybrid
        )
        return await retriever.recall(query, top_k=top_k, include_personal_context=False)


def _returned(response) -> list[str]:
    return [str(result.id) for result in response.results]


def _assert_trace_is_declared(response) -> None:
    """The two declarations are contracts in BOTH directions (the P3 gate's pin).

    This run may only write ``stage_ms`` keys ``RECALL_TRACE_STAGE_KEYS``
    declares and ``counts`` keys ``RECALL_TRACE_COUNTER_KEYS`` declares, so an
    undeclared extra cannot leak into a debug payload. Takes the model object
    or the API's serialized body, and is called on EVERY trace this file
    builds.
    """
    trace = response["trace"] if isinstance(response, dict) else response.trace
    stage_ms = trace["stage_ms"] if isinstance(trace, dict) else trace.stage_ms
    counts = trace["counts"] if isinstance(trace, dict) else trace.counts
    assert set(stage_ms) <= set(RECALL_TRACE_STAGE_KEYS)
    assert set(counts) <= set(RECALL_TRACE_COUNTER_KEYS)


def _coverage(env) -> dict[str, int]:
    """Canonical-vs-index coverage, read off the real file's sync engine."""
    with env.sync_engine.connect() as conn:
        return lexical_index.coverage(conn)


def _lexical_hits(env, query: str, user_id, *, limit: int = 10) -> list[dict]:
    with env.sync_engine.connect() as conn:
        return lexical_index.search(conn, query, user_id=user_id, limit=limit)


# ══ §9: top_k 1/5/10/15 with sufficient / stale-foreign-heavy / empty pools ══

GRID_TOP_K = (1, 5, 10, 15)
GRID_ELIGIBLE = 16


@pytest.mark.parametrize("top_k", GRID_TOP_K)
async def test_top_k_over_sufficient_stale_foreign_and_empty_pools(live, monkeypatch, top_k):
    """The returned count is ``min(top_k, eligible the bounded fetch offered)``.

    Three tenants in one real store, one query vector each:

    * **sufficient** — 16 eligible rows, nothing filterable ahead of them;
    * **stale/foreign-heavy** — ``3 x top_k`` superseded rows that outrank the
      eligible ones, plus a foreign tenant with the SAME vectors and text, so a
      tenant clause that leaked would answer with Bob's rows;
    * **empty** — a tenant with no points at all.

    The fetch sequence is the contract's own: the R4 pool, then at most ONE
    refill at the ``top_k * 4`` cap — never a second round, and never a short
    answer because the filter or the reranker ran.
    """
    base = _vector_for("P2 GATE grid query vector")
    _shaped_embedder(monkeypatch, base)
    holder = _seams(monkeypatch)
    holder["vector"] = base
    calls = _counting_store(monkeypatch)

    enough = await _new_user(live, "grid-sufficient@gate.invalid")
    empty = await _new_user(live, "grid-empty@gate.invalid")

    sufficient_rows = await _seed(
        live, enough, [f"sufficient row {index}" for index in range(GRID_ELIGIBLE)]
    )
    stale = await _seed(
        live, live.alice.id,
        [f"{BASE_VECTOR_MARKER} stale {index}" for index in range(3 * top_k)],
        superseded=True,
    )
    eligible = await _seed(
        live, live.alice.id, [f"eligible row {index}" for index in range(GRID_ELIGIBLE)]
    )
    foreign = await _seed(
        live, live.bob.id,
        [f"{BASE_VECTOR_MARKER} stale {index}" for index in range(3 * top_k)],
    )

    pool = max(top_k, math.ceil(top_k * settings.RETRIEVAL_RERANK_POOL_MULTIPLIER))

    # (a) sufficient: a pool full of eligible rows serves exactly top_k.
    calls.clear()
    response = await _recall(live, enough, "sufficient", top_k=top_k)
    expected = min(top_k, GRID_ELIGIBLE)
    entered = min(pool, GRID_ELIGIBLE)  # what page one put into scoring
    assert len(response.results) == expected, (top_k, _returned(response))
    assert len(response.results) == min(top_k, entered)  # the count invariant
    assert set(_returned(response)) <= {str(row.id) for row in sufficient_rows}
    assert calls == [pool], "a short page is the store's whole answer, no refill"
    assert response.trace.counts["dense"] == entered  # page one was the whole store
    assert response.trace.counts["eligible"] == entered
    assert response.trace.counts["returned"] == expected
    assert response.trace.stage_ms.get("refill", None) is None, "it never ran"
    _assert_trace_is_declared(response)

    # (b) stale/foreign-heavy: page one is ALL stale; the one bounded refill is
    # what finds the eligible rows behind them, and the count still holds.
    calls.clear()
    response = await _recall(live, live.alice.id, "stale-heavy", top_k=top_k)
    served = _returned(response)
    alice_eligible = {str(row.id) for row in eligible} | {str(live.alice_current.id)}
    assert len(served) == min(top_k, len(alice_eligible)), (top_k, served)
    assert set(served) <= alice_eligible, "a stale or foreign row was served"
    assert set(served).isdisjoint({str(row.id) for row in stale})
    assert set(served).isdisjoint({str(row.id) for row in foreign}), "tenant leak"
    assert calls == [pool, top_k * 4], "the R4 pool, then ONE refill at the cap"
    counts = response.trace.counts
    assert counts["dense"] == pool, "the first page filled the pool: all stale"
    assert counts["refill"] == counts["eligible"], "the refill is what ADDED the pool"
    assert counts["eligible"] >= len(served)
    assert counts["returned"] == len(served)
    assert response.trace.stage_ms["refill"] > 0.0
    _assert_trace_is_declared(response)

    # (c) empty: no points for this tenant — a measured zero, never a fabricated
    # one, and never an answer borrowed from another tenant's points.
    calls.clear()
    response = await _recall(live, empty, "empty pool", top_k=top_k)
    assert _returned(response) == []
    assert response.trace.counts["dense"] == 0
    assert response.trace.counts["returned"] == 0
    assert calls == [pool]
    _assert_trace_is_declared(response)

    # The pin is not vacuous: the stale rows really did own page one (they are
    # exactly what the refill had to look behind).
    assert not ({str(row.id) for row in stale} & set(served))


# ══ §9: rerank zero / timeout / invalid response / refill ═══════════════════

RERANK_ROWS = 6


async def test_rerank_zero_timeout_invalid_and_refill(live, monkeypatch):
    """The reranker can answer, fail or lie — none of it moves the count.

    Over the real store, four shapes: a ``0.0`` relevance (data, never
    absence), a transport timeout, a schema-invalid body, and the bounded
    refill that the filtering makes necessary.
    """
    holder = _seams(monkeypatch)
    calls = _counting_store(monkeypatch)
    rows = await _seed(live, live.alice.id, [f"rerank row {index}"
                                             for index in range(RERANK_ROWS)])
    first_dense = str(rows[3].id)
    holder["vector"] = _vector_for("rerank row 3")  # the dense head, by construction
    reset_fallback_counts()

    # (a) zero: a 0.0 relevance is read by KEY, not truthiness — the reason
    # string carries the score the reranker really returned.
    handed: list[int] = []

    async def _zero(_query, chunks, *, top_n=None):
        handed.append(len(chunks))
        return [dict(chunk, rerank_score=0.0) for chunk in chunks]

    monkeypatch.setattr(reranker_module, "rerank", _zero)
    response = await _recall(live, live.alice.id, "rerank", top_k=5, semantic_rerank=True)
    assert handed == [7]  # the whole pool: 6 seeded rows + the fixture's own cutover row
    assert len(response.results) == 5
    assert _returned(response)[0] == first_dense
    assert response.results[0].match_reasons[0] == "rerank:0.00"
    assert response.trace.counts["reranked"] == handed[0], "the whole head merged"
    assert fallback_counts().get("retrieval.rerank_failed", 0) == 0
    _assert_trace_is_declared(response)

    # (b) timeout: the typed transport failure degrades to dense order, the
    # registered counter fires, and the count is unchanged.
    async def _timeout(_query, chunks, *, top_n=None):
        raise RerankUnavailable("the store answered after the client gave up")

    monkeypatch.setattr(reranker_module, "rerank", _timeout)
    response = await _recall(live, live.alice.id, "rerank", top_k=5, semantic_rerank=True)
    assert len(response.results) == 5
    assert _returned(response)[0] == first_dense  # dense order kept
    assert fallback_counts()["retrieval.rerank_failed"] == 1
    assert response.trace.stage_ms["rerank"] > 0.0  # recorded in the finally
    assert "reranked" not in response.trace.counts  # no reranked head was served
    _assert_trace_is_declared(response)

    # (c) invalid response: the same contract, the same counter.
    async def _invalid(_query, chunks, *, top_n=None):
        raise RerankInvalidResponse("body was not the reranker's schema")

    monkeypatch.setattr(reranker_module, "rerank", _invalid)
    response = await _recall(live, live.alice.id, "rerank", top_k=5, semantic_rerank=True)
    assert len(response.results) == 5
    assert fallback_counts()["retrieval.rerank_failed"] == 2
    _assert_trace_is_declared(response)

    # (d) refill: a full first page that the filter empties earns exactly ONE
    # extra fetch, and the rows it adds are the answer.
    base = _vector_for("P2 GATE refill vector")
    _shaped_embedder(monkeypatch, base)
    refill_user = await _new_user(live, "refill@gate.invalid")
    behind = await _seed(live, refill_user, [f"behind row {index}" for index in range(3)])
    await _seed(
        live, refill_user,
        [f"{BASE_VECTOR_MARKER} stale {index}" for index in range(20)],
        superseded=True,
    )
    holder["vector"] = base
    calls.clear()
    response = await _recall(live, refill_user, "refill", top_k=10)
    assert calls == [20, 40], "one bounded refill at the top_k * 4 hard cap"
    assert len(response.results) == 3 == min(10, 3)
    assert set(_returned(response)) == {str(row.id) for row in behind}
    assert response.trace.counts["refill"] == 3
    assert response.trace.stage_ms["refill"] > 0.0
    _assert_trace_is_declared(response)


# ══ §9: no pre-ACL outbound text ════════════════════════════════════════════


async def test_no_pre_acl_text_reaches_the_rerank_transport(live, monkeypatch):
    """The point's payload is STALE and the reranker never sees it.

    The point is written with one text, the SQL row moves on, and the payload
    keeps the old copy — exactly the state the read path must never forward.
    The recording transport asserts every chunk it was handed is the CURRENT
    SQL-owned text, byte for byte.
    """
    holder = _seams(monkeypatch)
    (row,) = await _seed(live, live.alice.id, ["pre-acl stale text alpha"])
    holder["vector"] = _vector_for("pre-acl stale text alpha")  # matches the POINT

    stale_payload = _payloads(generation_name("memory"))[str(row.id)]["content"]
    assert stale_payload == "pre-acl stale text alpha"

    async with live.sessions() as db:  # the row moves on; the payload does not
        memory = await db.get(Memory, row.id)
        memory.content = "pre-acl CURRENT text zeta"
        await db.commit()

    seen: list[dict] = []

    async def _recording(_query, chunks, *, top_n=None):
        seen.extend(chunks)
        return [dict(chunk, rerank_score=0.5) for chunk in chunks]

    monkeypatch.setattr(reranker_module, "rerank", _recording)
    response = await _recall(live, live.alice.id, "pre-acl", top_k=5, semantic_rerank=True)

    assert seen, "the transport was handed nothing — the pin would be vacuous"
    async with live.sessions() as db:
        current = {
            str(memory.id): (f"Title: {memory.title}\n{memory.content}"
                             if memory.title else memory.content)
            for memory in (await db.execute(select(Memory))).scalars().all()
        }
    for chunk in seen:
        assert chunk["content"] == current[chunk["memory_id"]], (
            "a pre-authorization copy left the process"
        )
    assert str(row.id) in {chunk["memory_id"] for chunk in seen}
    assert "pre-acl stale text alpha" not in response.model_dump_json()
    _assert_trace_is_declared(response)


# ══ §9: heartbeat / event-loop responsiveness (ingest + recall) ═════════════

HEARTBEAT_DRAIN = 20
HEARTBEAT_INGEST = 20


@pytest.mark.skipif(
    not e5_local.arctic_files_cached(),
    reason="arctic onnx cache missing — run local, do not download in CI",
)
async def test_the_loop_beats_while_ingest_and_recall_run_concurrently(live, monkeypatch):
    """T1's ticker, the real embedder, the real stores — and correct answers.

    The barrier drains the recall path's own backlog while a live ingest runs
    beside it. The signed contract (R6) is ``max < 30 ms`` and ``p99 < 15 ms``,
    with the model warmed BEFORE the ticker starts; the same run must also land
    every ingest and serve rows of the backlog it just drained.
    """
    # Test-local budget, exactly like T1's concurrent test: both legs run through
    # the 2-worker embed executor, so the drain queues behind the ingest's
    # embeds. The signed 2.0 s RYW budget is a recall-ALONE measurement (the P3
    # gate's shape) and is untouched.
    monkeypatch.setattr(settings, "RECALL_FRESHNESS_BUDGET_SECONDS", 10.0)
    monkeypatch.setattr(vector_store, "embed_texts", real_embed_texts)
    monkeypatch.setattr(vector_store, "embed_texts_sync", real_embed_texts_sync)
    monkeypatch.setattr(retriever_module, "embed_query", real_embed_query)

    async def _identity(query, context=None, **_kwargs):
        return {"rewritten_query": query, "entities": [], "reasoning": None,
                "_fallback_used": False}

    monkeypatch.setattr(retriever_module, "rewrite_query", _identity)
    await warmup_embedder()  # the session build is boot's cost, not this measurement's

    shim = SimpleNamespace(sessions=live.sessions, owner=live.alice.id)
    seeded = await t1_seed_pending(
        shim, [t1_text("backlog", index) for index in range(HEARTBEAT_DRAIN)]
    )
    async with LoopTicker() as ticker:
        response, landed = await asyncio.gather(
            _recall(live, live.alice.id, t1_text("heartbeat", 0)),
            t1_ingest(shim, [t1_text("live", index) for index in range(HEARTBEAT_INGEST)]),
        )
    _assert_trace_is_declared(response)

    print(ticker.summary())
    assert landed == HEARTBEAT_INGEST  # the write-through really ran
    served = set(_returned(response))
    assert served, "the recall served nothing while the ingest ran"
    # …and the answers are the write the barrier just drained: at least one
    # backlog row is IN the answer of a twenty-document drain.
    assert served & {str(memory_id) for memory_id in seeded}, "the barrier's rows are missing"

    async with live.sessions() as db:
        intents = (await db.execute(select(IndexOutbox))).scalars().all()
    assert {row.status for row in intents} == {"done"}, "work stayed owed after the run"

    assert len(ticker.lags) >= 50, "too few heartbeats to judge"
    assert max(ticker.lags) < MAX_LAG_MS, ticker.summary()
    ordered = sorted(ticker.lags)
    p99 = ordered[max(math.ceil(0.99 * len(ordered)) - 1, 0)]
    assert p99 < P99_LAG_MS, ticker.summary()


# ══ §12.2: the signed RSS budget (≤ 1 GB peak), measured in a fresh child ═══

RSS_LIMIT_BYTES = 1024 ** 3
RSS_PROBE_TIMEOUT_SECONDS = 120

# R30: the budget is measured in a FRESH interpreter that boots the real stack
# alone — the same settings, an embedded Qdrant on a temp folder, the real
# arctic embedder (warm cache only; this never downloads) and one real
# upsert/search. An in-process ``ru_maxrss`` was contaminated by everything the
# suite had already loaded before this test, so the pin failed on load order
# and not on the service. Stdlib only: ``subprocess`` + ``sys.executable`` +
# ``resource``. The child exits non-zero and says so on any internal error, so
# a broken probe FAILS here instead of passing an unmeasured budget.
RSS_PROBE_CHILD = '''
import asyncio
import os
import resource
import sys
import traceback
import uuid


def peak_rss_bytes() -> int:
    """``ru_maxrss`` is BYTES on macOS (the dev box) and KiB on Linux (CI)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


async def main() -> None:
    for name in ("DATABASE_URL", "QDRANT_LOCAL_PATH", "QDRANT_MODE"):
        if not os.environ.get(name):
            raise RuntimeError(f"{name} must be set by the parent test")

    import app.main  # noqa: F401 — the app's whole import surface
    from app import database
    from app.models.memory import Memory
    from app.models.user import User
    from app.retrieval import e5_local, vector_backend
    from app.retrieval.embedder import embed_query, warmup_embedder
    from app.retrieval.memory import vector_store

    database.engine.echo = False  # a dev ambient must not drown the stdout

    if not e5_local.arctic_files_cached():
        raise RuntimeError("arctic onnx cache cold — the probe never downloads")

    await warmup_embedder()  # the boot warmup: the ONNX session, the heavy alloc
    await database.bootstrap_sqlite()  # the real ladder, on the fresh temp file

    async with database.AsyncSessionLocal() as db:
        user = User(id=uuid.uuid4(), email="rss-probe@gate.invalid", hashed_password="x",
                    display_name="RSS probe", is_verified=True, is_active=True)
        db.add(user)
        await db.flush()
        row = Memory(id=uuid.uuid4(), user_id=user.id, content="rss probe token kappa", tags=[])
        db.add(row)
        await db.commit()

    await vector_store.upsert_memory(row)  # real embed -> embedded Qdrant collection
    hits = await vector_store.search_memories(
        await embed_query("rss probe token kappa"), user_id=str(user.id), top_k=5
    )
    if [hit["memory_id"] for hit in hits] != [str(row.id)]:
        raise RuntimeError(f"the store probe served nothing: {hits}")

    ballast_mib = int(os.environ.get("RSS_PROBE_BALLAST_MIB") or "0")
    if ballast_mib:  # R30 fallibility hook: a leaking stack must be caught here
        _ballast = "x" * (ballast_mib * 1024 * 1024)
        print(f"BALLAST_MIB={ballast_mib}")

    await vector_backend.close_clients()
    await database.engine.dispose()

    print(f"PEAK_RSS_BYTES={peak_rss_bytes()}")
    print(f"PEAK_RSS_MIB={peak_rss_bytes() / 1024 ** 2:.1f}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        traceback.print_exc()
        print("RSS-PROBE-FAILED", file=sys.stderr)
        raise SystemExit(1) from None
'''

RSS_PROBE_MARKER = "PEAK_RSS_BYTES="


@pytest.mark.skipif(
    not e5_local.arctic_files_cached(),
    reason="arctic onnx cache missing — run local, do not download in CI",
)
def test_the_signed_rss_budget_holds_on_the_real_stack(tmp_path):
    """The §12.2 RSS budget, asserted on a FRESH boot of the real stack (R30).

    What dominates: the ONNX embedding session (the real arctic model — the
    cache must be warm, this test never downloads ~90 MB) and the embedded
    Qdrant client, whose local mode keeps indexes resident. Both are live in
    the child: the real settings, a throwaway SQLite file, a throwaway Qdrant
    folder, the boot warmup and one real upsert/search. The child prints its
    own ``ru_maxrss`` peak and this test asserts the signed ≤ 1 GB on it — a
    value nothing the suite ran earlier can inflate.
    """
    script = tmp_path / "rss_probe_child.py"
    script.write_text(textwrap.dedent(RSS_PROBE_CHILD))
    qdrant_dir = tmp_path / "rss-qdrant"
    qdrant_dir.mkdir()
    environment = {
        **os.environ,
        "DATABASE_URL": f"sqlite+aiosqlite:///{tmp_path / 'rss-probe.db'}",
        "QDRANT_MODE": "local",
        "QDRANT_LOCAL_PATH": str(qdrant_dir),
        "USE_LOCAL_EMBEDDINGS": "true",
        "LOCAL_EMBED_MODEL": "arctic",
        "PYTHONPATH": str(REPO_ROOT),
    }
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True,
        env=environment, cwd=str(REPO_ROOT), timeout=RSS_PROBE_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, (
        f"the RSS probe subprocess failed (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    markers = [line for line in result.stdout.splitlines()
               if line.startswith(RSS_PROBE_MARKER)]
    assert markers, (
        "the probe printed no peak-RSS marker — a silent subprocess must never "
        f"pass as an unmeasured budget:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    raw = markers[-1].split("=", 1)[1].strip()
    assert raw.isdigit(), f"unparseable {RSS_PROBE_MARKER} marker: {raw!r}"
    peak = int(raw)
    print(f"peak RSS (fresh subprocess): {peak / 1024 ** 2:.0f} MiB "
          f"(signed budget {RSS_LIMIT_BYTES // 1024 ** 2} MiB)")
    assert peak <= RSS_LIMIT_BYTES, (
        f"peak RSS {peak / 1024 ** 2:.0f} MiB exceeds the signed 1 GB budget"
    )


# ══ §9: FTS insert/update/rollback/delete/rebuild parity + the rollback drill ═

ROLLBACK_RECIPE = """
DROP TRIGGER IF EXISTS memories_fts_ai;
DROP TRIGGER IF EXISTS memories_fts_au;
DROP TRIGGER IF EXISTS memories_fts_ad;
DROP TABLE IF EXISTS memory_fts;     -- drops the FTS5 shadow tables too
PRAGMA user_version = 3;             -- the P1b ladder's top
"""


async def test_fts_parity_across_writes_rebuild_and_the_ladder_rollback(live, tmp_path):
    """Parity is the index's whole contract: canonical and indexed never drift.

    Insert / update / delete through the real triggers, a transaction that
    ROLLS BACK (the index row dies with it), the explicit repair path when a
    drift is introduced, and the ladder's published rollback drill
    (``docs/ROLLBACK_P1B.md`` §6) run on a COPY: a pre-P2 ladder boots the file
    with its rows intact, and rolling forward re-runs the step and returns the
    index to parity without rebuilding the memories table.
    """
    before = _coverage(live)
    assert before["missing"] == 0 and before["orphan"] == 0
    assert before["indexed"] == before["canonical"]

    # insert
    (row,) = await _seed(live, live.alice.id, ["parity insert token bravo"])
    after_insert = _coverage(live)
    assert after_insert["canonical"] == before["canonical"] + 1
    assert after_insert["indexed"] == after_insert["canonical"]
    assert [hit["memory_id"] for hit in _lexical_hits(live, "bravo", live.alice.id)] == [
        str(row.id)]

    # update: the old text leaves the index, the new one lands
    async with live.sessions() as db:
        memory = await db.get(Memory, row.id)
        memory.content = "parity update token charlie"
        await db.commit()
    assert _lexical_hits(live, "bravo", live.alice.id) == []
    assert [hit["memory_id"] for hit in _lexical_hits(live, "charlie", live.alice.id)] == [
        str(row.id)]
    assert _coverage(live)["indexed"] == _coverage(live)["canonical"]  # replaced, not added

    # rollback: the index row is written IN the writer's transaction and dies
    # with it — visible to the writer's own connection, gone after the rollback,
    # and never a drift for the repair path to find.
    async with live.sessions() as db:
        ghost = _memory(live.alice.id, "parity rollback token delta")
        db.add(ghost)
        await db.flush()  # NOT committed

        def _probe(session):
            conn = session.connection()
            return [hit["memory_id"] for hit in
                    lexical_index.search(conn, "delta", user_id=live.alice.id, limit=5)]

        assert await db.run_sync(_probe) == [str(ghost.id)], "not in the writer's txn"
        await db.rollback()
    assert _lexical_hits(live, "delta", live.alice.id) == []
    assert _coverage(live)["indexed"] == _coverage(live)["canonical"]

    # delete: the row and its index entry go together
    async with live.sessions() as db:
        memory = await db.get(Memory, row.id)
        await db.delete(memory)
        await db.commit()
    assert _lexical_hits(live, "charlie", live.alice.id) == []
    assert _coverage(live)["indexed"] == _coverage(live)["canonical"]

    # rebuild: the repair path. Introduce a REAL drift, watch coverage name it,
    # repair, and check the text is searchable again.
    with live.sync_engine.begin() as conn:
        conn.exec_driver_sql("DELETE FROM memory_fts WHERE 1=1")
    drifted = _coverage(live)
    assert drifted["missing"] == drifted["canonical"] > 0 and drifted["indexed"] == 0
    with live.sync_engine.begin() as conn:
        report = lexical_index.rebuild(conn)
    assert report["rebuilt"] is True
    assert _coverage(live)["missing"] == 0 and _coverage(live)["orphan"] == 0
    assert [hit["memory_id"] for hit in _lexical_hits(live, "current", live.alice.id)] == [
        str(live.alice_current.id)], "the rebuilt index really serves again"

    # the ladder's published rollback drill, on a COPY of the real file
    copy_path = tmp_path / "rollback-drill.db"
    with sqlite3.connect(live.db_path) as source, sqlite3.connect(copy_path) as target:
        source.backup(target)
    with sqlite3.connect(copy_path) as conn:
        conn.executescript(ROLLBACK_RECIPE)
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
        tables = {name for (name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        triggers = {name for (name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger'").fetchall()}
        rows = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
    assert "memory_fts" not in tables
    assert not {name for name in triggers if name.startswith("memories_fts")}
    assert rows == _coverage(live)["canonical"], "a pre-P2 reader still has its rows"

    # …and rolling forward re-runs the step: the index comes back at parity.
    engine = create_async_engine(f"sqlite+aiosqlite:///{copy_path}", poolclass=NullPool)
    event.listen(engine.sync_engine, "connect", database._configure_sqlite_connection)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(database.upgrade_sqlite_schema)
        async with engine.connect() as conn:
            version = int((await conn.execute(
                text("PRAGMA user_version"))).scalar_one())
            rolled = await conn.run_sync(lexical_index.coverage)
        assert version == database.SQLITE_SCHEMA_VERSION
        assert rolled["missing"] == 0 and rolled["orphan"] == 0
        assert rolled["indexed"] == rolled["canonical"] == rows
        assert Path(f"{copy_path}.pre-p2.bak").is_file(), "the ladder's own milestone backup"
    finally:
        await engine.dispose()


# ══ erasure leaves nothing of the memory in the lexical index ═══════════════


async def test_erasure_leaves_no_fts_row_behind(live):
    """The erased memory is gone from the FTS5 index, not just from ``memories``.

    The erasure path deletes the rows in SQL, so the v4 delete trigger runs with
    it: the index must hold no row for the erased memory, the lexical leg must
    stop serving it, and ``coverage`` must see no orphan. ``erase_memories`` is
    the real service over the real store (vectors and all) — the row deletion
    is the claim, not the receipt's vector bookkeeping.
    """
    from app.services.erasure_service import erase_memories

    (row,) = await _seed(live, live.alice.id, ["erasure pin token omega"])
    assert [hit["memory_id"] for hit in _lexical_hits(live, "omega", live.alice.id)] == [
        str(row.id)], "the fixture's own memory is not searchable — pin would be vacuous"

    async with live.sessions() as db:
        receipt = await erase_memories(db, live.alice.id, [row.id], requested_by="rest_api")
    assert receipt.detail["targets"][0]["status"] == "deleted"

    assert _lexical_hits(live, "omega", live.alice.id) == [], "the lexical leg still serves it"
    # The raw FTS table, not the join: the join would hide a surviving row (the
    # canonical row is gone), while the orphan is exactly what coverage counts.
    with live.sync_engine.connect() as conn:
        lingering = conn.exec_driver_sql(
            "SELECT count(*) FROM memory_fts WHERE memory_fts MATCH 'omega'"
        ).scalar_one()
    assert lingering == 0, "the erased memory's FTS row survived the erasure"
    after = _coverage(live)
    assert after["orphan"] == 0 and after["missing"] == 0
    assert after["indexed"] == after["canonical"]


# ══ §9: duplicate text with different UUIDs is not merged ═══════════════════


async def test_duplicate_text_with_different_uuids_is_not_merged(live, monkeypatch):
    """Two memories, one text, two UUIDs: two rows, on both legs and in fusion.

    The fusion dedupes by canonical UUID (R11b) and the index has no content
    key at all — ``duplicate`` must come back twice, never once.
    """
    base = _vector_for("P2 GATE duplicate vector")
    _shaped_embedder(monkeypatch, base)
    holder = _seams(monkeypatch)
    holder["vector"] = base
    text_value = f"{BASE_VECTOR_MARKER} duplicate text with one shared wording"
    first, second = await _seed(live, live.alice.id, [text_value, text_value])
    assert first.id != second.id

    hits = _lexical_hits(live, "duplicate", live.alice.id)
    assert {hit["memory_id"] for hit in hits} == {str(first.id), str(second.id)}
    assert {hit["rank"] for hit in hits} == {0, 1}, "no rowid/content collapse"

    response = await _recall(live, live.alice.id, "duplicate", top_k=10, hybrid=True)
    served = _returned(response)
    assert served.count(str(first.id)) == 1 and served.count(str(second.id)) == 1
    assert {str(first.id), str(second.id)} <= set(served)
    assert response.trace.counts["lexical"] >= 2
    assert response.trace.counts["fused"] >= 2
    _assert_trace_is_declared(response)


# ══ §9: exact id / Vietnamese (with and without diacritics) / English slices ═


SLICES = (
    ("exact_id", "deploy note ORIVORY-4417 rotated the staging key", "ORIVORY-4417"),
    ("vi_diacritics", "hồ sơ dự án đã ký kết với đối tác", "hồ sơ dự án"),
    ("vi_no_diacritics", "báo cáo tài chính quý ba đã hoàn tất", "bao cao tai chinh quy ba"),
    ("en", "the weekly release checklist includes smoke tests", "weekly release checklist"),
)


async def test_exact_id_vi_and_en_slices_rank_their_gold(live, monkeypatch):
    """Every §9 slice, over the real FTS5 leg and the real store.

    Each slice's query carries a token set only its gold holds, so the lexical
    leg's answer is unambiguous. The two Vietnamese slices are the diacritic
    pair: one query is typed WITH diacritics, the other without them, and both
    must reach their (fully diacritic) gold — the index folds diacritics
    (``remove_diacritics 2``) instead of the user having to.
    """
    holder = _seams(monkeypatch)
    golds: dict[str, Memory] = {}
    for name, content, _query in SLICES:
        (row,) = await _seed(live, live.alice.id, [content])
        golds[name] = row

    for name, _content, query in SLICES:
        holder["vector"] = _vector_for(golds[name].content)  # dense leg: gold first
        response = await _recall(live, live.alice.id, query, top_k=5, hybrid=True)
        served = _returned(response)
        assert str(golds[name].id) in served, f"{name}: the gold row was not served"
        assert served[0] == str(golds[name].id), f"{name}: a row outranked the gold"
        assert response.trace.counts["lexical"] >= 1, f"{name}: the lexical leg never ran"
        _assert_trace_is_declared(response)


# ══ §9: vector unavailable → explicit fallback ══════════════════════════════


async def test_vector_outage_answers_from_the_lexical_leg_or_stays_typed(live, monkeypatch):
    """R19 over the real file: the FTS5 leg answers; without an index, the typed error.

    The outage is REAL (a refused connection — the P3 gate's
    ``_unreachable_store``), not a stub that behaves like one. On SQLite the
    answer comes from the lexical leg, counted; dropping the index (the
    Postgres shape) leaves the typed readiness error to the caller.
    """
    from tests.retrieval.test_p3_gate import _unreachable_store

    holder = _seams(monkeypatch)
    (row,) = await _seed(live, live.alice.id, ["outage drill token zxqv"])
    holder["vector"] = _vector_for("outage drill token zxqv")
    reset_fallback_counts()

    async with _unreachable_store(monkeypatch):
        # The outage is REAL: the store itself refuses the connection (nothing
        # is stubbed into "behaving like it is down"), and the recall answers
        # from the lexical leg instead of failing or serving a false empty.
        with pytest.raises(VectorUnavailableError):
            await vector_store.search_memories(
                holder["vector"], user_id=str(live.alice.id), top_k=5)
        response = await _recall(live, live.alice.id, "zxqv", top_k=5)
    assert str(row.id) in _returned(response)
    counts = response.trace.counts
    assert counts["lexical"] >= 1
    assert "dense" not in counts, "the dense leg never answered"
    assert "fused" not in counts, "one leg is not a fusion"
    assert response.trace.stage_ms["lexical"] > 0.0
    assert fallback_counts()["retrieval.vector_unavailable"] == 1
    _assert_trace_is_declared(response)

    # The other branch: with no lexical leg (the Postgres shape — realised here
    # by dropping the index) the typed outage stands, never a silent [].
    with live.sync_engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE memory_fts")
    async with _unreachable_store(monkeypatch):
        with pytest.raises(VectorUnavailableError):
            await _recall(live, live.alice.id, "zxqv", top_k=5)
    assert fallback_counts()["retrieval.vector_unavailable"] == 1  # no false count

    # The store is back: the same tenant answers normally again.
    response = await _recall(live, live.alice.id, "outage drill token zxqv", top_k=5)
    assert str(row.id) in _returned(response)
    _assert_trace_is_declared(response)


# ══ §9: MCP and API share auth + semantics; MCP payload stays index-only ═══

MCP_QUERY = "shared fixture query"


async def test_mcp_and_api_share_semantics_and_the_mcp_payload_is_index_only(live, monkeypatch):
    """One fixture, two surfaces: the SAME order, each with its own payload.

    The MCP tool ranks through the shared recall seam (R22), so for the same
    tenant and the same fixture the id order is identical. The MCP answer is an
    INDEX — no content field, a clipped snippet — and it never leaves the
    caller's tenant. The ledger row is written for every authorized call.
    """
    holder = _seams(monkeypatch)
    alice_rows = await _seed(live, live.alice.id, [f"shared fixture row {index}"
                                                   for index in range(5)])
    bob_rows = await _seed(live, live.bob.id, [f"shared fixture row {index}"
                                               for index in range(5)])
    holder["vector"] = _vector_for("shared fixture row 2")

    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
        id=live.alice.id)
    app.dependency_overrides[enforce_llm_quota] = lambda: None
    try:
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as client:
            api = await client.post(
                "/api/v1/memories/recall", json={"query": MCP_QUERY, "top_k": 10}
            )
    finally:
        app.dependency_overrides.clear()
    assert api.status_code == 200, api.text
    _assert_trace_is_declared(api.json())

    principal = AgentPrincipal(
        user_id=live.alice.id, agent_client_id=uuid.uuid4(), name="GateAgent",
        scopes=frozenset({"memory:read"}),
    )
    monkeypatch.setattr(hub_tools, "_current_principal", lambda: principal)
    monkeypatch.setattr(hub_tools, "_session", live.sessions)
    mcp = await hub_tools.search_memory(MCP_QUERY, limit=10)

    api_ids = [str(result["id"]) for result in api.json()["results"]]
    mcp_ids = [row["id"] for row in mcp["results"]]
    assert api_ids, "the fixture served nothing — the comparison is vacuous"
    assert api_ids == mcp_ids, "the two surfaces disagree on the ordering"
    alice_ids = {str(row.id) for row in alice_rows} | {str(live.alice_current.id)}
    assert set(mcp_ids) <= alice_ids

    for row in mcp["results"]:  # index-only: the progressive-disclosure shape
        assert set(row) == {"id", "title", "snippet", "tags", "salience",
                            "captured_at", "state"}
        assert "content" not in row

    async with live.sessions() as db:
        ledger = (await db.execute(
            select(MemoryAccessLog).where(
                MemoryAccessLog.user_id == live.alice.id,
                MemoryAccessLog.action == ACTION_SEARCH,
            )
        )).scalars().all()
    assert len(ledger) == 1
    assert ledger[0].detail["memory_ids"] == mcp_ids

    # Auth is the hub's own: no principal, no answer — and the foreign tenant's
    # rows never appear in Alice's answer.
    monkeypatch.setattr(hub_tools, "_current_principal", lambda: None)
    assert await hub_tools.search_memory(MCP_QUERY) == {"error": "agent identity required"}

    bob_principal = AgentPrincipal(
        user_id=live.bob.id, agent_client_id=uuid.uuid4(), name="GateAgent",
        scopes=frozenset({"memory:read"}),
    )
    monkeypatch.setattr(hub_tools, "_current_principal", lambda: bob_principal)
    bob = await hub_tools.search_memory(MCP_QUERY, limit=10)
    bob_ids = {str(row.id) for row in bob_rows} | {str(live.bob_current.id),
                                                   str(live.bob_extra.id)}
    assert {row["id"] for row in bob["results"]} <= bob_ids
    assert not (bob_ids & set(mcp_ids))


# ══ §9: ablation + hydration verdicts present in the artifact ══════════════

HYDRATION_VERDICTS = (
    "not implemented: no measured benefit",
    "implemented: late hydration (arm e parity measured)",
)


def test_the_ablation_artifact_carries_the_enable_and_hydration_verdicts():
    """The artifact is the decision record; the shipped flag still ships OFF.

    The gate reads the committed artifact and asserts the two §9 verdicts are
    present and coherent — the enable rule's verdict for the arm that decides,
    and the hydration verdict with the arm it was measured on. It also pins
    that the artifact did NOT flip the default: R2(p2) makes the flip a
    separate, documented decision.
    """
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert artifact["scales"]["kind"] == "fixture"
    assert artifact["limitations"], "an artifact without its limits is a production claim"

    verdicts = artifact["enable_rule"]["verdicts"]
    assert verdicts["hybrid_rrf"]["decides_enable"] is True
    assert verdicts["hybrid_rrf"]["verdict"] in ("PASS", "FAIL")
    assert artifact["enable_rule"]["signed_thresholds"] == {
        "max_slice_loss_recall@5": 0.02, "min_overall_gain_recall@5": 0.02}

    hydration = artifact["hydration"]
    assert hydration["verdict"] in HYDRATION_VERDICTS
    assert hydration["deciding_arm"] in artifact["arms"]
    assert hydration["parity_arm"]["status"] in ("not run", "required")

    # The shipped state, not the measured one: the flag is OFF and the cap is
    # the R13 default. Nothing in this artifact may have moved either.
    assert _shipped_default("RETRIEVAL_HYBRID_ENABLED") is False
    assert _shipped_default("RERANK_TOP_N") == 20


# ══ the pins the plan attached to this task (code, examples, CI) ═══════════


def test_the_lifespan_warms_the_embedder_before_the_boot_drain():
    """F7 carry: the cold session is boot's cost, and it is paid FIRST.

    ``_warm_embedder_at_boot`` must be awaited before
    ``_drain_index_outbox_at_boot`` inside ``lifespan`` — the boot drain embeds,
    and a cold ``InferenceSession`` (610-685 ms) would otherwise land inside it.
    """
    tree = ast.parse(MAIN_PY.read_text(encoding="utf-8"))
    lifespan = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan"
    )
    order: dict[str, int] = {}
    for node in ast.walk(lifespan):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"_warm_embedder_at_boot", "_drain_index_outbox_at_boot"}:
                order.setdefault(node.func.id, node.lineno)
    assert set(order) == {"_warm_embedder_at_boot", "_drain_index_outbox_at_boot"}
    assert order["_warm_embedder_at_boot"] < order["_drain_index_outbox_at_boot"]


@pytest.mark.parametrize("path", ENV_EXAMPLES, ids=lambda path: path.name)
def test_the_example_env_files_carry_the_p2_retrieval_defaults(path):
    """CI copies ``.env.test.example`` to ``.env``: a stale example is a stale run.

    The R13 cap (20, not the pre-R13 5) and every P2 knob the plan added must be
    in BOTH examples, UNCOMMENTED and equal to the shipped default — an
    example-seeded deployment silently running the pre-R13 cap loses the
    whole-window rerank, and a commented-out line is a missing one. Read from
    the model, not from the live settings: a developer's ``.env`` is not shipped
    and must not decide what the examples have to carry.
    """
    parsed = dotenv_values(path)
    shipped = {knob: _example_value(_shipped_default(knob)) for knob in ENV_EXAMPLE_KNOBS}
    from_file = {knob: parsed.get(knob) for knob in ENV_EXAMPLE_KNOBS}
    assert from_file == shipped, (
        f"{path.name} does not carry the shipped retrieval defaults"
    )


def test_ci_runs_the_p2_gate_suites_and_the_ablation():
    """The workflow is parsed, so narrowing CI fails here instead of silently.

    The P2 step's suite list is EXPLICIT (a new file never runs otherwise), and
    the ablation script + its contract test — which appeared in no step before
    this task — have a step of their own.
    """
    jobs = yaml.safe_load(CI_YML.read_text())["jobs"]
    steps = [step for job in jobs.values() for step in job.get("steps", [])]

    matches = [step for step in steps if step.get("name") == CI_STEP_NAME]
    assert len(matches) == 1, [step.get("name") for step in steps]
    step = matches[0]
    assert step["env"]["DATABASE_URL"].startswith("sqlite+aiosqlite:///")

    for module in (
        GATE_MODULE,
        "tests/retrieval/test_rerank_pool.py",
        "tests/retrieval/test_hybrid_recall.py",
        "tests/retrieval/test_event_loop_responsiveness.py",
        "tests/retrieval/test_p2_ablation_contract.py",
        # the graph-build offload suite (T9); needs no artifact guard
        "tests/retrieval/test_graph_build_offload.py",
        # the suites that had no CI step of their own before this task
        "tests/retrieval/test_semantic_rerank.py",
        "tests/retrieval/test_p0_final_safety.py",
        "tests/retrieval/test_retriever_filters.py",
        "tests/retrieval/test_e5_local.py",
        "tests/retrieval/test_embedding_fingerprint.py",
        "tests/retrieval/test_vector_degradation.py",
        "tests/retrieval/test_xs_parity.py",
        # the v4 ladder / FTS5 module — v4 IS the P2 schema step, and the
        # final fix wave found it wired into NO step (I3); it must not silently
        # leave again.
        "tests/lite/test_sqlite_schema_v4.py",
        "tests/observability",
    ):
        assert module in step["run"], f"{module} is not wired into the P2 step"

    ablation_steps = [s for s in steps if s.get("name") == CI_ABLATION_STEP_NAME]
    assert len(ablation_steps) == 1, [s.get("name") for s in steps]
    assert "eval/ablation_retrieval_p2.py" in ablation_steps[0]["run"]
    schema_steps = [
        s for s in jobs["gate-schema"]["steps"]
        if s.get("name") == "Run SQLite schema-adoption + v2 ladder tests (DB-free)"
    ]
    assert len(schema_steps) == 1
    assert "tests/lite/test_sqlite_schema_adoption.py" in schema_steps[0]["run"]
    assert "tests/lite/test_sqlite_schema_v2.py" in schema_steps[0]["run"]
    p2_steps = jobs["gate-p2"]["steps"]
    seed_steps = [s for s in p2_steps if s.get("name") == "Seed test config for P2 settings"]
    assert len(seed_steps) == 1 and seed_steps[0]["run"] == "cp .env.test.example .env"
    assert p2_steps.index(ablation_steps[0]) < p2_steps.index(seed_steps[0]) < p2_steps.index(step)

    # …and the existing gates remain wired in this workflow.
    for name_prefix, module in (
        ("Run P1a durable-canonical gate", "tests/retrieval/test_p1a_gate.py"),
        ("Run P1b Qdrant + migration", "tests/retrieval/test_p1b_gate.py"),
        ("Run P3 background-indexing", "tests/retrieval/test_p3_gate.py"),
    ):
        found = [s for s in steps if s.get("name", "").startswith(name_prefix)]
        assert found and module in found[0]["run"], f"{module} left its CI step"


def test_slow_gate_jobs_are_not_serialized():
    """Keep the long independent suites on parallel runners."""
    jobs = yaml.safe_load(CI_YML.read_text())["jobs"]
    for name in ("gate-p1b", "gate-p3", "gate-p4a", "gate-p4b", "gate-schema", "gate-p2"):
        assert "needs" not in jobs[name], f"{name} is serialized behind another job"


def test_p2_timing_probes_are_the_last_ci_command():
    jobs = yaml.safe_load(CI_YML.read_text())["jobs"]
    steps = jobs["gate-p2"]["steps"]
    step = steps[-1]
    assert step.get("name") == CI_STEP_NAME
    shell_commands = [line.strip() for line in step["run"].splitlines() if line.strip()]
    commands = [line.strip() for line in step["run"].splitlines() if line.strip().startswith("python -m pytest")]
    assert shell_commands[-1] == commands[-1]
    assert "not test_the_loop_beats_while_ingest_and_recall_run_concurrently" in commands[0]
    assert "not test_the_signed_rss_budget_holds_on_the_real_stack" in commands[0]
    assert "tests/retrieval/test_event_loop_responsiveness.py" in commands[-1]
    assert "test_the_loop_beats_while_ingest_and_recall_run_concurrently" in commands[-1]
    assert "test_the_signed_rss_budget_holds_on_the_real_stack" in commands[-1]
