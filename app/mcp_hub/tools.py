"""MCP memory tools — the hub's public surface to agents.

Design rules:
  - Every tool resolves its own AgentPrincipal (via the ``_current_principal``
    seam — in production the FastMCP wrappers in ``server.py`` publish the
    principal resolved from the MCP Context's HTTP request headers) and
    enforces scopes; ordinary tool failures return ``{"error": ...}`` dicts.
    Embedding contract failures raise their typed integrity error.
  - Every authorized call appends a ``MemoryAccessLog`` row — the ledger is the
    product. Identity/scope denials return *before* any DB write.
  - Reads bump nothing (salience bumping stays in the chat pipeline); writes
    reuse ``index_new_memory`` so embedding + graph stay best-effort.
  - ``search_memory`` ranks through the SHARED recall seam — the same
    ``MemoryRetriever`` ordering the API serves, never a second ranking
    implementation (ruling R22(p2)) — while keeping its own index-only payload.
    The recall's typed readiness errors (freshness barrier timeout, vector
    outage) degrade to the SQL ordering at this boundary instead of becoming a
    5xx the MCP host cannot parse (ruling R23(p2)); so does a degraded leg
    that would otherwise be served as a confident ``results: []`` (embed
    outage, untyped store failure — ruling R25(p2)).
"""
from __future__ import annotations

import logging
from contextvars import ContextVar
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import literal, select, tuple_

from app.database import AsyncSessionLocal
from app.mcp_hub.identity import (
    ACTION_ADD,
    ACTION_CORRECT,
    ACTION_DELETE,
    ACTION_FORGET,
    ACTION_GET,
    ACTION_LIST,
    ACTION_SEARCH,
    AgentPrincipal,
)
from app.models.memory import Memory
from app.models.memory_access_log import MemoryAccessLog
from app.observability.fallbacks import count_fallback
from app.retrieval.embedder import EmbeddingDimensionMismatch
from app.retrieval.memory.correction import Slot, get_cm, resolve_correction, state_of
from app.retrieval.memory.namespaces import namespace_of, personal_namespace
from app.retrieval.memory.outbox import IndexFreshnessTimeout, mark_done
from app.retrieval.memory.retriever import MemoryRetriever
from app.retrieval.memory.visibility import namespace_predicate, not_dirty_predicate
from app.retrieval.memory.write_back import index_new_memory
from app.retrieval.vector_retriever import VectorUnavailableError
from app.services.erasure_service import erase_memories

log = logging.getLogger(__name__)

MAX_SEARCH_LIMIT = 20
MAX_LIST_LIMIT = 100

# R23(p2): the counter for "search could not rank semantically and answered
# from the SQL ordering" — a rising rate means the recall path is failing.
SQL_FALLBACK_PATH = "mcp.search_sql_fallback"

IDENTITY_ERROR = {"error": "agent identity required"}
READ_SCOPE_ERROR = {"error": "scope memory:read required"}
WRITE_SCOPE_ERROR = {"error": "scope memory:write required"}

# Set by the FastMCP wrappers in server.py for the duration of one tool call;
# ``_current_principal`` reads it so the tool bodies stay framework-free
# (and tests can monkeypatch the function outright).
_principal_var: ContextVar[AgentPrincipal | None] = ContextVar("mcp_hub_principal", default=None)


def _current_principal() -> AgentPrincipal | None:
    """Principal for the active MCP call; ``None`` outside an MCP request."""
    return _principal_var.get()


def _session():
    """DB session seam — production wraps ``AsyncSessionLocal`` (async CM)."""
    return AsyncSessionLocal()


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _memory_brief(memory: Memory) -> dict[str, Any]:
    return {
        "id": str(memory.id),
        "title": memory.title,
        "content": memory.content,
        "tags": list(memory.tags or []),
        "salience": memory.salience,
        "captured_at": _iso(memory.captured_at),
        "state": state_of(memory),
    }


def _owned(memory: Memory | None, principal: AgentPrincipal) -> bool:
    """May ``principal`` read/write ``memory``? Theirs AND in their namespace.

    The same spelling as the REST surfaces' check (``_owned`` in
    ``app/api/v1/memories.py``): a primary-key read (``db.get``) cannot carry a
    predicate, so the boundary is checked on the loaded row against the same
    ``namespaces`` value the SQL predicate is built from. A foreign row and a
    missing one answer the same "not found".
    """
    return (memory is not None
            and memory.user_id == principal.user_id
            and namespace_of(memory) == personal_namespace(principal.user_id))


def _namespace_of(principal: AgentPrincipal) -> str:
    """The principal's own namespace — every read/write below is scoped to it."""
    return personal_namespace(principal.user_id)


def _memory_index_row(memory: Memory) -> dict[str, Any]:
    """Progressive-disclosure index row: no full content, a snippet only.

    Search results are for FILTERING — full content is fetched per-id by
    ``get_memory`` after the caller decides which hits matter. Snippets are
    clipped to 160 chars so a 20-hit search stays ~1k tokens instead of
    dumping every memory body into context.
    """
    content = memory.content or ""
    return {
        "id": str(memory.id),
        "title": memory.title,
        "snippet": (content[:160].rstrip() + "…") if len(content) > 160 else content,
        "tags": list(memory.tags or []),
        "salience": memory.salience,
        "captured_at": _iso(memory.captured_at),
        "state": state_of(memory),
    }


def _ledger_entry(
    principal: AgentPrincipal,
    action: str,
    *,
    memory_id: UUID | None = None,
    detail: dict[str, Any] | None = None,
) -> MemoryAccessLog:
    """Build one append-only ledger row (never logged back to the caller)."""
    return MemoryAccessLog(
        user_id=principal.user_id,
        agent_client_id=principal.agent_client_id,
        action=action,
        memory_id=memory_id,
        detail=detail if detail is not None else {},
    )


async def _sql_recall_ids(principal: AgentPrincipal, limit: int) -> list[tuple[UUID, float]]:
    """The SQL ordering: salience desc, then captured_at desc.

    The pre-P2 body of the seam, kept as the R23(p2)/R25(p2) fallback when the
    recall path cannot answer. Dirty rows are filtered before the LIMIT so they
    cannot consume capped candidate slots; superseded rows stay eligible
    (history widening happens at the hydration step). The namespace predicate
    is the same boundary the healthy path reads — the fallback is a read of
    ``memories`` like any other, never a wider one.

    The score half of each pair is the row's RAW salience, NOT the recall's
    fused/decayed score — the two orderings' numbers live on different scales.
    Only the ids (the ordering itself) are consumed downstream today, so the
    mismatch is invisible; never start comparing these scores with the healthy
    path's.
    """
    namespace = _namespace_of(principal)
    async with _session() as db:
        rows = (
            await db.execute(
                select(Memory.id, Memory.salience)
                .where(Memory.user_id == principal.user_id,
                       namespace_predicate(namespace),
                       not_dirty_predicate())
                .order_by(Memory.salience.desc(), Memory.captured_at.desc())
                .limit(limit)
            )
        ).all()
    return [(row.id, float(row.salience)) for row in rows]


async def _recall_memory_ids(query: str, limit: int) -> list[tuple[UUID, float]]:
    """Recall seam used by ``search_memory`` — the SHARED retriever ordering.

    Ruling R22(p2): the ordering is ``MemoryRetriever.recall_ids`` — the same
    pipeline (barrier, dense/hybrid/lexical, rerank, counters) the API serves,
    through the deployment's own flags. MCP adds no ranking of its own.

    Ruling R23(p2): the recall's typed readiness errors are caught HERE, at the
    tool boundary, and the call answers from the SQL ordering instead — a tool
    call must never become a 5xx the MCP host cannot parse (MCP had no barrier
    before this wiring; it must not gain one's 503 semantics). Ruling R25(p2):
    a DEGRADED leg that surfaces no error (embed outage, untyped store
    failure) gets the same treatment when the order it produced is empty —
    the seam's ``degraded_reason`` is the discriminator, never the empty list.
    The embedding contract mismatch is NOT caught by any clause here: it is a
    data-integrity failure and keeps raising its typed error, like every other
    tool in this module.

    Tests monkeypatch this function and return ``[(memory_id, score), ...]``
    pairs.
    """
    principal = _current_principal()
    if principal is None:
        return []
    try:
        async with _session() as db:
            recalled, degraded_reason = await MemoryRetriever(
                db, principal.user_id
            ).recall_ids(query, top_k=limit)
    except (IndexFreshnessTimeout, VectorUnavailableError) as exc:
        count_fallback(SQL_FALLBACK_PATH)
        log.warning(
            "MCP search answered from the SQL ordering: %s: %s",
            type(exc).__name__, exc,
        )
        return await _sql_recall_ids(principal, limit)
    if degraded_reason is not None and not recalled:
        # R25(p2): the recall degraded a leg and served nothing — an
        # infrastructure failure, not a no-match. The caller must not read that
        # as "no memories matched": same escape as the typed errors above, same
        # counter, one ledger row downstream. A NON-empty order from a
        # partially degraded pipeline is the recall's answer and is served
        # as-is (that is the ordering the API would serve).
        count_fallback(SQL_FALLBACK_PATH)
        log.warning("MCP search answered from the SQL ordering: %s", degraded_reason)
        return await _sql_recall_ids(principal, limit)
    return recalled


async def search_memory(query: str, limit: int = 8, include_history: bool = False) -> dict[str, Any]:
    """Search the caller's memories — returns an INDEX, not full content.

    Progressive-disclosure step 1 (the workflow that saves ~10x tokens):
    results carry ``id/title/salience/captured_at/snippet`` only. Review the
    index, then call ``get_memory`` on the few ids that matter (or
    ``timeline`` for context around one). Do NOT fetch details for every hit.
    Requires the ``memory:read`` scope.

    Ranking is the shared recall's (ruling R22(p2)): the caller's tenant, the
    same dense/hybrid/lexical semantics as the API, inherited from the
    deployment's flags. That ranking runs with superseded rows ELIGIBLE
    (``recall_ids`` passes ``include_superseded=True``) while this tool drops
    them at hydration unless ``include_history`` is set — so a superseded row
    can occupy one of the capped window's slots and leave fewer than ``limit``
    current rows in the answer. With the recall path down (freshness barrier,
    vector outage) or a leg degraded (embed outage, store failure), the answer
    falls back to the SQL ordering instead of failing or reading as an empty
    match (rulings R23(p2)/R25(p2)).
    """
    principal = _current_principal()
    if principal is None:
        return IDENTITY_ERROR
    if not principal.can_read():
        return READ_SCOPE_ERROR
    capped = max(1, min(limit, MAX_SEARCH_LIMIT))
    async with _session() as db:
        recalled = await _recall_memory_ids(query, capped)
        results: list[dict[str, Any]] = []
        if recalled:
            # Hydrate the ranked ids; re-check ownership in the same query so
            # the recall seam can never widen access beyond the principal —
            # tenant AND namespace.
            rows = (
                await db.execute(
                    select(Memory).where(
                        Memory.id.in_([mid for mid, _ in recalled]),
                        Memory.user_id == principal.user_id,
                        namespace_predicate(_namespace_of(principal)),
                        # Dirty rows are never served, not even in history.
                        not_dirty_predicate(),
                    )
                )
            ).scalars().all()
            by_id = {row.id: row for row in rows}
            rows_in_rank = [mem for mem in (by_id.get(mid) for mid, _ in recalled) if mem is not None]
            # Defence in depth (the query above already filters dirty): this
            # mirror of state_of is the layer that also catches a hand-written
            # falsy marker. History widens to superseded, never to dirty.
            rows_in_rank = [m for m in rows_in_rank if state_of(m) != "dirty"]
            if not include_history:
                rows_in_rank = [m for m in rows_in_rank if state_of(m) != "superseded"]
            results = [_memory_index_row(m) for m in rows_in_rank]
        db.add(
            _ledger_entry(
                principal,
                ACTION_SEARCH,
                detail={
                    "query": query,
                    "returned": len(results),
                    "memory_ids": [entry["id"] for entry in results],
                },
            )
        )
        await db.commit()
    return {"query": query, "results": results}


async def timeline(memory_id: str, window: int = 4) -> dict[str, Any]:
    """Context AROUND one memory (progressive-disclosure step 2).

    Given an id from ``search``, returns the memory plus up to ``window``
    neighbours captured before and after it — chronological context without
    fetching every detail. Cheaper than ``get_memory`` on many ids when
    you need surrounding history. Requires the ``memory:read`` scope.

    The anchor is read by primary key (the caller asked for THAT id: theirs and
    in their namespace, answered "not found" otherwise — no existence oracle).
    The neighbours are a query, so their predicate is in the statement: same
    tenant, same namespace, and never a dirty row — a stale derived row is
    wrong data, not the history this tool exists to show. Superseded
    neighbours stay, labelled.
    """
    principal = _current_principal()
    if principal is None:
        return IDENTITY_ERROR
    if not principal.can_read():
        return READ_SCOPE_ERROR
    try:
        anchor_id = UUID(memory_id)
    except ValueError:
        return {"error": "invalid memory id"}
    capped = max(1, min(window, 10))
    async with _session() as db:
        anchor = await db.get(Memory, anchor_id)
        if anchor is None or not _owned(anchor, principal):
            return {"error": "memory not found"}

        def _before_or_at(m: Memory) -> bool:
            return (m.captured_at, m.id) < (anchor.captured_at, anchor.id)

        def _after(m: Memory) -> bool:
            return (m.captured_at, m.id) > (anchor.captured_at, anchor.id)

        boundary = namespace_predicate(_namespace_of(principal))
        before_rows = (
            (
                await db.execute(
                    select(Memory)
                    .where(
                        Memory.user_id == principal.user_id,
                        boundary,
                        not_dirty_predicate(),
                        tuple_(Memory.captured_at, Memory.id) < tuple_(literal(anchor.captured_at), literal(anchor.id)),
                    )
                    .order_by(Memory.captured_at.desc(), Memory.id.desc())
                    .limit(capped)
                )
            )
            .scalars()
            .all()
        )
        after_rows = (
            (
                await db.execute(
                    select(Memory)
                    .where(
                        Memory.user_id == principal.user_id,
                        boundary,
                        not_dirty_predicate(),
                        tuple_(Memory.captured_at, Memory.id) > tuple_(literal(anchor.captured_at), literal(anchor.id)),
                    )
                    .order_by(Memory.captured_at.asc(), Memory.id.asc())
                    .limit(capped)
                )
            )
            .scalars()
            .all()
        )

        neighbours = [
            m
            for m in before_rows
            if _before_or_at(m) and m.id != anchor.id
        ][:capped]
        neighbours_after = [m for m in after_rows if _after(m)][:capped]

        db.add(
            _ledger_entry(
                principal,
                ACTION_SEARCH,
                detail={
                    "timeline_anchor": str(anchor.id),
                    "window": capped,
                    "returned": len(neighbours) + len(neighbours_after) + 1,
                },
            )
        )
        await db.commit()

    def _row(m: Memory) -> dict[str, Any]:
        return {"id": str(m.id), "title": m.title,
                "snippet": (m.content or "")[:160], "captured_at": _iso(m.captured_at),
                "state": state_of(m)}

    return {
        "anchor": _row(anchor),
        "before": [_row(m) for m in reversed(neighbours)],
        "after": [_row(m) for m in neighbours_after],
    }


async def get_memory(memory_id: str) -> dict[str, Any]:
    """Fetch one memory owned by the caller (requires ``memory:read``)."""
    principal = _current_principal()
    if principal is None:
        return IDENTITY_ERROR
    if not principal.can_read():
        return READ_SCOPE_ERROR
    try:
        mid = UUID(memory_id)
    except ValueError:
        return {"error": "invalid memory id"}
    async with _session() as db:
        # Ownership is enforced in the query: a foreign id — another tenant's,
        # or one of the caller's own rows outside their namespace — reads as
        # "not found" instead of leaking a memory.
        row = (
            await db.execute(
                select(Memory).where(
                    Memory.id == mid,
                    Memory.user_id == principal.user_id,
                    namespace_predicate(_namespace_of(principal)),
                )
            )
        ).scalars().first()
        db.add(_ledger_entry(principal, ACTION_GET, memory_id=mid, detail={"found": row is not None}))
        await db.commit()
    if row is None:
        return {"error": "memory not found"}
    return {**_memory_brief(row), **_memory_provenance(row)}


async def list_recent(limit: int = 20) -> dict[str, Any]:
    """List the caller's most recent memories (requires ``memory:read``)."""
    principal = _current_principal()
    if principal is None:
        return IDENTITY_ERROR
    if not principal.can_read():
        return READ_SCOPE_ERROR
    capped = max(1, min(limit, MAX_LIST_LIMIT))
    async with _session() as db:
        rows = (
            await db.execute(
                select(Memory)
                .where(Memory.user_id == principal.user_id,
                       namespace_predicate(_namespace_of(principal)),
                       not_dirty_predicate())
                .order_by(Memory.captured_at.desc(), Memory.id.desc())
                .limit(capped)
            )
        ).scalars().all()
        results = [_memory_brief(row) for row in rows]
        db.add(
            _ledger_entry(
                principal,
                ACTION_LIST,
                detail={"returned": len(results), "memory_ids": [entry["id"] for entry in results]},
            )
        )
        await db.commit()
    return {"results": results}


async def add_memory(title: str, content: str, tags: list[str] | None = None) -> dict[str, Any]:
    """Store a new memory owned by the caller (requires ``memory:write``)."""
    principal = _current_principal()
    if principal is None:
        return IDENTITY_ERROR
    if not principal.can_write():
        return WRITE_SCOPE_ERROR
    # Compression-before-storage (feature-flagged, best-effort): failures
    # store the original content — never block an agent write.
    summary_out: str | None = None
    from app.services.compression_service import compress_memory

    compressed = await compress_memory(content)
    if compressed is not None:
        content, summary_out = compressed[1], compressed[0]

    async with _session() as db:
        out = await resolve_correction(
            db, user_id=principal.user_id, title=title, content=content,
            tags=list(tags or []), source_ref=f"agent:{principal.name}",
            summary=summary_out)
        memory = out["memory"]
    try:
        indexed = await index_new_memory(memory)  # best-effort: embed + graph enqueue
    except EmbeddingDimensionMismatch:
        raise
    except Exception as exc:
        indexed = False
        log.warning("MCP add_memory indexing failed for %s: %s", memory.id, exc)
    if indexed:
        # Indexed in the fast path: ack the durable intent so a boot drain does
        # not re-embed this revision.
        async with _session() as db:
            await mark_done(db, entity_id=memory.id, revision=memory.revision)
    async with _session() as db:
        db.add(
            _ledger_entry(
                principal,
                ACTION_ADD,
                memory_id=memory.id,
                detail={"title": title, "memory_id": str(memory.id)},
            )
        )
        await db.commit()
    return {**_memory_brief(memory), **_memory_provenance(memory)}


async def delete_memory(memory_id: str) -> dict[str, Any]:
    """Delete one memory owned by the caller (requires ``memory:write``).

    Goes through the durable erasure path (``erase_memories``): one closure
    transaction, a delete intent per affected id, and a receipt. Response shape
    is unchanged; ``receipt_id`` is additive.
    """
    principal = _current_principal()
    if principal is None:
        return IDENTITY_ERROR
    if not principal.can_write():
        return WRITE_SCOPE_ERROR
    try:
        mid = UUID(memory_id)
    except ValueError:
        return {"error": "invalid memory id"}
    async with _session() as db:
        row = (
            await db.execute(
                select(Memory).where(
                    Memory.id == mid,
                    Memory.user_id == principal.user_id,
                    namespace_predicate(_namespace_of(principal)),
                )
            )
        ).scalars().first()
        if row is None:
            # The id rides ``detail``, not the ``memory_id`` column: that column
            # FKs memories.id (SET NULL) and a dangling reference fails the
            # INSERT on Postgres. A row outside the caller's namespace is the
            # same "not found" — the erasure path never sees it.
            db.add(_ledger_entry(principal, ACTION_DELETE,
                                 detail={"deleted": False, "memory_id": str(mid)}))
            await db.commit()
            return {"error": "memory not found"}
        receipt = await erase_memories(db, principal.user_id, [mid],
                                       requested_by=f"agent:{principal.name}")
        receipt_id = str(receipt.id)
        db.add(_ledger_entry(principal, ACTION_DELETE,
                             detail={"deleted": True, "memory_id": str(mid),
                                     "receipt_id": receipt_id}))
        await db.commit()
    return {"deleted": True, "id": str(mid), "receipt_id": receipt_id}


async def forget_memory(memory_ids: list[str]) -> dict[str, Any]:
    """Erase memories + every derived artifact, with a verification receipt.

    Requires ``memory:write``. Foreign/missing ids are recorded in the
    receipt as ``not_found_or_foreign`` (never an existence leak). Every
    authorized call appends one ``mcp_forget`` ledger row pointing at the
    receipt; the receipt carries the per-target cascade + verification detail.
    """
    principal = _current_principal()
    if principal is None:
        return IDENTITY_ERROR
    if not principal.can_write():
        return WRITE_SCOPE_ERROR
    valid: list[UUID] = []
    invalid: list[str] = []
    for raw in memory_ids:
        try:
            valid.append(UUID(raw))
        except (ValueError, TypeError, AttributeError):
            invalid.append(raw)
    valid = list(dict.fromkeys(valid))  # dedupe: service call + ledger stay consistent
    if not valid:
        return {"error": "invalid memory id"}
    async with _session() as db:
        receipt = await erase_memories(db, principal.user_id, valid, requested_by=f"agent:{principal.name}")
        summary = receipt.detail.get("summary", {})
        db.add(
            _ledger_entry(
                principal,
                ACTION_FORGET,
                detail={
                    "receipt_id": str(receipt.id),
                    "requested": [str(m) for m in valid],
                    "erased": summary.get("erased", 0),
                    "skipped": summary.get("skipped", 0),
                },
            )
        )
        await db.commit()
    return {
        "receipt_id": str(receipt.id),
        "status": receipt.status,
        "erased": summary.get("erased", 0),
        "skipped": summary.get("skipped", 0),
        "invalid": invalid,
    }


def _memory_provenance(memory: Memory) -> dict[str, Any]:
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
        if not _owned(target, principal):
            # A foreign id and one of the caller's own rows outside their
            # namespace answer the same: correcting is a write, and a write
            # never reaches across the boundary.
            return {"error": "memory not found"}
    async with _session() as db:
        out = await resolve_correction(
            db, user_id=principal.user_id, title=title or (target.title if target else ""),
            content=content, slot=Slot.of(subject, attribute, scope),
            valid_from=valid_from, evidence_ids=list(evidence_ids or []),
            memory_id=str(target.id) if target else None,
            source_ref=f"agent:{principal.name}")
        new = out["memory"]
        db.add(_ledger_entry(principal, ACTION_CORRECT, memory_id=new.id,
            detail={"status": out["status"], "superseded": out["superseded"],
                    "dirtied": out["dirtied"], "memory_id": str(new.id)}))
        await db.commit()
    try:
        indexed = await index_new_memory(new)
    except EmbeddingDimensionMismatch:
        raise
    except Exception as exc:
        indexed = False
        log.warning("MCP correct_memory indexing failed for %s: %s", new.id, exc)
    if indexed:
        async with _session() as db:
            await mark_done(db, entity_id=new.id, revision=new.revision)
    return {"status": out["status"], "id": str(new.id),
            "superseded": out["superseded"], "dirtied": out["dirtied"],
            **_memory_provenance(new)}


__all__ = [
    "add_memory",
    "correct_memory",
    "delete_memory",
    "forget_memory",
    "get_memory",
    "list_recent",
    "search_memory", "timeline",
]
