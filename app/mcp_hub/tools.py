"""MCP memory tools — the hub's public surface to agents.

Design rules:
  - Every tool resolves its own AgentPrincipal (via the ``_current_principal``
    seam — in production the FastMCP wrappers in ``server.py`` publish the
    principal resolved from the MCP Context's HTTP request headers) and
    enforces scopes; failures return ``{"error": ...}`` dicts, never raise.
  - Every authorized call appends a ``MemoryAccessLog`` row — the ledger is the
    product. Identity/scope denials return *before* any DB write.
  - Reads bump nothing (salience bumping stays in the chat pipeline); writes
    reuse ``index_new_memory`` so embedding + graph stay best-effort.
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
from app.retrieval.memory.correction import Slot, get_cm, resolve_correction, state_of
from app.retrieval.memory.write_back import index_new_memory, safe_delete_from_chroma
from app.services.erasure_service import erase_memories

log = logging.getLogger(__name__)

MAX_SEARCH_LIMIT = 20
MAX_LIST_LIMIT = 100

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
    }


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


async def _recall_memory_ids(query: str, limit: int) -> list[tuple[UUID, float]]:
    """Recall seam used by ``search_memory``.

    MVP ranking is a plain SQL select — the user's memories ordered by
    salience desc, then captured_at desc (``query`` is kept for seam
    compatibility; semantic recall replaces this body later without touching
    the tool bodies). Tests monkeypatch this and return
    ``[(memory_id, score), ...]`` pairs.
    """
    principal = _current_principal()
    if principal is None:
        return []
    async with _session() as db:
        rows = (
            await db.execute(
                select(Memory.id, Memory.salience)
                .where(Memory.user_id == principal.user_id)
                .order_by(Memory.salience.desc(), Memory.captured_at.desc())
                .limit(limit)
            )
        ).all()
    return [(row.id, float(row.salience)) for row in rows]


async def search_memory(query: str, limit: int = 8, include_history: bool = False) -> dict[str, Any]:
    """Search the caller's memories — returns an INDEX, not full content.

    Progressive-disclosure step 1 (the workflow that saves ~10x tokens):
    results carry ``id/title/salience/captured_at/snippet`` only. Review the
    index, then call ``get_memory`` on the few ids that matter (or
    ``timeline`` for context around one). Do NOT fetch details for every hit.
    Requires the ``memory:read`` scope."""
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
            # the recall seam can never widen access beyond the principal.
            rows = (
                await db.execute(
                    select(Memory).where(
                        Memory.id.in_([mid for mid, _ in recalled]),
                        Memory.user_id == principal.user_id,
                    )
                )
            ).scalars().all()
            by_id = {row.id: row for row in rows}
            rows_in_rank = [mem for mem in (by_id.get(mid) for mid, _ in recalled) if mem is not None]
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
        if anchor is None or anchor.user_id != principal.user_id:
            return {"error": "memory not found"}

        def _before_or_at(m: Memory) -> bool:
            return (m.captured_at, m.id) < (anchor.captured_at, anchor.id)

        def _after(m: Memory) -> bool:
            return (m.captured_at, m.id) > (anchor.captured_at, anchor.id)

        before_rows = (
            (
                await db.execute(
                    select(Memory)
                    .where(
                        Memory.user_id == principal.user_id,
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
                "snippet": (m.content or "")[:160], "captured_at": _iso(m.captured_at)}

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
        # Ownership is enforced in the query: a foreign id reads as "not found"
        # instead of leaking another user's memory.
        row = (
            await db.execute(
                select(Memory).where(Memory.id == mid, Memory.user_id == principal.user_id)
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
                .where(Memory.user_id == principal.user_id)
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
        await index_new_memory(memory)  # best-effort: embed + graph enqueue
    except Exception as exc:
        log.warning("MCP add_memory indexing failed for %s: %s", memory.id, exc)
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
    """Delete one memory owned by the caller (requires ``memory:write``)."""
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
                select(Memory).where(Memory.id == mid, Memory.user_id == principal.user_id)
            )
        ).scalars().first()
        if row is None:
            db.add(_ledger_entry(principal, ACTION_DELETE, memory_id=mid, detail={"deleted": False}))
            await db.commit()
            return {"error": "memory not found"}
        await db.delete(row)
        await safe_delete_from_chroma(mid)  # best-effort vector cleanup
        db.add(_ledger_entry(principal, ACTION_DELETE, memory_id=mid, detail={"deleted": True}))
        await db.commit()
    return {"deleted": True, "id": str(mid)}


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
        if target is None or target.user_id != principal.user_id:
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
        await index_new_memory(new)
    except Exception as exc:
        log.warning("MCP correct_memory indexing failed for %s: %s", new.id, exc)
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
