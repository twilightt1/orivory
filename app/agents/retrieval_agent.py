from __future__ import annotations

import asyncio
import hashlib
import logging
import time

from app.agents.state import AgentState
from app.config import settings
from app.database import AsyncSessionLocal
from app.retrieval.bm25_retriever import bm25_retriever
from app.retrieval.hybrid_retriever import reciprocal_rank_fusion
from app.retrieval.parent_store import get_parents_batch
from app.retrieval.reranker import rerank
from app.retrieval.retrieval_cache import get_cached_chunks, set_cached_chunks
from app.retrieval.vector_retriever import VectorUnavailableError
from app.retrieval.vector_retriever import search as vector_search

log = logging.getLogger(__name__)

TOP_K_RAW = 15
TOP_K_FUSE = 20
TOP_N_FINAL = 5


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _query_hash(state: AgentState) -> str:
    # Include retry_count: a retrieval retry after "irrelevant context" must
    # not replay the byte-identical cached result (the retry would re-grade
    # the same chunks to the same verdict — MAX_RETRIES becomes pure burn).
    history_tail = state.get("history", [])[-2:]
    payload = f"{state['query']}::{history_tail}::retry={state.get('retry_count', 0)}"
    return hashlib.md5(payload.encode()).hexdigest()


async def _ensure_bm25_index(conversation_id: str) -> dict[str, bool | str]:
    try:
        async with AsyncSessionLocal() as db:
            return await bm25_retriever.ensure_async(db, conversation_id)
    except Exception as exc:
        log.warning(
            "BM25 lazy rebuild failed",
            extra={"conversation_id": conversation_id, "error": str(exc)},
        )
        return {
            "had_index": bm25_retriever.has_index(conversation_id),
            "rebuilt": False,
            "has_index": bm25_retriever.has_index(conversation_id),
            "error": str(exc),
        }


def _vector_unavailable(vector_results: list, flattened: list[dict]) -> bool:
    """Degradation classifier: True only when vector search itself failed.

    Empty results WITHOUT errors mean "genuinely no vectors" (new
    conversation, no matches) — not an outage. Partial success means the
    backend is reachable. Only all-failed + no-results is Chroma-down.
    """
    if flattened:
        return False
    return bool(vector_results) and all(
        isinstance(res, VectorUnavailableError) for res in vector_results
    )


async def retrieval_agent(state: AgentState) -> AgentState:
    cid = state["conversation_id"]
    state.setdefault("agent_trace", {})
    timing: dict[str, float] = {}
    total_start = time.perf_counter()

    if not state.get("has_documents"):
        state.update(
            {
                "fused_chunks": [],
                "bm25_results": [],
                "vector_results": [],
                "reranked_chunks": [],
            }
        )
        state["agent_trace"]["retrieval"] = "no_documents"
        state["agent_trace"].setdefault("timing", {})["retrieval_ms"] = _elapsed_ms(total_start)
        return state

    cache_start = time.perf_counter()
    query_hash = _query_hash(state)
    cached = await get_cached_chunks(cid, query_hash)
    timing["cache_lookup_ms"] = _elapsed_ms(cache_start)

    if cached is not None:
        state["reranked_chunks"] = cached
        state["fused_chunks"] = cached
        state["agent_trace"]["retrieval"] = {
            "cache": "hit",
            "final": len(cached),
            "retry_count": state.get("retry_count", 0),
        }
        state["agent_trace"].setdefault("timing", {})["retrieval_ms"] = _elapsed_ms(total_start)
        state["agent_trace"]["retrieval"]["timing_ms"] = timing
        return state

    all_result_lists: list[list[dict]] = []

    standalone = state.get("rewritten_query", state["query"])
    queries = list(state.get("search_variants", []))
    if standalone not in queries:
        queries.insert(0, standalone)

    bm25_start = time.perf_counter()
    bm25_index = await _ensure_bm25_index(cid)
    bm25_res = await bm25_retriever.search(standalone, TOP_K_RAW, cid)
    timing["bm25_ms"] = _elapsed_ms(bm25_start)
    if bm25_res:
        all_result_lists.append(bm25_res)
    state["bm25_results"] = bm25_res

    vector_start = time.perf_counter()

    async def _vector(q: str):
        return await vector_search(q, TOP_K_RAW, cid)

    vector_tasks = [_vector(q) for q in queries]
    vector_results = await asyncio.gather(*vector_tasks, return_exceptions=True)

    # Flatten vector results early so HyDE can append to it
    flattened_vector_results: list[dict] = []
    for res in vector_results:
        if isinstance(res, list) and res:
            flattened_vector_results.extend(res)
            all_result_lists.append(res)

    # ── HyDE Enhancement ────────────────────────────────────────────────────
    # If HyDE is enabled and we have a hypothetical document, use it for
    # additional retrieval to capture documents that match the "ideal answer" pattern
    if settings.HYDE_ENABLED and settings.HYDE_USE_IN_RETRIEVAL:
        hyde_state = state.get("hyde_result")
        if hyde_state and hyde_state.get("passages"):
            hyde_passages = hyde_state["passages"]
            log.debug(f"HyDE: Adding {len(hyde_passages)} hypothetical passages to vector search")

            # Search for each hypothetical passage
            hyde_tasks = [_vector(p) for p in hyde_passages]
            hyde_results = await asyncio.gather(*hyde_tasks, return_exceptions=True)

            # Add HyDE results to fusion pool
            for res in hyde_results:
                if isinstance(res, list) and res:
                    flattened_vector_results.extend(res)
                    all_result_lists.append(res)

            state["agent_trace"]["hyde_enhanced"] = {
                "passage_count": len(hyde_passages),
                "hyde_results_added": sum(1 for r in hyde_results if isinstance(r, list)),
            }
    # ── End HyDE Enhancement ───────────────────────────────────────────────
    state["vector_results"] = flattened_vector_results
    timing["vector_ms"] = _elapsed_ms(vector_start)

    # Degradation contract: Chroma down -> BM25-only (Postgres) + flag it.
    # Never silent: the flag lands in state, the trace, and a warning log
    # (countable for fallback-rate alerting). Zero extra Chroma round
    # trips — classification reuses the gather() results.
    vector_unavailable = _vector_unavailable(vector_results, flattened_vector_results)
    state["vector_unavailable"] = vector_unavailable
    if vector_unavailable:
        log.warning(
            "Vector search unavailable, BM25-only retrieval",
            extra={"conversation_id": cid, "query_variants": len(queries)},
        )

    if not all_result_lists:
        state["reranked_chunks"] = []
        state["agent_trace"]["retrieval"] = {
            "cache": "miss",
            "bm25_index": bm25_index,
            "bm25_result_count": len(bm25_res),
            "vector_result_count": len(flattened_vector_results),
            "vector_unavailable": vector_unavailable,
            "result": "no_results",
            "retry_count": state.get("retry_count", 0),
            "timing_ms": timing,
        }
        state["agent_trace"].setdefault("timing", {})["retrieval_ms"] = _elapsed_ms(total_start)
        return state

    fusion_start = time.perf_counter()
    fused_children = reciprocal_rank_fusion(all_result_lists)[:TOP_K_FUSE]
    state["fused_chunks"] = fused_children
    timing["fusion_ms"] = _elapsed_ms(fusion_start)

    parent_start = time.perf_counter()
    parent_ids = list(
        {
            c.get("parent_id") or c.get("metadata", {}).get("parent_id", "")
            for c in fused_children
            if c.get("parent_id") or c.get("metadata", {}).get("parent_id")
        }
    )

    async with AsyncSessionLocal() as db:
        parent_map = await get_parents_batch(cid, parent_ids, db=db)

    seen_parents: set[str] = set()
    expanded: list[dict] = []

    for child in fused_children:
        pid = child.get("parent_id") or child.get("metadata", {}).get("parent_id", "")
        if pid and pid not in seen_parents and pid in parent_map:
            parent = parent_map[pid]
            seen_parents.add(pid)
            expanded.append(
                {
                    **child,
                    "content": parent["content"],
                    "child_content": child["content"],
                    "parent_id": pid,
                }
            )
        elif not pid:
            expanded.append(child)

    if not expanded:
        expanded = fused_children[:TOP_N_FINAL]
    timing["parent_expansion_ms"] = _elapsed_ms(parent_start)

    state["agent_trace"]["parent_expansion"] = {
        "children_retrieved": len(fused_children),
        "unique_parents": len(seen_parents),
        "expanded": len(expanded),
    }

    rerank_start = time.perf_counter()
    try:
        rerank_input_size = max(15, TOP_N_FINAL * 2)
        reranked = await rerank(standalone, expanded[:rerank_input_size])
        min_rerank_score = 0.05
        top_reranked = [
            c for c in reranked if c.get("rerank_score", 0) > min_rerank_score
        ][:TOP_N_FINAL]
    except Exception as e:
        log.warning("Reranker failed", extra={"error": str(e)})
        top_reranked = expanded[:TOP_N_FINAL]
    timing["rerank_ms"] = _elapsed_ms(rerank_start)

    final = top_reranked
    state["reranked_chunks"] = final
    state["agent_trace"]["retrieval"] = {
        "cache": "miss",
        "bm25_index": bm25_index,
        "bm25_result_count": len(bm25_res),
        "vector_result_count": len(flattened_vector_results),
        "vector_unavailable": vector_unavailable,
        "query_variants": len(queries),
        "result_lists": len(all_result_lists),
        "fused_children": len(fused_children),
        "after_expansion": len(expanded),
        "final": len(final),
        "retry_count": state.get("retry_count", 0),
        "timing_ms": timing,
    }

    cache_set_start = time.perf_counter()
    await set_cached_chunks(cid, query_hash, final)
    timing["cache_set_ms"] = _elapsed_ms(cache_set_start)
    state["agent_trace"].setdefault("timing", {})["retrieval_ms"] = _elapsed_ms(total_start)

    return state
