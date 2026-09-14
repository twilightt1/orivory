"""
Phase 3 — ``MemoryRetriever``: the orchestrator for personal-context recall.

Pipeline (one call to :py:meth:`MemoryRetriever.recall`):

    1. Fetch personal context (pinned + last 7 days + last 20).
    2. LLM rewrite the query + extract entities (1 call; best-effort).
    3. Embed the rewritten query (1 call; falls back to original).
    4. Vector search in ChromaDB (top_k * 3 for rerank headroom).
    5. Hydrate the top candidates with full ``Memory`` rows from Postgres,
       including ``entity_links`` (so we can apply entity boost).
    6. Apply entity_boost + time_decay to each candidate.
    7. Sort by combined score, return top_k.
    8. Build the ``RecallTrace`` with timings + fallbacks used.

Every step degrades gracefully. The worst case (LLM down + ChromaDB
down + no context) still returns a 200 with an empty ``results`` list
and a trace indicating what was attempted.
"""
from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.models.entity import MemoryEntity
from app.models.memory import Memory
from app.retrieval.embedder import EmbeddingDimensionMismatch, embed_query
from app.retrieval.memory.context import fetch_personal_context
from app.retrieval.memory.correction import needs_rewrite as _needs_rewrite
from app.retrieval.memory.correction import state_of as _state_of
from app.retrieval.memory.query_rewriter import rewrite_query
from app.retrieval.memory.scoring import entity_boost, time_decay_score
from app.retrieval.memory.vector_store import search_memories
from app.schemas.Orivory import (
    RECALL_TRACE_STAGE_KEYS,
    MemoryResponse,
    MemoryWithScore,
    RecallResponse,
    RecallTrace,
)

log = logging.getLogger(__name__)


class MemoryRetriever:
    """High-level orchestrator. One instance per (db, user) pair."""

    def __init__(
        self,
        db: AsyncSession,
        user_id: UUID,
        *,
        half_life_days: float = 30.0,
        entity_boost_per_match: float = 0.3,
        entity_boost_max: float = 1.0,
        rerank_factor: int = 3,
        decay_floor: float = 0.1,
        semantic_rerank: bool | None = None,
    ) -> None:
        self.db = db
        self.user_id = user_id
        self.half_life_days = half_life_days
        self.entity_boost_per_match = entity_boost_per_match
        self.entity_boost_max = entity_boost_max
        self.rerank_factor = rerank_factor
        self.decay_floor = decay_floor
        # None → defer to the deployment flag (settings.RETRIEVAL_SEMANTIC_RERANK)
        self.semantic_rerank = (
            settings.RETRIEVAL_SEMANTIC_RERANK
            if semantic_rerank is None
            else semantic_rerank
        )

    # ── main entry point ─────────────────────────────────────────────────

    async def recall(
        self,
        query: str,
        top_k: int = 10,
        include_personal_context: bool = True,
    ) -> RecallResponse:
        """Run the full recall pipeline and return a ``RecallResponse``."""
        t0 = time.perf_counter()
        stage_ms: dict[str, float] = {
            **dict.fromkeys(RECALL_TRACE_STAGE_KEYS, 0.0),
            "rewrite_ms": 0.0,
            "embed_ms": 0.0,
            "search_ms": 0.0,
            "hydrate_ms": 0.0,
        }

        # 1) Personal context
        context: list[Memory] = []
        if include_personal_context:
            t_context = time.perf_counter()
            try:
                context = await fetch_personal_context(self.db, self.user_id)
                context = [
                    memory
                    for memory in context
                    if _state_of(memory) not in ("superseded", "dirty")
                ]
            except Exception as e:
                log.warning("fetch_personal_context failed", extra={"error": str(e)})
            finally:
                stage_ms["context"] = (time.perf_counter() - t_context) * 1000.0

        # 2) LLM rewrite + entity extraction (fast-path: skip LLM when no pronouns)
        t_rewrite = time.perf_counter()
        rewrite_skipped = False
        try:
            if include_personal_context or _needs_rewrite(query):
                rewrite_result = await rewrite_query(query, context=context)
            else:
                rewrite_result = {"rewritten_query": query, "entities": [],
                                  "reasoning": "fast-path: no pronouns", "_fallback_used": False}
                rewrite_skipped = True
        finally:
            stage_ms["rewrite_ms"] = (time.perf_counter() - t_rewrite) * 1000.0
        rewritten = rewrite_result["rewritten_query"]
        entities = rewrite_result["entities"]
        llm_fallback = bool(rewrite_result.get("_fallback_used"))
        llm_reasoning = rewrite_result.get("reasoning") or None

        # Lowercased entity names for matching
        query_entity_names: set[str] = {e["name"].lower() for e in entities}

        # 3) Embed (use rewritten if LLM succeeded, else original)
        t_embed = time.perf_counter()
        embedding = None
        embed_error: Exception | None = None
        try:
            embedding = await embed_query(rewritten if not llm_fallback else query)
        except Exception as e:
            embed_error = e
            log.error("embed_query failed", extra={"error": str(e)})
        finally:
            embed_ms = (time.perf_counter() - t_embed) * 1000.0
            stage_ms["embed_ms"] = embed_ms
            stage_ms["embed_compute"] = embed_ms
        if embed_error is not None:
            return self._empty_response(
                query, rewritten, entities, llm_fallback, llm_reasoning,
                context if include_personal_context else None,
                t0, reason=f"embedding_failed:{embed_error}",
                rewrite_skipped=rewrite_skipped, stage_ms=stage_ms,
            )
        assert embedding is not None

        # 4) Vector search (top_k * rerank_factor for headroom)
        t_search = time.perf_counter()
        candidates: list[dict] = []
        try:
            candidates = await search_memories(
                embedding,
                user_id=str(self.user_id),
                top_k=top_k * self.rerank_factor,
            )
        except EmbeddingDimensionMismatch:
            # A contract mismatch is a readiness/data-integrity failure, not
            # an ordinary no-match result.
            raise
        except Exception as e:
            log.error("search_memories failed", extra={"error": str(e)})
        finally:
            stage_ms["search_ms"] = (time.perf_counter() - t_search) * 1000.0

        num_candidates = len(candidates)

        # 5) Hydrate and authorize from SQL before any candidate text can be
        # sent to a remote reranker. Vector payload content is stale/untrusted.
        t_hydrate = time.perf_counter()
        try:
            memory_ids: list[UUID] = []
            for cand in candidates:
                try:
                    memory_ids.append(UUID(str(cand["memory_id"])))
                except (AttributeError, TypeError, ValueError):
                    log.debug("Skipping malformed vector candidate")
            if memory_ids:
                hydrated = await self._hydrate(memory_ids)
            else:
                hydrated = {}
        finally:
            stage_ms["hydrate_ms"] = (time.perf_counter() - t_hydrate) * 1000.0

        # Hide superseded + derived-dirty candidates and replace vector text
        # with the current SQL-owned document before rerank.
        t_eligibility = time.perf_counter()
        try:
            visible = []
            for cand in candidates:
                mid = str(cand.get("memory_id", ""))
                mem = hydrated.get(mid)
                if mem is None:
                    continue
                if _state_of(mem) in ("superseded", "dirty"):
                    continue
                authorized = dict(cand)
                authorized["memory_id"] = mid
                authorized["content"] = (
                    f"Title: {mem.title}\n{mem.content}" if mem.title else mem.content
                )
                visible.append(authorized)
            candidates = visible
        finally:
            stage_ms["eligibility"] = (time.perf_counter() - t_eligibility) * 1000.0

        # 5b) Semantic rerank (Jina cross-encoder, opt-in): the input is now
        # SQL-authorized current content. Fallback to vector order on an
        # ordinary reranker failure; contract failures already propagated.
        if self.semantic_rerank and len(candidates) > 1:
            fallback_candidates = candidates
            t_rerank = time.perf_counter()
            try:
                from app.retrieval.reranker import rerank

                reranked = await rerank(
                    rewritten if not llm_fallback else query,
                    candidates,
                )
            except Exception as e:
                log.warning(
                    "semantic rerank failed — using vector order",
                    extra={"error": str(e)},
                )
                candidates = fallback_candidates
            else:
                # Rerankers should return their input IDs. Ignore malformed or
                # newly invented IDs rather than letting the network response
                # widen the SQL authorization set.
                safe_reranked = []
                for cand in reranked or []:
                    if not isinstance(cand, dict):
                        continue
                    mid = str(cand.get("memory_id", ""))
                    try:
                        UUID(mid)
                    except (ValueError, AttributeError):
                        continue
                    if mid in hydrated:
                        safe_reranked.append(cand)
                if safe_reranked:
                    # Re-check ownership/state and refresh content after the
                    # network round in case SQL changed while scoring ran.
                    candidates = safe_reranked
                    t_revalidate = time.perf_counter()
                    refreshed = await self._hydrate(
                        [UUID(str(c["memory_id"])) for c in candidates]
                    )
                    stage_ms["hydrate_ms"] += (
                        time.perf_counter() - t_revalidate
                    ) * 1000.0
                    t_reeligibility = time.perf_counter()
                    current = []
                    for cand in candidates:
                        mid = str(cand.get("memory_id", ""))
                        mem = refreshed.get(mid)
                        if mem is None or _state_of(mem) in ("superseded", "dirty"):
                            continue
                        current_cand = dict(cand)
                        current_cand["memory_id"] = mid
                        current_cand["content"] = (
                            f"Title: {mem.title}\n{mem.content}" if mem.title else mem.content
                        )
                        current.append(current_cand)
                    candidates = current
                    hydrated = refreshed
                    stage_ms["eligibility"] += (
                        time.perf_counter() - t_reeligibility
                    ) * 1000.0
                else:
                    candidates = fallback_candidates
            finally:
                stage_ms["rerank"] = (time.perf_counter() - t_rerank) * 1000.0

        # 6) Score: entity_boost + time_decay
        scored: list[tuple[Memory, float, list[str]]] = []
        t_score = time.perf_counter()
        try:
            for cand in candidates:
                mid = cand["memory_id"]
                memory = hydrated.get(mid)
                if memory is None:
                    # Memory was deleted from PG but still in Chroma.
                    log.debug("Skipping stale Chroma candidate", extra={"memory_id": mid})
                    continue

                mem_entity_names: set[str] = {
                    link.entity.name.lower()
                    for link in (memory.entity_links or [])
                    if link.entity is not None and link.entity.name
                }

                # When the cross-encoder ranked this pool, its relevance score
                # IS the semantic signal — modifiers may only NUDGE it (±15%),
                # never multiply it away. The n=100 run measured the failure
                # mode: decay(0.1) × rerank(0.9) = 0.09 lost to
                # decay(0.97) × rerank(0.2) = 0.19 — the cross-encoder's
                # decision was overwritten by age. Salience/recency now break
                # near-ties instead of dominating.
                rerank_score = cand.get("rerank_score")
                has_rerank_score = rerank_score is not None
                base_score = float(rerank_score if has_rerank_score else cand["score"])

                if has_rerank_score:
                    # modifier nudge: +7.5% if fresh-ish salient, −7.5% if not
                    salience_mult = 0.925 + 0.15 * float(memory.salience or 0.5)
                    captured = memory.captured_at
                    if captured.tzinfo is None:
                        captured = captured.replace(tzinfo=UTC)
                    age_days = max(
                        0.0,
                        (datetime.now(UTC) - captured).total_seconds() / 86400.0,
                    )
                    fresh_mult = 1.0 if age_days < 90 else 0.95
                    final_score = base_score * salience_mult * fresh_mult * (
                        1.5 if memory.pinned else 1.0
                    )
                    reasons = [f"rerank:{base_score:.2f}"]
                    if memory.pinned:
                        reasons.append("pinned")
                    scored.append((memory, final_score, reasons))
                    continue

                # Rerank-off path: full modifier chain (entity boost + decay)
                score_after_boost, boost_reasons = entity_boost(
                    base_score,
                    mem_entity_names,
                    query_entity_names,
                    boost_per_match=self.entity_boost_per_match,
                    max_boost=self.entity_boost_max,
                )

                final_score, decay_reasons = time_decay_score(
                    score_after_boost,
                    captured_at=memory.captured_at,
                    salience=float(memory.salience or 0.5),
                    pinned=bool(memory.pinned),
                    half_life_days=self.half_life_days,
                    decay_floor=self.decay_floor,
                )

                reasons = boost_reasons + decay_reasons
                scored.append((memory, final_score, reasons))

            # 7) Sort by score desc, take top_k
            scored.sort(key=lambda t: t[1], reverse=True)
            top = scored[:top_k]
        finally:
            stage_ms["score"] = (time.perf_counter() - t_score) * 1000.0

        # 8) Build response
        t_serialization = time.perf_counter()
        try:
            results: list[MemoryWithScore] = []
            for memory, score, reasons in top:
                base = _memory_response(memory)
                results.append(
                    MemoryWithScore(
                        **base.model_dump(),
                        score=round(score, 6),
                        match_reasons=reasons,
                    )
                )

            latency_ms = (time.perf_counter() - t0) * 1000.0
            trace = RecallTrace(
                rewritten_query=rewritten,
                entities=entities,
                latency_ms=round(latency_ms, 2),
                num_candidates=num_candidates,
                num_results=len(results),
                used_personal_context=bool(include_personal_context and context),
                llm_fallback=llm_fallback,
                llm_reasoning=llm_reasoning,
                half_life_days=self.half_life_days,
                rewrite_skipped=rewrite_skipped,
                stage_ms=stage_ms,
            )
            response = RecallResponse(
                results=results,
                personal_context=[_memory_response(m) for m in context]
                                 if include_personal_context else None,
                trace=trace,
            )
        finally:
            stage_ms["serialization"] = (time.perf_counter() - t_serialization) * 1000.0
            stage_ms["total"] = (time.perf_counter() - t0) * 1000.0

        response.trace.stage_ms.update(stage_ms)
        response.trace.latency_ms = round(stage_ms["total"], 2)
        return response

    # ── helpers ─────────────────────────────────────────────────────────

    async def _hydrate(self, memory_ids: list[UUID]) -> dict[str, Memory]:
        """Fetch Memory rows + entity_links in one query, keyed by id (str)."""
        if not memory_ids:
            return {}
        stmt = (
            select(Memory)
            .where(Memory.id.in_(memory_ids), Memory.user_id == self.user_id)
            .options(selectinload(Memory.entity_links).selectinload(MemoryEntity.entity))
        )
        rows = (await self.db.execute(stmt)).scalars().all()
        return {str(m.id): m for m in rows}

    def _empty_response(
        self,
        query: str,
        rewritten: str,
        entities: list[dict],
        llm_fallback: bool,
        llm_reasoning: str | None,
        context: list[Memory] | None,
        t0: float,
        reason: str,
        rewrite_skipped: bool = False,
        stage_ms: dict[str, float] | None = None,
    ) -> RecallResponse:
        log.info("Returning empty recall", extra={"reason": reason})
        trace_stage_ms: dict[str, float] = dict.fromkeys(RECALL_TRACE_STAGE_KEYS, 0.0)
        trace_stage_ms.update(stage_ms or {})
        t_serialization = time.perf_counter()
        try:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            trace = RecallTrace(
                rewritten_query=rewritten,
                entities=entities,
                latency_ms=round(latency_ms, 2),
                num_candidates=0,
                num_results=0,
                used_personal_context=bool(context),
                llm_fallback=llm_fallback,
                llm_reasoning=llm_reasoning,
                half_life_days=self.half_life_days,
                rewrite_skipped=rewrite_skipped,
                stage_ms=trace_stage_ms,
            )
            response = RecallResponse(
                results=[],
                personal_context=[_memory_response(m) for m in context]
                                 if context else None,
                trace=trace,
            )
        finally:
            trace_stage_ms["serialization"] = (time.perf_counter() - t_serialization) * 1000.0
            trace_stage_ms["total"] = (time.perf_counter() - t0) * 1000.0

        response.trace.stage_ms.update(trace_stage_ms)
        response.trace.latency_ms = round(trace_stage_ms["total"], 2)
        return response


def _memory_response(memory: Memory) -> MemoryResponse:
    """Map ORM Memory.extra_metadata to API field `metadata`."""
    return MemoryResponse(
        id=memory.id,
        user_id=memory.user_id,
        parent_id=memory.parent_id,
        source_type=memory.source_type,
        source_ref=memory.source_ref,
        source_url=memory.source_url,
        title=memory.title,
        content=memory.content,
        summary=memory.summary,
        tags=memory.tags or [],
        salience=memory.salience,
        pinned=memory.pinned,
        recall_count=memory.recall_count,
        last_used_at=memory.last_used_at,
        captured_at=memory.captured_at,
        indexed_at=memory.indexed_at,
        updated_at=memory.updated_at,
        revision=memory.revision or 1,  # unsaved/detached rows carry the column default
        state=_state_of(memory),
        metadata=getattr(memory, "extra_metadata", getattr(memory, "metadata", {})) or {},
    )
