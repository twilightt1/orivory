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
from app.retrieval.embedder import embed_query
from app.retrieval.memory.context import fetch_personal_context
from app.retrieval.memory.correction import needs_rewrite as _needs_rewrite
from app.retrieval.memory.correction import state_of as _state_of
from app.retrieval.memory.query_rewriter import rewrite_query
from app.retrieval.memory.scoring import entity_boost, lexical_bonus, time_decay_score
from app.retrieval.memory.vector_store import search_memories
from app.schemas.Orivory import (
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
        lexical_refine: bool = True,
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
        self.lexical_refine = lexical_refine

    # ── main entry point ─────────────────────────────────────────────────

    async def recall(
        self,
        query: str,
        top_k: int = 10,
        include_personal_context: bool = True,
    ) -> RecallResponse:
        """Run the full recall pipeline and return a ``RecallResponse``."""
        t0 = time.perf_counter()
        stage_ms: dict[str, float] = {}

        # 1) Personal context
        context: list[Memory] = []
        if include_personal_context:
            try:
                context = await fetch_personal_context(self.db, self.user_id)
            except Exception as e:
                log.warning("fetch_personal_context failed", extra={"error": str(e)})

        # 2) LLM rewrite + entity extraction (fast-path: skip LLM when no pronouns)
        t_rewrite = time.perf_counter()
        if include_personal_context or _needs_rewrite(query):
            rewrite_result = await rewrite_query(query, context=context)
            rewrite_skipped = False
        else:
            rewrite_result = {"rewritten_query": query, "entities": [],
                              "reasoning": "fast-path: no pronouns", "_fallback_used": False}
            rewrite_skipped = True
        stage_ms["rewrite_ms"] = (time.perf_counter() - t_rewrite) * 1000.0
        rewritten = rewrite_result["rewritten_query"]
        entities = rewrite_result["entities"]
        llm_fallback = bool(rewrite_result.get("_fallback_used"))
        llm_reasoning = rewrite_result.get("reasoning") or None

        # Lowercased entity names for matching
        query_entity_names: set[str] = {e["name"].lower() for e in entities}

        # 3) Embed (use rewritten if LLM succeeded, else original)
        t_embed = time.perf_counter()
        try:
            embedding = await embed_query(rewritten if not llm_fallback else query)
        except Exception as e:
            log.error("embed_query failed", extra={"error": str(e)})
            return self._empty_response(
                query, rewritten, entities, llm_fallback, llm_reasoning,
                context if include_personal_context else None,
                t0, reason=f"embedding_failed:{e}",
                rewrite_skipped=rewrite_skipped, stage_ms=stage_ms,
            )
        stage_ms["embed_ms"] = (time.perf_counter() - t_embed) * 1000.0

        # 4) Vector search (top_k * rerank_factor for headroom)
        t_search = time.perf_counter()
        try:
            candidates = await search_memories(
                embedding,
                user_id=str(self.user_id),
                top_k=top_k * self.rerank_factor,
            )
        except Exception as e:
            log.error("search_memories failed", extra={"error": str(e)})
            candidates = []
        stage_ms["search_ms"] = (time.perf_counter() - t_search) * 1000.0

        num_candidates = len(candidates)

        # 4b) Semantic rerank (Jina cross-encoder, opt-in): reorder the
        # candidate pool by true query↔document relevance before the
        # modifier pass. The vector cosine is an approximation; the
        # cross-encoder reads query + document together and is materially
        # better at "which of these 45 actually answers the question" —
        # the exact failure mode the benchmark runs measured. Fallback to
        # the vector order on any reranker failure (never block recall).
        if self.semantic_rerank and num_candidates > 1:
            try:
                from app.retrieval.reranker import rerank

                reranked = await rerank(
                    rewritten if not llm_fallback else query,
                    candidates,
                )
                if reranked:
                    candidates = reranked
            except Exception as e:
                log.warning(
                    "semantic rerank failed — using vector order",
                    extra={"error": str(e)},
                )

        # 5) Hydrate from Postgres (with entity_links)
        t_hydrate = time.perf_counter()
        if candidates:
            memory_ids = [UUID(c["memory_id"]) for c in candidates]
            hydrated = await self._hydrate(memory_ids)
        else:
            hydrated = {}
        stage_ms["hydrate_ms"] = (time.perf_counter() - t_hydrate) * 1000.0

        # 5b) Hide superseded + derived-dirty candidates (never in Chroma metadata)
        visible = []
        for cand in candidates:
            mem = hydrated.get(cand["memory_id"])
            if mem is None:
                continue
            if _state_of(mem) in ("superseded", "dirty"):
                continue
            visible.append(cand)
        candidates = visible

        # 6) Score: entity_boost + time_decay
        lex_query = rewritten if not llm_fallback else query
        scored: list[tuple[Memory, float, list[str]]] = []
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
            base_score = float(cand.get("rerank_score") or cand["score"])

            if "rerank_score" in cand:
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
                if self.lexical_refine:
                    bonus, lex_reasons = lexical_bonus(
                        lex_query, f"{memory.title or ''} {memory.content or ''}")
                    final_score += bonus
                    reasons += lex_reasons
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
            if self.lexical_refine:
                bonus, lex_reasons = lexical_bonus(
                    lex_query, f"{memory.title or ''} {memory.content or ''}")
                final_score += bonus
                reasons += lex_reasons
            scored.append((memory, final_score, reasons))

        # 7) Sort by score desc, take top_k
        scored.sort(key=lambda t: t[1], reverse=True)
        top = scored[:top_k]

        # 8) Build response
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
        return RecallResponse(
            results=results,
            personal_context=[_memory_response(m) for m in context]
                             if include_personal_context else None,
            trace=trace,
        )

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
            stage_ms=stage_ms or {},
        )
        return RecallResponse(
            results=[],
            personal_context=[_memory_response(m) for m in context]
                             if context else None,
            trace=trace,
        )


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
        metadata=getattr(memory, "extra_metadata", getattr(memory, "metadata", {})) or {},
    )
