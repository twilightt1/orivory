"""
Phase 3 — ``MemoryRetriever``: the orchestrator for personal-context recall.

Pipeline (one call to :py:meth:`MemoryRetriever.recall`):

    0. Wait — bounded — for this tenant's own pending index intents (the
       freshness barrier), so a memory written moments ago is not read as a
       no-match.
    1. Fetch personal context (pinned + last 7 days + last 20).
    2. LLM rewrite the query + extract entities (1 call; best-effort).
    3. Embed the rewritten query (1 call; falls back to original).
    4. Vector search in Qdrant — one pool of
       `ceil(top_k * RETRIEVAL_RERANK_POOL_MULTIPLIER)` candidates (ruling
       R4(p2)) for the reranker, the filter and scoring to share.
    4a. When ``RETRIEVAL_HYBRID_ENABLED`` (opt-in, ruling R2(p2)): the SQLite
       FTS5 lexical leg over the same query, rank-fused with the dense pool by
       RRF over canonical memory UUIDs (ruling R11b(p2)). The lexical leg
       returns ids and scores only.
    5. Hydrate the top candidates with full ``Memory`` rows from Postgres,
       including ``entity_links`` (so we can apply entity boost).
    6. Apply entity_boost + time_decay to each candidate.
    7. Sort by combined score, return top_k.
    8. Build the ``RecallTrace`` with timings + fallbacks used.

Every step degrades gracefully, EXCEPT three typed signals that must never be
served as an empty result: an embedding contract mismatch, an unreachable
vector store, and a freshness barrier that timed out (a write still in flight
is not a no-match). The vector outage keeps that rule where this deployment
has no lexical index too (Postgres); where one exists (SQLite FTS5), the
recall answers from the lexical leg instead and counts the fallback (ruling
R19). Those remaining typed signals propagate to the API as a 503 readiness
error (see ``app.main`` handlers). Everything else (LLM down, DB read errors)
still returns a 200 with an empty ``results`` list and a trace indicating what
was attempted.
"""
from __future__ import annotations

import logging
import math
import time
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.models.entity import MemoryEntity
from app.models.memory import Memory
from app.observability.fallbacks import count_fallback
from app.retrieval.embedder import EmbeddingDimensionMismatch, embed_query
from app.retrieval.hybrid_retriever import fuse_by_uuid
from app.retrieval.memory import lexical_index
from app.retrieval.memory.context import fetch_personal_context
from app.retrieval.memory.correction import needs_rewrite as _needs_rewrite
from app.retrieval.memory.correction import state_of as _state_of
from app.retrieval.memory.freshness import await_freshness
from app.retrieval.memory.query_rewriter import rewrite_query
from app.retrieval.memory.scoring import entity_boost, time_decay_score
from app.retrieval.memory.vector_store import search_memories
from app.retrieval.vector_retriever import VectorUnavailableError
from app.schemas.Orivory import (
    RECALL_TRACE_ZERO_KEYS,
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
        pool_multiplier: float | None = None,
        decay_floor: float = 0.1,
        semantic_rerank: bool | None = None,
        hybrid: bool | None = None,
        rrf_k: int | None = None,
    ) -> None:
        self.db = db
        self.user_id = user_id
        self.half_life_days = half_life_days
        self.entity_boost_per_match = entity_boost_per_match
        self.entity_boost_max = entity_boost_max
        # None → defer to the deployment setting
        # (settings.RETRIEVAL_RERANK_POOL_MULTIPLIER), the same seam as
        # `semantic_rerank` below.
        self.pool_multiplier = (
            settings.RETRIEVAL_RERANK_POOL_MULTIPLIER
            if pool_multiplier is None
            else pool_multiplier
        )
        self.decay_floor = decay_floor
        # None → defer to the deployment flag (settings.RETRIEVAL_SEMANTIC_RERANK)
        self.semantic_rerank = (
            settings.RETRIEVAL_SEMANTIC_RERANK
            if semantic_rerank is None
            else semantic_rerank
        )
        # None → defer to the deployment flag (settings.RETRIEVAL_HYBRID_ENABLED,
        # shipped OFF by ruling R2(p2)): with the flag off the recall below is
        # the dense-only pipeline this class always was — the lexical index is
        # not probed and the trace gains no hybrid counters.
        self.hybrid = (
            settings.RETRIEVAL_HYBRID_ENABLED if hybrid is None else hybrid
        )
        # The RRF constant (ruling R11b(p2)); a seam so a test and the T7
        # ablation can pin a k without moving the deployment global.
        self.rrf_k = settings.RETRIEVAL_RRF_K if rrf_k is None else rrf_k
        # R25(p2): WHY an empty served order is empty. ``None`` means every leg
        # that ran answered — an empty result is then a genuine no-match, and
        # no caller may swap in a different ordering for it. Set by the two
        # degraded legs (embed failure, untyped store failure); read by
        # ``recall_ids``, which is the surface that has to tell the difference.
        self.degraded_reason: str | None = None

    # ── main entry point ─────────────────────────────────────────────────

    async def recall(
        self,
        query: str,
        top_k: int = 10,
        include_personal_context: bool = True,
        include_superseded: bool = False,
    ) -> RecallResponse:
        """Run the full recall pipeline and return a ``RecallResponse``.

        ``include_superseded`` (default off) keeps superseded rows eligible
        through the visibility filter so they can be RANKED and returned: the
        MCP id seam asks for it (ruling R11(p2)) because that surface owns its
        own ``history`` widening, and it widens at hydration, not here. Dirty
        rows stay hidden either way — never a served row, on any surface.
        """
        t0 = time.perf_counter()
        # R25(p2): this call's answer only — a reused instance must not carry a
        # previous call's degraded leg into an empty-but-healthy result.
        self.degraded_reason = None
        stage_ms: dict[str, float] = dict.fromkeys(RECALL_TRACE_ZERO_KEYS, 0.0)
        # The candidate counters (T3): each key is written where its leg
        # produces a REAL number — never pre-filled, so an absent key means
        # "this leg did not run" and a `0` is a measured zero.
        counts: dict[str, int] = {}

        # 0) Freshness barrier (P3, ruling R1): wait — bounded — for THIS
        # tenant's own pending index intents before anything reads the index,
        # so a memory written moments ago is never served as a no-match. The
        # wait is claimed through the drain loop's single-flight door; its
        # IndexFreshnessTimeout is deliberately NOT caught here — like the two
        # contract errors below it is a readiness failure, and the API answers
        # it with the typed 503 body (ruling R10). ``queue_wait`` records the
        # barrier's own measurement, in milliseconds, exactly once.
        stage_ms["queue_wait"] = (
            await await_freshness(
                user_id=str(self.user_id),
                timeout=settings.RECALL_FRESHNESS_BUDGET_SECONDS,
            )
            * 1000.0
        )

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
        except EmbeddingDimensionMismatch:
            # A contract mismatch is a readiness failure, not a degraded
            # embed: it must reach the caller as a typed error.
            raise
        except Exception as e:
            embed_error = e
            log.error("embed_query failed", extra={"error": str(e)})
        finally:
            embed_ms = (time.perf_counter() - t_embed) * 1000.0
            stage_ms["embed_ms"] = embed_ms
            stage_ms["embed_compute"] = embed_ms
        if embed_error is not None:
            # R25(p2): an empty order for THIS reason is not a no-match; the
            # MCP seam surfaces the reason so the boundary can fall back.
            self.degraded_reason = f"embedding_failed:{embed_error}"
            return self._empty_response(
                query, rewritten, entities, llm_fallback, llm_reasoning,
                context if include_personal_context else None,
                t0, reason=f"embedding_failed:{embed_error}",
                rewrite_skipped=rewrite_skipped, stage_ms=stage_ms,
            )
        assert embedding is not None

        # 4) Vector search: ONE pool shared by the eligibility filter, the
        # reranker and scoring — `ceil(top_k * pool_multiplier)` (ruling
        # R4(p2)), floored at the request's own top_k (a multiplier below 1.0
        # must not make the count invariant unsatisfiable).
        pool = max(top_k, math.ceil(top_k * self.pool_multiplier))
        t_search = time.perf_counter()
        candidates: list[dict] = []
        dense_down = False
        try:
            candidates = await search_memories(
                embedding,
                user_id=str(self.user_id),
                top_k=pool,
            )
        except EmbeddingDimensionMismatch:
            # A contract mismatch is a readiness/data integrity failure in
            # every mode, not an ordinary no-match result.
            raise
        except VectorUnavailableError:
            # R19(p2): a vector outage is not a no-match. Where this
            # deployment owns a lexical index (SQLite FTS5), the recall
            # answers from it and counts the fallback; where it does not
            # (Postgres), the typed readiness error stands.
            t_lexical = time.perf_counter()
            try:
                lexical_rows = await self._lexical_leg(
                    rewritten if not llm_fallback else query, pool
                )
            except Exception as e:
                # A lexical leg that cannot even be read is not an answer
                # either: fall through to the typed outage.
                log.error("lexical fallback failed", extra={"error": str(e)})
                lexical_rows = None
            finally:
                stage_ms["lexical"] = (time.perf_counter() - t_lexical) * 1000.0
            if lexical_rows is None:
                raise
            count_fallback("retrieval.vector_unavailable")
            log.warning(
                "vector store unavailable — answering from the lexical leg",
                extra={"user_id": str(self.user_id), "lexical": len(lexical_rows)},
            )
            # A single leg, through the same rank rule: a lexical row's fused
            # score is 1/(k+rank+1) — positive and monotone in rank, unlike
            # raw BM25 (negative, lower-is-better) which the score chain below
            # would sort upside down.
            candidates = fuse_by_uuid([], lexical_rows, k=self.rrf_k)
            counts["lexical"] = len(lexical_rows)
            dense_down = True
        except Exception as e:
            # R25(p2): the store failed without a typed signal. The answer
            # below can only be empty for an infrastructure reason — never a
            # no-match — so the failed leg is recorded for the MCP seam. (With
            # hybrid on, the lexical leg may still fill the pool; the seam's
            # caller only falls back when the served order ends up empty.)
            self.degraded_reason = f"search_failed:{type(e).__name__}"
            log.error("search_memories failed", extra={"error": str(e)})
        finally:
            stage_ms["search_ms"] = (time.perf_counter() - t_search) * 1000.0

        # The DENSE page size drives the refill gate below (a page that filled
        # its limit is what justifies one more fetch): fusion and the outage
        # fallback add rows on top of it but must not change when a refill is
        # earned. `num_candidates` is the pool the answer was built from.
        dense_page = 0 if dense_down else len(candidates)
        num_candidates = len(candidates)
        if not dense_down:
            counts["dense"] = dense_page  # the first fetch, pre-filter (T3)
        # The lexical page the hybrid fusion ran over, kept for the refill:
        # R21(p2) re-fuses the WIDENED dense page with this SAME page. `None`
        # means no fusion ran at all (flag off, no FTS, or the leg failed
        # open) — the pool is then on the store's own cosine scale, and the
        # refill keeps the plain append path.
        lexical_page: list[dict] | None = None

        # 4a) Hybrid fusion (opt-in, rulings R2/R11b). Both legs return
        # ids/scores only, so nothing but ids reaches hydration — the SQL
        # authorization below still owns every byte of text.
        if self.hybrid and not dense_down:
            t_lexical = time.perf_counter()
            try:
                lexical_rows = await self._lexical_leg(
                    rewritten if not llm_fallback else query, pool
                )
            except Exception as e:
                # Fail open: the FTS index is derived data, and the dense
                # answer already in hand must not die with it. Logged, not
                # counted — no registered fallback path describes this, and a
                # mislabelled counter is worse than a log line.
                stage_ms["lexical"] = (time.perf_counter() - t_lexical) * 1000.0
                log.error(
                    "lexical leg failed — dense-only answer",
                    extra={"error": str(e), "kind": type(e).__name__},
                )
            else:
                if lexical_rows is None:
                    # No FTS on this deployment (R3(p2)): the leg never ran,
                    # so it writes no counter and no stage ms (C3).
                    log.info("hybrid recall without a lexical index — dense-only")
                else:
                    stage_ms["lexical"] = (time.perf_counter() - t_lexical) * 1000.0
                    counts["lexical"] = len(lexical_rows)
                    lexical_page = lexical_rows
                    candidates = fuse_by_uuid(candidates, lexical_rows, k=self.rrf_k)
                    counts["fused"] = len(candidates)
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
                if _hidden(mem, include_superseded=include_superseded):
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

        # 5a) Bounded refill (ruling R12(p2)): the visibility/state filter can
        # eat most of a pool that WAS full while the caller asked for top_k.
        # At most ONE extra search, only when the first fetch filled its limit
        # (a short page means the store handed over everything it had, so a
        # larger limit would only repeat it), hard-capped at top_k * 4. The
        # refill rows are hydrated, SQL-authorized and filtered exactly like
        # the first page's — nothing enters scoring unchecked.
        #
        # R21(p2): on the hybrid path that fetch is the WIDENED DENSE leg of a
        # re-fusion, never an append list. The widened page and the SAME
        # lexical page are fused again and the whole pool is re-authorized, so
        # every row that reaches scoring carries an RRF sum: an appended
        # refill row would carry its bare cosine (~0.3) next to sums capped at
        # 2/(k+1) ~ 0.033 and outrank the entire fused head — silently
        # discarding the fusion ordering in exactly the slice the refill
        # exists for. The widened page is the same snapshot's top-N with a
        # larger N, so it normally only adds rows; the re-fusion below still
        # unions the previous pool back in, so the pool cannot shrink.
        if not dense_down and len(candidates) < top_k and dense_page >= pool:
            added = 0  # rows the refill really ADDS to the pool (T3 counter)
            t_refill = time.perf_counter()
            try:
                refill_rows = await search_memories(
                    embedding, user_id=str(self.user_id), top_k=top_k * 4
                )
                if self.hybrid and lexical_page is not None:
                    before = len(candidates)
                    widened = fuse_by_uuid(
                        refill_rows or [], lexical_page, k=self.rrf_k
                    )
                    refill_ids: list[UUID] = []
                    for cand in widened:
                        try:
                            refill_ids.append(UUID(str(cand.get("memory_id", ""))))
                        except (AttributeError, TypeError, ValueError):
                            log.debug("Skipping malformed refill candidate")
                    refetched = await self._hydrate(refill_ids)
                    visible: list[dict] = []
                    for cand in widened:
                        mid = str(cand.get("memory_id", ""))
                        mem = refetched.get(mid)
                        if mem is None or _hidden(mem, include_superseded=include_superseded):
                            continue
                        authorized = dict(cand)
                        authorized["memory_id"] = mid
                        authorized["content"] = (
                            f"Title: {mem.title}\n{mem.content}"
                            if mem.title
                            else mem.content
                        )
                        visible.append(authorized)
                        hydrated[mid] = mem
                    # Assigned only once the re-authorized pool is complete: a
                    # failure above leaves the old pool (already SQL-authorized)
                    # in place, never a raw fused page carrying vector copies.
                    # Never shrink: a widened page that (racing a concurrent
                    # index change) no longer returns a row the caller already
                    # had keeps that row — its first-page RRF sum still stands.
                    fused_ids = {str(cand.get("memory_id", "")) for cand in visible}
                    kept = [
                        cand
                        for cand in candidates
                        if str(cand.get("memory_id", "")) not in fused_ids
                    ]
                    added = len(visible) + len(kept) - before
                    candidates = [*visible, *kept]
                else:
                    seen_ids = {str(cand.get("memory_id", "")) for cand in candidates}
                    fresh: list[dict] = []
                    refill_ids = []
                    for cand in refill_rows or []:
                        mid = str(cand.get("memory_id", ""))
                        if not mid or mid in seen_ids:
                            continue
                        try:
                            refill_ids.append(UUID(mid))
                        except (AttributeError, TypeError, ValueError):
                            log.debug("Skipping malformed refill candidate")
                            continue
                        seen_ids.add(mid)
                        fresh.append(cand)
                    if fresh:
                        hydrated_refill = await self._hydrate(refill_ids)
                        for cand in fresh:
                            mid = str(cand["memory_id"])
                            mem = hydrated_refill.get(mid)
                            if mem is None or _hidden(mem, include_superseded=include_superseded):
                                continue
                            authorized = dict(cand)
                            authorized["memory_id"] = mid
                            authorized["content"] = (
                                f"Title: {mem.title}\n{mem.content}"
                                if mem.title
                                else mem.content
                            )
                            candidates.append(authorized)
                            hydrated[mid] = mem
                            added += 1
            except (EmbeddingDimensionMismatch, VectorUnavailableError):
                # The same contract as the first search: a store outage is a
                # readiness signal, never a short result list.
                raise
            except Exception as e:
                log.warning("rerank pool refill failed", extra={"error": str(e)})
            finally:
                # Written only when the refill ran (in `finally`, so a failed
                # attempt is recorded too) — an absent key means "not needed".
                stage_ms["refill"] = (time.perf_counter() - t_refill) * 1000.0
                counts["refill"] = added

        # 5b) Semantic rerank (Jina cross-encoder, opt-in): the input is now
        # SQL-authorized current content. A rerank FAILURE — the typed
        # RerankUnavailable/RerankInvalidResponse, or anything unexpected the
        # transport throws — degrades to dense order and is counted (R11(p2));
        # a SUCCESS merges instead of replacing (R10(p2)).
        if self.semantic_rerank and len(candidates) > 1:
            fallback_candidates = candidates
            t_rerank = time.perf_counter()
            try:
                from app.retrieval.reranker import rerank

                reranked = await rerank(
                    rewritten if not llm_fallback else query,
                    candidates,
                    top_n=top_k,
                )
            except Exception as e:
                # The already-registered counter is what turns a week-long
                # reranker outage into a visible alert instead of a silent
                # ordering change.
                log.warning(
                    "semantic rerank failed — using vector order",
                    extra={"error": str(e), "kind": type(e).__name__},
                )
                count_fallback("retrieval.rerank_failed")
            else:
                # MERGE, never replace (ruling R10(p2)): the reranker is handed
                # the whole pool but answers with at most `top_n` rows, so the
                # candidates it did not rank keep their dense order behind the
                # ranked head — the served count must never depend on it.
                # Rerankers return their input IDs: ignore malformed or newly
                # invented ones rather than letting the network response widen
                # the SQL authorization set.
                ranked: list[dict] = []
                for cand in reranked or []:
                    if not isinstance(cand, dict):
                        continue
                    mid = str(cand.get("memory_id", ""))
                    if mid in hydrated:
                        ranked.append(cand)
                seen_ids = {str(cand.get("memory_id", "")) for cand in ranked}
                # T3: the size of the reranked head that merged. A fallback
                # writes no `reranked` count — it served no reranked head.
                counts["reranked"] = len(ranked)
                candidates = ranked + [
                    cand
                    for cand in fallback_candidates
                    if str(cand.get("memory_id", "")) not in seen_ids
                ]
                # Re-check ownership/state and refresh content after the
                # network round in case SQL changed while scoring ran.
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
                    if mem is None or _hidden(mem, include_superseded=include_superseded):
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
            finally:
                stage_ms["rerank"] = (time.perf_counter() - t_rerank) * 1000.0

        # T3: the counters for the legs that feed scoring — the pool that
        # survived every filter (eligibility, refill, the post-network
        # re-validation), and the SQL rows it resolves against.
        counts["eligible"] = len(candidates)
        counts["hydrated"] = len(hydrated)

        # 6) Score: entity_boost + time_decay
        scored: list[tuple[Memory, float, list[str]]] = []
        t_score = time.perf_counter()
        try:
            for cand in candidates:
                mid = cand["memory_id"]
                memory = hydrated.get(mid)
                if memory is None:
                    # Memory was deleted from PG but still in the index.
                    log.debug("Skipping stale vector candidate", extra={"memory_id": mid})
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
            counts["returned"] = len(results)
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
                counts=counts,
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

    async def recall_ids(
        self, query: str, top_k: int = 10
    ) -> tuple[list[tuple[UUID, float]], str | None]:
        """The ranked-ids view of :py:meth:`recall` — the shared id seam.

        Ruling R22(p2): the MCP hub serves the SAME ordering this class
        computes, as ``[(memory_id, score), ...]`` best first, instead of
        re-implementing a second ranking. No personal-context block (the MCP
        payload has none) and superseded rows stay ELIGIBLE: that surface
        widens to them at hydration when its caller asks for history.

        Ruling R25(p2): the second half is WHY the first is empty, when empty
        for an infrastructure reason — the name of the failed leg (embed
        outage, untyped store failure), or ``None`` when every leg that ran
        answered. ``None`` with an empty list is a GENUINE no-match: the
        discriminator is this leg state, never the empty list itself.
        """
        response = await self.recall(
            query,
            top_k=top_k,
            include_personal_context=False,
            include_superseded=True,
        )
        return (
            [(UUID(str(r.id)), float(r.score)) for r in response.results],
            self.degraded_reason,
        )

    # ── helpers ─────────────────────────────────────────────────────────

    async def _lexical_leg(self, query: str, limit: int) -> list[dict] | None:
        """The FTS5 lexical leg, or ``None`` when this deployment has no index.

        ``is_available`` answers False on a non-SQLite connection WITHOUT
        touching it (ruling R3(p2)), so Postgres pays one boolean instead of a
        caught exception. The search itself is the SYNC surface T4 pinned —
        ``run_sync`` hands over the sync Session, whose ``connection()`` is the
        Connection that surface takes — and it returns ``{"memory_id",
        "score", "rank"}``: ids and scores only, zero-based ranks, tenant +
        visibility filtered before its LIMIT, and global BM25 in ``score``
        which nothing downstream may read as tenant-local (fusion is by rank,
        R11b(p2)).
        """

        def _search(session):
            conn = session.connection()
            if not lexical_index.is_available(conn):
                return None
            return lexical_index.search(conn, query, user_id=self.user_id, limit=limit)

        return await self.db.run_sync(_search)

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
        trace_stage_ms: dict[str, float] = dict.fromkeys(RECALL_TRACE_ZERO_KEYS, 0.0)
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
                # The one counter this path owns: the response really carried
                # zero results. Every other counter names a leg that never ran
                # — absent, never a fabricated zero (T3/C3).
                counts={"returned": 0},
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


def _hidden(memory: Memory, *, include_superseded: bool = False) -> bool:
    """Rows no serving path may return: dirty always, superseded unless asked.

    Superseded rows are history — the API path hides them unconditionally,
    while ``recall(include_superseded=True)`` keeps them rankable so the MCP
    seam can widen to them at hydration (ruling R11(p2)). Dirty rows are wrong
    data, not history: never eligible, on any surface.
    """
    state = _state_of(memory)
    return state == "dirty" or (state == "superseded" and not include_superseded)


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
