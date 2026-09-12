"""
Phase 3 — Scoring helpers for personal-memory retrieval.

Two functions, both pure (no IO, no DB):

* ``time_decay_score`` combines a base vector-similarity score with
  salience and recency, plus a pinned bonus.

* ``entity_boost`` applies a multiplicative boost when the memory
  shares entities with the query.

Both return reasons alongside the new score so the caller can show
the user *why* a memory was selected (``match_reasons: [...]``).
"""
from __future__ import annotations

import math
from datetime import UTC, datetime

# ── time-decay scoring ──────────────────────────────────────────────────────


def time_decay_score(
    base_score: float,
    captured_at: datetime,
    salience: float = 0.5,
    pinned: bool = False,
    now: datetime | None = None,
    half_life_days: float = 30.0,
    decay_floor: float = 0.1,
) -> tuple[float, list[str]]:
    """
    Combine vector-similarity score with salience + recency.

    Formula:
        score = base * (0.5 + salience) * decay * pinned_mult

    - ``salience`` ∈ [0, 1] → multiplier ∈ [0.5, 1.5]
    - ``pinned`` → ×1.5 (evergreen)

    Decay is a FLOOR at ``decay_floor`` (default 0.1): unbounded exponential
    decay made semantic match irrelevant for anything older than a few
    months — a 2023 memory scored against a 2026 "now" got decay ≈ 1e-16,
    so ranking became pure noise (measured in the LongMemEval runs: the
    full-context baseline beat the stack 0.600 vs 0.450/0.300). With the
    floor, semantic match always dominates and recency only breaks ties
    among near-equal vector scores. Set ``decay_floor=0.0`` to restore the
    unbounded behavior.

    Returns ``(new_score, match_reasons)``.
    """
    if now is None:
        now = datetime.now(UTC)

    # Normalize to UTC-aware datetimes so subtraction works
    if captured_at.tzinfo is None:
        captured_at = captured_at.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    age_seconds = max(0.0, (now - captured_at).total_seconds())
    age_days = age_seconds / 86400.0

    decay = max(decay_floor, math.exp(-age_days / half_life_days))
    salience_mult = 0.5 + float(salience)         # 0.5x .. 1.5x
    pinned_mult = 1.5 if pinned else 1.0           # +50% for pinned

    new_score = base_score * salience_mult * decay * pinned_mult

    reasons: list[str] = []
    if pinned:
        reasons.append("pinned")
    if salience > 0.7:
        reasons.append(f"high_salience:{salience:.2f}")
    if age_days > 0:
        reasons.append(f"decay:{decay:.2f}x")  # e.g. "decay:0.83x"
    return new_score, reasons


# ── entity boost ────────────────────────────────────────────────────────────


def entity_boost(
    base_score: float,
    memory_entity_ids: set[str] | list[str] | None,
    query_entity_ids: set[str] | list[str] | None,
    boost_per_match: float = 0.3,
    max_boost: float = 1.0,
) -> tuple[float, list[str]]:
    """
    Boost a score by the number of shared entities between memory and query.

    Boost is multiplicative: ``score * (1 + min(n_matches * boost_per, max_boost))``.

    Default: 1 match → +0.3, 3 matches → +0.9, 5+ → +1.0 (capped).

    Returns ``(new_score, match_reasons)`` where each reason is
    ``"entity:<name>"`` for every matched entity.
    """
    if not query_entity_ids or not memory_entity_ids:
        return base_score, []

    mem_set = {str(e).lower() for e in memory_entity_ids}
    qry_set = {str(e).lower() for e in query_entity_ids}
    matches = mem_set & qry_set

    if not matches:
        return base_score, []

    boost = min(len(matches) * boost_per_match, max_boost)
    new_score = base_score * (1.0 + boost)
    reasons = [f"entity:{name}" for name in sorted(matches)]
    return new_score, reasons


# ── combined helper ─────────────────────────────────────────────────────────


def rerank(
    base_score: float,
    *,
    captured_at: datetime,
    salience: float,
    pinned: bool,
    memory_entity_ids: set[str] | list[str] | None = None,
    query_entity_ids: set[str] | list[str] | None = None,
    half_life_days: float = 30.0,
    entity_boost_per_match: float = 0.3,
    entity_boost_max: float = 1.0,
    now: datetime | None = None,
) -> tuple[float, list[str]]:
    """
    Apply entity-boost first, then time-decay. Returns final score + reasons.
    """
    score, entity_reasons = entity_boost(
        base_score,
        memory_entity_ids,
        query_entity_ids,
        boost_per_match=entity_boost_per_match,
        max_boost=entity_boost_max,
    )
    score, decay_reasons = time_decay_score(
        score,
        captured_at=captured_at,
        salience=salience,
        pinned=pinned,
        now=now,
        half_life_days=half_life_days,
    )
    return score, entity_reasons + decay_reasons
