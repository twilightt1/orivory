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
import re
from datetime import UTC, datetime

from app.retrieval.memory.correction import _stored_key, state_of

# ── lexical exact-match refinement ──────────────────────────────────────────

_QUOTED_RE = re.compile(r'"([^"]+)"')
_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ỹ]+(?:[/\-.][0-9A-Za-zÀ-ỹ]+)*")

#: Additive score nudge for verbatim overlap (capped — never dominates semantics).
LEXICAL_BONUS = 0.15


def lexical_bonus(query: str, text: str) -> tuple[float, list[str]]:
    """Reward verbatim overlap between query terms and candidate text.

    Terms = quoted phrases + capitalized tokens + numbers/dates (stdlib
    ``re`` only). Match is case-insensitive containment. Returns
    ``(0.15, reasons)`` on any hit else ``(0.0, [])`` — one reason per
    matched term (``"lexical:<term>"``).
    """
    if not query or not text:
        return 0.0, []
    terms: list[str] = []
    for m in _QUOTED_RE.findall(query):
        if m.strip():
            terms.append(m.strip())
    for tok in _TOKEN_RE.findall(query):
        if any(ch.isdigit() for ch in tok):
            terms.append(tok)  # numbers/dates
        elif tok[0].isupper() and len(tok) > 1:
            terms.append(tok)  # capitalized tokens
    seen: set[str] = set()
    terms = [t for t in terms if not (t.lower() in seen or seen.add(t.lower()))]
    tl = text.lower()
    matched = [t for t in terms if t.lower() in tl]
    if not matched:
        return 0.0, []
    return LEXICAL_BONUS, [f"lexical:{m}" for m in matched]

# ── query routing ─────────────────────────────────────────────────────────────

#: Time/summary intent → favor recency over entity match.
_ROUTE_TIME_RE = re.compile(
    r"when|giai đoạn|khi nào|tóm tắt|summar|timeline|history|quá trình",
    re.IGNORECASE,
)


def route_query(query: str, entities) -> str:
    """Route a query to ``'local'`` | ``'relational'`` | ``'general'``.

    ``'local'`` wins on time/summary words (recency matters most);
    ``'relational'`` when entities are present or ≥2 distinct capitalized
    tokens appear; otherwise ``'general'`` (no weight adjustment).
    ``entities`` accepts the rewrite result (list of ``{name, type}``),
    plain names, or None.
    """
    if query and _ROUTE_TIME_RE.search(query):
        return "local"
    if entities:
        return "relational"
    caps = {t for t in _TOKEN_RE.findall(query or "") if t[0].isupper() and len(t) > 1}
    if len(caps) >= 2:
        return "relational"
    return "general"


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


# ── same-slot evidence closure ────────────────────────────────────────────


def apply_closure(scored, top_k, cap=2):
    """Pull same-slot evidence mates from the pool tail into the top-k.

    ``scored`` is ``[(memory, score, reasons)]`` sorted desc; ``top_k``
    counts how many lead entries own the slots. Up to ``cap`` mates from
    the rest sharing a lead slot replace the lowest top-k entries, each
    gaining a ``'closure:slot'`` reason. Unslotted memories (empty
    subject/attribute) never match; superseded/dirty mates are skipped.
    Input tuples are not mutated. ``cap <= 0`` returns the slice unchanged.
    # ponytail: O(n) scan; fine at rerank-pool sizes (≤ ~30).
    """
    if top_k <= 0:
        return []
    top = [(m, s, list(r)) for m, s, r in scored[:top_k]]
    if cap <= 0 or len(scored) <= len(top):
        return top
    slots = set()
    for m, _, _ in top:
        key = _stored_key(m)
        if key[0] and key[1]:
            slots.add(key)
    if not slots:
        return top
    out = list(top)
    swaps = 0
    for m, s, r in scored[len(top):]:
        if swaps >= cap or swaps >= len(out):
            break
        key = _stored_key(m)
        if not (key[0] and key[1]) or key not in slots:
            continue
        if state_of(m) != "current":
            continue
        out[len(out) - 1 - swaps] = (m, s, [*r, "closure:slot"])
        swaps += 1
    return out
