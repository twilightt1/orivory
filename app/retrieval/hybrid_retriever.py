import hashlib
from collections import defaultdict


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Fuse ranked result lists with Reciprocal Rank Fusion.

    Document-path reference; no shipped caller in this release, and kept as the
    RRF reference: it dedupes by ``parent_id`` or an md5 of the content. That
    rule is FORBIDDEN for memory retrieval (§7.4 / ruling R11b(p2)) — two
    memory facts can carry the same text and are distinct — which is why
    memory recall fuses through :func:`fuse_by_uuid` instead of this function.

    Each fused doc carries a stable ``id`` (parent id, or a content hash when
    the retriever provides no parent) so downstream consumers (CRAG grading,
    dedupe) can reference chunks without a KeyError. When several children of
    the same parent appear, the best-ranked child (lowest rank across all
    lists) is kept as the representative.
    """
    scores:      dict[str, float] = defaultdict(float)
    content_map: dict[str, dict]  = {}
    best_rank:   dict[str, int]   = {}
    for results in result_lists:
        for rank, item in enumerate(results):

            doc_id = item.get("parent_id") or item.get("metadata", {}).get("parent_id")
            if not doc_id:
                doc_id = hashlib.md5(item["content"].encode()).hexdigest()
            scores[doc_id] += 1.0 / (k + rank + 1)
            if doc_id not in best_rank or rank < best_rank[doc_id]:
                best_rank[doc_id] = rank
                content_map[doc_id] = item
    return [
        {**content_map[d], "id": d, "rrf_score": scores[d]}
        for d in sorted(scores, key=scores.get, reverse=True)
    ]


def fuse_by_uuid(
    dense: list[dict],
    lexical: list[dict],
    *,
    k: int = 60,
) -> list[dict]:
    """Fuse the dense and lexical memory legs by Reciprocal Rank Fusion.

    Memory identity is the canonical Memory UUID (§7.4 / ruling R11b(p2)):
    NEVER dedupe by content or ``parent_id`` — identical text with two UUIDs
    is two memories, and a fused row must stay addressable by the id the SQL
    authorization later hydrates.

    Each leg is a best-first list; ranks are ZERO-BASED per list and a row's
    fused score is ``sum(1 / (k + rank + 1))`` over the legs that returned it,
    with ``k = RETRIEVAL_RRF_K`` when called from recall. Only that sum is
    cross-leg comparable: the legs' own scores are kept apart on the candidate
    as ``dense_score`` / ``lexical_score`` (``None`` when that leg did not
    return the row) and are never fused, compared or treated as tenant-local —
    the lexical one is raw global BM25.

    The fused ``score`` IS the RRF sum, so the existing score chain (entity
    boost + time decay) keeps operating on a single positive scale. Passing an
    empty list for a leg is the single-leg case (the vector-outage fallback):
    the fused score is that leg's own rank score. A row repeated within one
    leg votes once — a drifted FTS index can repeat a ``memory_id`` — and ties
    break on ``memory_id`` so a fused order is deterministic.

    Malformed rows are SKIPPED, never fatal (the recall path's own posture: a
    candidate without a usable ``memory_id`` is dropped with a debug log, and
    a non-numeric ``score`` leaves that leg's score ``None``). Recall tears
    the pool down on an unexpected exception — and inside the vector-outage
    handler a raise here would escape as a 500 where the contract says typed
    503 — so one bad row in a leg must never take the pool with it.
    """
    scores: dict[str, float] = defaultdict(float)
    rows: dict[str, dict] = {}
    for leg, score_key in ((dense, "dense_score"), (lexical, "lexical_score")):
        seen: set[str] = set()
        for rank, item in enumerate(leg):
            if not isinstance(item, dict):
                continue
            memory_id = str(item.get("memory_id") or "")
            if not memory_id or memory_id in seen:
                continue
            seen.add(memory_id)
            scores[memory_id] += 1.0 / (k + rank + 1)
            row = rows.setdefault(
                memory_id,
                {**item, "memory_id": memory_id,
                 "dense_score": None, "lexical_score": None},
            )
            # The leg's own score is informational (fusion is by rank); an
            # unusable one is None, exactly like a leg that did not return the
            # row, and never a reason to drop a ranked row.
            raw_score = item.get("score")
            try:
                row[score_key] = None if raw_score is None else float(raw_score)
            except (TypeError, ValueError):
                row[score_key] = None
    return [
        {**rows[memory_id], "score": scores[memory_id]}
        for memory_id in sorted(scores, key=lambda mid: (-scores[mid], mid))
    ]
