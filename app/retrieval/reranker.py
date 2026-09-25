import logging

from app.config import settings

log = logging.getLogger(__name__)


class RerankUnavailable(RuntimeError):
    """The scorer could not answer: model files missing or corrupt, a broken
    ONNX session, an OOM.

    Retryable and never fatal — the caller falls back to dense order and
    counts ``retrieval.rerank_failed`` (ruling R11(p2)).
    """


class RerankInvalidResponse(RuntimeError):
    """The scorer answered and left nothing to rank on (ruling R14(p2)).

    Per-ROW damage is skipped, never typed: one unusable score must not throw
    the whole pool away.
    """


def _finalize(
    rows: list[tuple[int, float]],
    chunks: list[dict],
    limit: int,
    *,
    transport: str,
    raw_count: int,
) -> list[dict]:
    """Dedup by memory, stamp ``rerank_score``, best first, then cut to ``limit``.

    ``raw_count`` is how many rows the scorer answered with, so an empty
    answer and an all-unusable one stay distinguishable in the raised message.

    The cap is the scorer's ANSWER size. The lane scores the whole pool, so the
    cut happens here: "at most ``RERANK_TOP_N`` rows out" holds for every
    caller, and the remainder stays with the caller's dense merge.
    """
    reranked = []
    seen: set = set()
    for index, score in rows:
        original = chunks[index]
        key = original.get("memory_id", index)
        if key in seen:
            # One memory, one row (M1): a row repeating an already-ranked index
            # would return the same memory twice and occupy a head slot that a
            # distinct row should have had. The FIRST row for a memory wins.
            continue
        seen.add(key)
        original = original.copy()
        # A 0.0 relevance IS a score: consumers read it by key presence
        # (retriever.py), never by truthiness.
        original["rerank_score"] = score
        reranked.append(original)

    if not reranked:
        # Ruling R14(p2): scoring nothing at all is the same failure as rows
        # that are all unusable — returning [] here would hide it as "nothing
        # to rerank" (dense order, uncounted); as an invalid response it is
        # counted and dense order continues all the same.
        detail = f"all {raw_count} rows were unusable" if raw_count else "no rows at all"
        raise RerankInvalidResponse(f"{transport}: {detail}")

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    reranked = reranked[:limit]
    log.info(
        "Reranked",
        extra={"in": len(chunks), "out": len(reranked), "top_n": limit, "transport": transport},
    )
    return reranked


async def _local_rerank(query: str, chunks: list[dict], limit: int) -> list[dict]:
    """The bundled ONNX lane: score every chunk, then the shared finalize.

    Any model-side failure (missing files, a broken session, an OOM) is typed
    as :class:`RerankUnavailable` so the caller's existing fallback — dense
    order, ``retrieval.rerank_failed`` counted — covers it unchanged.
    """
    from app.retrieval import local_reranker

    try:
        scores = await local_reranker.score_pairs(query, [c["content"] for c in chunks])
    except Exception as e:
        raise RerankUnavailable(f"Local rerank failed: {type(e).__name__}: {e}") from e
    # Sorted before the shared finalize so its "first row for a memory wins"
    # rule lands on the highest-scored chunk (this lane's rows arrive in pool
    # order, and one memory can hold several chunks).
    parsed = sorted(enumerate(scores), key=lambda row: row[1], reverse=True)
    return _finalize(
        parsed, chunks, limit, transport="Local rerank", raw_count=len(chunks)
    )


async def rerank(query: str, chunks: list[dict], *, top_n: int | None = None) -> list[dict]:
    """Score ``chunks`` against ``query`` and return the best rows, best first.

    The lane is the bundled ONNX cross-encoder
    (:mod:`app.retrieval.local_reranker`): torch-free, no API key, no per-call
    cost, nothing leaves the box. Opt-in through ``RETRIEVAL_SEMANTIC_RERANK``.

    ``top_n`` is PER CALL — the request's own top_k — and the deployment's
    ``RERANK_TOP_N`` only CAPS it (ruling R4(p2); the global is no longer the
    value). Fewer rows may come back than were handed in: the caller merges the
    rest back in dense order, so the served result count never depends on this
    lane.

    Raises :class:`RerankUnavailable` (every model-side failure) or
    :class:`RerankInvalidResponse` (nothing usable came back).
    """
    if not chunks:
        return []

    limit = settings.RERANK_TOP_N
    if top_n is not None:
        limit = min(int(top_n), limit)

    return await _local_rerank(query, chunks, limit)
