import logging

import httpx

from app.config import settings

log = logging.getLogger(__name__)
JINA_URL = "https://api.jina.ai/v1/rerank"


class RerankUnavailable(RuntimeError):
    """Transport failure: connect/read error, timeout, or a non-2xx status.

    Retryable and never fatal — the caller falls back to dense order and
    counts ``retrieval.rerank_failed`` (ruling R11(p2)).
    """


class RerankInvalidResponse(RuntimeError):
    """A 2xx whose body is not a usable rerank response: no ``results`` list,
    an EMPTY one, or no usable row in it (ruling R14(p2)). Per-ROW damage is
    skipped, never typed: one bad ``index``/``relevance_score`` must not throw
    the whole pool away."""


_client: httpx.AsyncClient | None = None

def get_jina_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=30.0)
    return _client


def _backend() -> str:
    """Resolve ``RERANK_BACKEND`` for this call: "auto" (default) is the bundled
    local ONNX cross-encoder — the $0 self-host path is the default, so an
    opt-in rerank never reaches for the paid API unless it is asked to. "jina"
    pins the paid HTTP lane (the model the frozen benchmark was reranked
    with), "local" pins the bundled one. The setting is validated at load, so
    anything that arrives here is one of the three spells."""
    backend = (settings.RERANK_BACKEND or "auto").strip().lower()
    return "local" if backend == "auto" else backend


def _finalize(
    rows: list[tuple[int, float]],
    chunks: list[dict],
    limit: int,
    *,
    transport: str,
    raw_count: int,
) -> list[dict]:
    """Dedup by memory, stamp ``rerank_score``, best first.

    ``raw_count`` is how many rows the transport answered with, so an empty
    answer and an all-unusable one stay distinguishable in the raised message.
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
        # Ruling R14(p2): an EMPTY ``results`` list is the same failure as rows
        # that are all unusable — the transport answered and left nothing to
        # rank on. Returning [] here would hide it as "nothing to rerank"
        # (dense order, uncounted); as an invalid response it is counted and
        # dense order continues all the same.
        detail = f"all {raw_count} rows were unusable" if raw_count else "an empty 'results' list"
        raise RerankInvalidResponse(f"{transport}: {detail}")

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    # The cap is the transport's ANSWER size. Jina enforces it server-side; this
    # lane scores the whole pool, so the cut has to happen here — same place for
    # both, so "at most JINA_RERANKER_TOP_N rows" holds whichever one ran.
    reranked = reranked[:limit]
    log.info(
        "Reranked",
        extra={"in": len(chunks), "out": len(reranked), "top_n": limit, "transport": transport},
    )
    return reranked


async def _local_rerank(query: str, chunks: list[dict], limit: int) -> list[dict]:
    """The local ONNX lane: score every chunk, then the shared finalize.

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
    # rule lands on the same chunk the Jina lane would have kept (Jina answers
    # pre-sorted; this lane's rows arrive in pool order).
    parsed = sorted(enumerate(scores), key=lambda row: row[1], reverse=True)
    return _finalize(
        parsed, chunks, limit, transport="Local rerank", raw_count=len(chunks)
    )


async def rerank(query: str, chunks: list[dict], *, top_n: int | None = None) -> list[dict]:
    """Score ``chunks`` against ``query`` and return the best rows, best first.

    ``top_n`` is PER CALL — the request's own top_k — and the deployment's
    ``JINA_RERANKER_TOP_N`` only CAPS it (ruling R4(p2); the global is no
    longer the value). Fewer rows may come back than were handed in: the
    caller merges the rest back in dense order, so the served result count
    never depends on this transport.

    Which transport runs is ``RERANK_BACKEND`` (``_backend``): the bundled
    local ONNX cross-encoder by default, the paid Jina HTTP lane when pinned.

    Raises :class:`RerankUnavailable` (transport/status/timeout — bounded by
    ``JINA_RERANKER_TIMEOUT_SECONDS`` — and every local model-side failure) or
    :class:`RerankInvalidResponse` (unusable body: no ``results`` list, an
    empty one, or nothing usable in it). A malformed ROW is skipped instead of
    killing the pool (ruling R11(p2) — one bad ``index`` used to raise and drop
    every row).
    """
    if not chunks:
        return []

    limit = settings.JINA_RERANKER_TOP_N
    if top_n is not None:
        limit = min(int(top_n), limit)

    if _backend() == "local":
        return await _local_rerank(query, chunks, limit)

    client = get_jina_client()
    try:
        resp = await client.post(
            JINA_URL,
            json={
                "model":     settings.JINA_RERANKER_MODEL,
                "query":     query,
                "documents": [c["content"] for c in chunks],
                "top_n":     limit,
            },
            headers={
                "Authorization": f"Bearer {settings.JINA_API_KEY}",
                "Content-Type":  "application/json",
            },
            timeout=settings.JINA_RERANKER_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
    except httpx.HTTPError as e:
        # HTTPStatusError is a sibling of TransportError under HTTPError, and
        # TimeoutException is under TransportError: one clause types them all.
        raise RerankUnavailable(
            f"Jina rerank transport error: {type(e).__name__}: {e}"
        ) from e

    try:
        data = resp.json()
    except ValueError as e:
        raise RerankInvalidResponse(f"Jina rerank returned non-JSON: {e}") from e
    rows = data.get("results") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise RerankInvalidResponse("Jina rerank response has no usable 'results' list")

    parsed: list[tuple[int, float]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        index, score = item.get("index"), item.get("relevance_score")
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(chunks):
            log.warning("Skipping rerank row with an unusable index", extra={"index": index})
            continue
        try:
            score = float(score)
        except (TypeError, ValueError):
            log.warning("Skipping rerank row without a numeric score", extra={"index": index})
            continue
        parsed.append((index, score))

    return _finalize(parsed, chunks, limit, transport="Jina rerank", raw_count=len(rows))
