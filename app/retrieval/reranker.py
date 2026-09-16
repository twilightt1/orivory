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

async def rerank(query: str, chunks: list[dict], *, top_n: int | None = None) -> list[dict]:
    """Score ``chunks`` against ``query`` and return the best rows, best first.

    ``top_n`` is PER CALL — the request's own top_k — and the deployment's
    ``JINA_RERANKER_TOP_N`` only CAPS it (ruling R4(p2); the global is no
    longer the value). Fewer rows may come back than were handed in: the
    caller merges the rest back in dense order, so the served result count
    never depends on this transport.

    Raises :class:`RerankUnavailable` (transport/status/timeout — bounded by
    ``JINA_RERANKER_TIMEOUT_SECONDS``) or :class:`RerankInvalidResponse`
    (unusable body: no ``results`` list, an empty one, or nothing usable in
    it). A malformed ROW is skipped instead of killing the pool (ruling
    R11(p2) — one bad ``index`` used to raise and drop every row).
    """
    if not chunks:
        return []

    limit = settings.JINA_RERANKER_TOP_N
    if top_n is not None:
        limit = min(int(top_n), limit)

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

    reranked = []
    seen: set = set()
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
        detail = f"all {len(rows)} rows were unusable" if rows else "an empty 'results' list"
        raise RerankInvalidResponse(f"Jina rerank: {detail}")

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    log.info("Reranked", extra={"in": len(chunks), "out": len(reranked), "top_n": limit})
    return reranked
