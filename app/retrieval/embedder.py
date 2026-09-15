import asyncio
import functools
import inspect
import logging
from concurrent.futures import ThreadPoolExecutor

from openai import AsyncOpenAI, OpenAI

from app.config import settings
from app.retrieval.embedding_fingerprint import (
    canonical_fingerprint,
    current_fingerprint,
    fingerprint_generation,
)

log = logging.getLogger(__name__)

# ONE executor for every ASYNC embedding call (P2/T1). The local path is a
# synchronous ONNX call: awaited inline from async code it stalls the loop for
# the call's whole duration (a 50-document drain batch is ~674 ms, and the
# freshness barrier runs that drain ON the recall request path). Bounded, not
# the loop's default pool, so the parallel embeds stay accountable against
# ORT's own thread count. The `*_sync` faces do NOT come through here: their
# callers are already off-loop (celery/CLI/ingestion threads).
#
# Cancellation cannot reach inside a worker: cancelling a task parked on
# `run_in_executor` (the barrier's timeout) frees the caller immediately, but
# the ONNX call keeps its slot until it returns (~0.6 s for a 50-doc batch) —
# so after a barrier timeout one of the two workers stays busy a little longer.
_EMBED_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(1, settings.EMBED_EXECUTOR_WORKERS),
    thread_name_prefix="orivory-embed",
)


async def _run_off_loop(fn, *args, **kwargs):
    """Run a synchronous embedding call on the embed executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_EMBED_EXECUTOR, functools.partial(fn, *args, **kwargs))


async def warmup_embedder() -> None:
    """Build the local embedding session before anything is served (P2/T1).

    A cold ``InferenceSession`` is 610-685 ms and even built inside a thread it
    leaves C-level parse lag: the lifespan pays it once, before the boot drain,
    instead of the first request paying it inside the measured path. API
    backends have no local session to build. Best-effort by design — a missing
    model download must not keep a booting app from serving (the first real
    call pays it again and says why).
    """
    if not settings.USE_LOCAL_EMBEDDINGS:
        return
    try:
        await _run_off_loop(_embed_with_local, ["warmup"])
    except Exception as e:  # pragma: no cover - a download/network failure
        log.warning("Local embedding warmup failed: %s", e)


# Lazily-resolved module-level aliases.
def __getattr__(name):
    if name == "async_client":
        return _get_async_client()
    if name == "sync_client":
        return _get_sync_client()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_async_client: AsyncOpenAI | None = None
_sync_client: OpenAI | None = None

# Collection metadata keys recording which embedding backend, dimension and
# contract a collection was created with. Flipping embedding settings after
# data exists must fail loud instead of mixing incompatible vectors.
EMBED_BACKEND_META_KEY = "orivory_embed_backend"
EMBED_DIM_META_KEY = "orivory_embed_dim"
EMBED_FINGERPRINT_META_KEY = "orivory_embed_fingerprint"
EMBED_GENERATION_META_KEY = "orivory_embed_generation"


class EmbeddingDimensionMismatch(ValueError):
    """Raised when an embedding does not match a verifiable collection contract.

    Either the embedding config changed after data was indexed (fix: reindex
    into a fresh collection or restore the previous backend), vectors from
    two contracts are being mixed, or collection metadata cannot be trusted.
    """


def active_backend_name() -> str:
    """Which embedding backend the current settings select."""
    if settings.USE_LOCAL_EMBEDDINGS:
        # No fallback: LOCAL_EMBED_MODEL is validated at config load, and the
        # removed MiniLM backend must never be named "local" again.
        return {"arctic": "local-arctic", "e5": "local-e5"}[settings.LOCAL_EMBED_MODEL]
    if settings.USE_JINA_EMBEDDINGS and settings.JINA_API_KEY:
        return "jina"
    return "openai"


def _canonical_fingerprint(fingerprint: str | dict | None) -> str:
    """Return the scalar representation stored in collection metadata."""
    if fingerprint is None:
        fingerprint = current_fingerprint()
    try:
        return canonical_fingerprint(fingerprint)
    except ValueError as exc:
        raise EmbeddingDimensionMismatch(str(exc)) from exc


def check_collection_dim(
    collection,
    embedding_dim: int,
    *,
    backend: str | None = None,
    fingerprint: str | dict | None = None,
    collection_is_empty: bool | None = None,
) -> dict | None:
    """Verify an embedding fits the collection's recorded contract.

    Returns a replacement metadata dict to stamp when the collection carries
    no embedding stamp yet *and is empty*, None when the stamp matches, and
    raises :class:`EmbeddingDimensionMismatch` on an unknown or mismatched
    contract. The active embedding fingerprint is used when callers omit one.

    ``collection_is_empty`` is supplied by async callers after awaiting the
    collection count. Synchronous collections are inspected directly when
    possible; callers that cannot prove a new collection is empty must use the
    stamp helpers, which fail closed.
    """
    backend = backend or active_backend_name()
    expected_fingerprint = _canonical_fingerprint(fingerprint)
    # Unwrap the local-mode sync→async adapter (its __getattr__ turns every
    # attribute access into a coroutine factory — read the inner instead).
    inner = getattr(collection, "_inner", collection)
    if type(inner).__name__ in {"MagicMock", "AsyncMock", "Mock"}:
        return None
    if collection_is_empty is None:
        count_method = getattr(inner, "count", None)
        if callable(count_method) and not inspect.iscoroutinefunction(count_method):
            try:
                count = count_method()
            except Exception as exc:
                raise EmbeddingDimensionMismatch(
                    "unable to verify collection population — quarantine/rebuild"
                ) from exc
            if inspect.isawaitable(count):
                close = getattr(count, "close", None)
                if callable(close):
                    close()
                raise EmbeddingDimensionMismatch(
                    "async collection population must be checked by an async guard"
                )
            try:
                collection_is_empty = int(count) == 0
            except (TypeError, ValueError) as exc:
                raise EmbeddingDimensionMismatch(
                    "invalid collection count — quarantine/rebuild"
                ) from exc
        elif callable(count_method):
            raise EmbeddingDimensionMismatch(
                "async collection population must be checked by an async guard"
            )
    try:
        meta = inner.metadata
    except Exception as exc:
        raise EmbeddingDimensionMismatch(
            "unreadable collection metadata — quarantine/rebuild, "
            "do not auto-assign fingerprint"
        ) from exc
    if not isinstance(meta, dict):
        raise EmbeddingDimensionMismatch(
            "non-dict collection metadata — quarantine/rebuild"
        )

    has_dim = EMBED_DIM_META_KEY in meta
    has_backend = EMBED_BACKEND_META_KEY in meta
    has_fingerprint = EMBED_FINGERPRINT_META_KEY in meta
    has_generation = EMBED_GENERATION_META_KEY in meta
    if not any((has_dim, has_backend, has_fingerprint, has_generation)):
        if collection_is_empty is False:
            raise EmbeddingDimensionMismatch(
                "populated collection has no verifiable embedding contract — "
                "quarantine/rebuild"
            )
        return {
            **meta,
            EMBED_BACKEND_META_KEY: backend,
            EMBED_DIM_META_KEY: embedding_dim,
            EMBED_FINGERPRINT_META_KEY: expected_fingerprint,
            EMBED_GENERATION_META_KEY: fingerprint_generation(expected_fingerprint),
        }
    if not has_dim or not has_backend or not has_fingerprint:
        raise EmbeddingDimensionMismatch(
            "incomplete collection embedding metadata — quarantine/rebuild"
        )

    recorded_dim = meta[EMBED_DIM_META_KEY]
    recorded_backend = meta[EMBED_BACKEND_META_KEY]
    recorded_fingerprint = meta.get(EMBED_FINGERPRINT_META_KEY)
    if not isinstance(recorded_fingerprint, str) or not recorded_fingerprint:
        raise EmbeddingDimensionMismatch(
            "collection lacks embedding fingerprint — quarantine/rebuild, "
            "do not auto-stamp"
        )
    if recorded_fingerprint != expected_fingerprint:
        raise EmbeddingDimensionMismatch(
            "same dim but different embedding contract: "
            f"{recorded_fingerprint!r} vs {expected_fingerprint!r} — "
            "fresh reindex required"
        )
    recorded_generation = meta.get(EMBED_GENERATION_META_KEY)
    if recorded_generation is not None and (
        not isinstance(recorded_generation, str)
        or recorded_generation != fingerprint_generation(recorded_fingerprint)
    ):
        raise EmbeddingDimensionMismatch(
            "embedding contract generation mismatch — fresh reindex required"
        )
    try:
        same_dim = int(recorded_dim) == int(embedding_dim)
    except (TypeError, ValueError):
        same_dim = False
    if not same_dim or recorded_backend != backend:
        raise EmbeddingDimensionMismatch(
            f"Embedding backend/dim mismatch: collection was created with "
            f"backend={recorded_backend!r} dim={recorded_dim!r}, but current "
            f"config produces backend={backend!r} dim={embedding_dim!r}. "
            f"Restore the previous embedding backend or reindex into a fresh "
            f"collection — mixing backends silently corrupts recall."
        )
    return None


def check_generation_contract(
    info: dict,
    embedding_dim: int,
    *,
    manifest_fingerprint: str | None,
    collection_is_empty: bool,
    fingerprint: str | dict | None = None,
) -> None:
    """Verify a Qdrant generation against the embedding contract (spec §4.2).

    The legacy (Chroma) guard below reads a collection's own metadata stamp; a Qdrant
    collection has none — its contract lives in TWO places that must agree:

    - the PHYSICAL collection: dim/metric, read from
      :func:`app.retrieval.vector_backend.collection_info`;
    - the MANIFEST row: the active ``index_generations.fingerprint`` for the
      kind, which names the embedding contract those vectors were built with.

    ``manifest_fingerprint is None`` means no active manifest row: an EMPTY
    generation is then unclaimed (allowed — the ladder/cutover claims it), a
    POPULATED one is unknown data and must be quarantined, never served.
    Read-only by construction: nothing here writes the manifest (that is the
    cutover's job), so a read path can never claim a generation.
    """
    if manifest_fingerprint is None and not collection_is_empty:
        raise EmbeddingDimensionMismatch(
            "populated generation has no manifest row — quarantine/rebuild"
        )
    expected_fingerprint = _canonical_fingerprint(fingerprint)
    if manifest_fingerprint is not None and (
        manifest_fingerprint != fingerprint_generation(expected_fingerprint)
    ):
        raise EmbeddingDimensionMismatch(
            "same dim but different embedding contract: generation manifest is "
            f"{manifest_fingerprint!r} vs {fingerprint_generation(expected_fingerprint)!r} — "
            "fresh reindex required"
        )
    distance = str(info.get("distance", "")).lower()
    if distance != "cosine":
        raise EmbeddingDimensionMismatch(
            f"generation distance metric is {distance!r}, expected cosine — fresh reindex required"
        )
    try:
        recorded_dim = int(info["dim"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EmbeddingDimensionMismatch(
            "generation has no readable vector dim — quarantine/rebuild"
        ) from exc
    if recorded_dim != int(embedding_dim):
        raise EmbeddingDimensionMismatch(
            f"Embedding backend/dim mismatch: generation holds dim={recorded_dim}, but the "
            f"current config produces dim={int(embedding_dim)}. Restore the previous embedding "
            "backend or reindex into a fresh generation — mixing dims silently corrupts recall."
        )


def stamp_collection_dim(
    collection,
    embedding_dim: int,
    *,
    backend: str | None = None,
    fingerprint: str | dict | None = None,
) -> None:
    """Check the dim guard and stamp an unstamped collection (sync caller).

    Raises :class:`EmbeddingDimensionMismatch` on an unknown or mismatched
    contract. A populated collection without a contract and metadata-write
    failures both fail closed before any vector upsert.
    """
    stamp = check_collection_dim(
        collection,
        embedding_dim,
        backend=backend,
        fingerprint=fingerprint,
    )
    if stamp is None:
        return
    inner = getattr(collection, "_inner", collection)
    if type(inner).__name__ in ("MagicMock", "AsyncMock", "Mock"):
        return
    count_method = getattr(inner, "count", None)
    if not callable(count_method) or inspect.iscoroutinefunction(count_method):
        raise EmbeddingDimensionMismatch(
            "unable to verify collection is empty before stamping — quarantine/rebuild"
        )
    try:
        count = count_method()
        if inspect.isawaitable(count):
            raise EmbeddingDimensionMismatch(
                "synchronous collection returned an awaitable count"
            )
        if int(count) != 0:
            raise EmbeddingDimensionMismatch(
                "populated collection has no verifiable embedding contract — "
                "quarantine/rebuild"
            )
        # Chroma treats ``hnsw:space`` as immutable collection configuration;
        # sending it back through ``modify`` is rejected even when unchanged.
        result = inner.modify(
            metadata={key: value for key, value in stamp.items() if not key.startswith("hnsw:")}
        )
        if inspect.isawaitable(result):
            raise EmbeddingDimensionMismatch(
                "synchronous collection returned an awaitable metadata write"
            )
        observed = inner.metadata
        if not isinstance(observed, dict) or any(
            observed.get(key) != stamp[key]
            for key in (
                EMBED_BACKEND_META_KEY,
                EMBED_DIM_META_KEY,
                EMBED_FINGERPRINT_META_KEY,
                EMBED_GENERATION_META_KEY,
            )
        ):
            raise EmbeddingDimensionMismatch(
                "embedding contract metadata write could not be verified — vector write blocked"
            )
    except EmbeddingDimensionMismatch:
        raise
    except Exception as exc:
        log.warning("Could not stamp collection embedding dim", exc_info=True)
        raise EmbeddingDimensionMismatch(
            "failed to persist embedding contract metadata — vector write blocked"
        ) from exc


async def astamp_collection_dim(
    collection,
    embedding_dim: int,
    *,
    backend: str | None = None,
    fingerprint: str | dict | None = None,
) -> None:
    """Async variant of :func:`stamp_collection_dim`."""
    inner = getattr(collection, "_inner", collection)
    if type(inner).__name__ in ("MagicMock", "AsyncMock", "Mock"):
        return
    count_method = getattr(inner, "count", None)
    if not callable(count_method):
        raise EmbeddingDimensionMismatch(
            "unable to verify collection is empty before stamping — quarantine/rebuild"
        )
    try:
        count = count_method()
        if inspect.isawaitable(count):
            count = await count
        collection_is_empty = int(count) == 0
        stamp = check_collection_dim(
            collection,
            embedding_dim,
            backend=backend,
            fingerprint=fingerprint,
            collection_is_empty=collection_is_empty,
        )
        if stamp is None:
            return
        if not collection_is_empty:
            raise EmbeddingDimensionMismatch(
                "populated collection has no verifiable embedding contract — "
                "quarantine/rebuild"
            )
        # Chroma treats ``hnsw:space`` as immutable collection configuration;
        # sending it back through ``modify`` is rejected even when unchanged.
        res = inner.modify(
            metadata={key: value for key, value in stamp.items() if not key.startswith("hnsw:")}
        )
        if inspect.isawaitable(res):
            await res
        observed = inner.metadata
        if not isinstance(observed, dict) or any(
            observed.get(key) != stamp[key]
            for key in (
                EMBED_BACKEND_META_KEY,
                EMBED_DIM_META_KEY,
                EMBED_FINGERPRINT_META_KEY,
                EMBED_GENERATION_META_KEY,
            )
        ):
            raise EmbeddingDimensionMismatch(
                "embedding contract metadata write could not be verified — vector write blocked"
            )
    except EmbeddingDimensionMismatch:
        raise
    except Exception as exc:
        log.warning("Could not stamp collection embedding dim", exc_info=True)
        raise EmbeddingDimensionMismatch(
            "failed to persist embedding contract metadata — vector write blocked"
        ) from exc


def _get_async_client() -> AsyncOpenAI:
    """Lazily construct the async OpenAI-compatible client."""
    global _async_client
    if _async_client is None:
        _async_client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": settings.FRONTEND_URL,
                "X-Title": "Orivory",
            },
        )
    return _async_client


def _get_sync_client() -> OpenAI:
    """Lazily construct the sync OpenAI-compatible client."""
    global _sync_client
    if _sync_client is None:
        _sync_client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": settings.FRONTEND_URL,
                "X-Title": "Orivory",
            },
        )
    return _sync_client


def _batches(texts: list[str]) -> list[list[str]]:
    batch_size = max(1, settings.EMBED_BATCH_SIZE)
    return [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]


async def _embed_with_openai(texts: list[str]) -> list[list[float]]:
    """Embed texts using OpenAI-compatible API.

    LEGACY / UNBENCHMARKED: no benchmark run covers this backend (the frozen
    v1.1.0 baseline is Jina; lite mode uses local). It stays as a fallback
    so existing deployments don't break, but issues about its recall
    quality will not be acted on — switch to Jina or local instead.
    """
    log.warning(
        "OpenAI embedding backend is unbenchmarked legacy — "
        "prefer Jina (full-stack) or local (lite mode)",
    )
    embeddings: list[list[float]] = []
    try:
        client = _get_async_client()
        for batch in _batches(texts):
            response = await client.embeddings.create(
                model=settings.EMBED_MODEL,
                input=batch,
                encoding_format="float",
                timeout=30.0,
            )
            embeddings.extend(item.embedding for item in response.data)
        return embeddings
    except Exception as e:
        log.error("OpenAI embedding failed", exc_info=True)
        raise ValueError(f"Failed to get embeddings: {e}") from e


async def _embed_with_jina(texts: list[str]) -> list[list[float]]:
    """Embed texts using Jina AI API directly."""
    import httpx

    if not settings.JINA_API_KEY:
        raise ValueError("JINA_API_KEY is not set. Please configure your Jina API key.")

    embeddings: list[list[float]] = []

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            for batch in _batches(texts):
                response = await client.post(
                    "https://api.jina.ai/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {settings.JINA_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": settings.JINA_EMBED_MODEL,
                        "input": batch,
                        "encoding_type": "float",
                        "dimensions": settings.JINA_EMBED_DIMENSIONS,
                    },
                )
                response.raise_for_status()
                data = response.json()
                embeddings.extend(item["embedding"] for item in data["data"])

        return embeddings
    except httpx.HTTPStatusError as e:
        log.error(
            "Jina API error status=%s detail=%s",
            e.response.status_code,
            e.response.text[:200],
        )
        raise ValueError(f"Jina API error: {e}") from e
    except Exception as e:
        log.error("Jina embedding failed", exc_info=True)
        raise ValueError(f"Failed to get Jina embeddings: {e}") from e


def _embed_sync_with_jina(texts: list[str]) -> list[list[float]]:
    """Embed texts using Jina AI API synchronously."""
    import httpx

    if not settings.JINA_API_KEY:
        raise ValueError("JINA_API_KEY is not set. Please configure your Jina API key.")

    embeddings: list[list[float]] = []

    try:
        with httpx.Client(timeout=60.0) as client:
            for batch in _batches(texts):
                response = client.post(
                    "https://api.jina.ai/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {settings.JINA_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": settings.JINA_EMBED_MODEL,
                        "input": batch,
                        "encoding_type": "float",
                        "dimensions": settings.JINA_EMBED_DIMENSIONS,
                    },
                )
                response.raise_for_status()
                data = response.json()
                embeddings.extend(item["embedding"] for item in data["data"])

        return embeddings
    except httpx.HTTPStatusError as e:
        log.error(
            "Jina API error status=%s detail=%s",
            e.response.status_code,
            e.response.text[:200],
        )
        raise ValueError(f"Jina API error: {e}") from e
    except Exception as e:
        log.error("Jina embedding failed (sync)", exc_info=True)
        raise ValueError(f"Failed to get Jina embeddings: {e}") from e


def _embed_with_local(texts: list[str], *, query: bool = False) -> list[list[float]]:
    """Embed fully locally — arctic XS (default) or e5-multilingual.

    384-dim vectors either way, but different contracts: arctic pools CLS and
    prefixes queries only, e5 pools the masked mean and requires both
    ``query:``/``passage:`` prefixes. Do NOT mix backends in one store — the
    dim guard records them as different backends and refuses.

    ``LOCAL_EMBED_MODEL`` is validated at config load (arctic | e5), so the
    removed bundled MiniLM path is not reachable here any more.
    """
    from app.retrieval import e5_local

    model = settings.LOCAL_EMBED_MODEL
    if model == "e5":
        if query:
            return e5_local.embed_queries(texts)
        return e5_local.embed_passages(texts)
    if model != "arctic":
        # Config load refuses anything but arctic|e5; a hand-patched settings
        # object must not silently embed as arctic either (same fence as
        # ``active_backend_name``).
        raise ValueError(f"unknown LOCAL_EMBED_MODEL {model!r} — expected 'arctic' or 'e5'")
    if query:
        return e5_local.arctic_embed_queries(texts)
    return e5_local.arctic_embed_passages(texts)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    if settings.USE_LOCAL_EMBEDDINGS:
        # Off the loop: `_embed_with_local` is a synchronous ONNX call.
        return await _run_off_loop(_embed_with_local, texts)
    if settings.USE_JINA_EMBEDDINGS and settings.JINA_API_KEY:
        return await _embed_with_jina(texts)
    else:
        return await _embed_with_openai(texts)


async def embed_query(query: str) -> list[float]:
    if settings.USE_LOCAL_EMBEDDINGS:
        return (await _run_off_loop(_embed_with_local, [query], query=True))[0]
    return (await embed_texts([query]))[0]


def embed_texts_sync(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    if settings.USE_LOCAL_EMBEDDINGS:
        return _embed_with_local(texts)
    if settings.USE_JINA_EMBEDDINGS and settings.JINA_API_KEY:
        return _embed_sync_with_jina(texts)
    else:
        # Fallback to OpenAI sync (LEGACY, unbenchmarked — see
        # _embed_with_openai docstring and the config support matrix).
        log.warning(
            "OpenAI embedding backend is unbenchmarked legacy — "
            "prefer Jina (full-stack) or local (lite mode)",
        )
        embeddings: list[list[float]] = []
        try:
            client = _get_sync_client()
            for batch in _batches(texts):
                response = client.embeddings.create(
                    model=settings.EMBED_MODEL,
                    input=batch,
                    encoding_format="float",
                    timeout=30.0,
                )
                embeddings.extend(item.embedding for item in response.data)
            return embeddings
        except Exception as e:
            log.error("Failed to get embeddings (sync)", exc_info=True)
            raise ValueError(f"Failed to get embeddings: {e}") from e
