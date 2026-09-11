import logging

from openai import AsyncOpenAI, OpenAI

from app.config import settings

log = logging.getLogger(__name__)


# Lazily-resolved module-level aliases.
def __getattr__(name):
    if name == "async_client":
        return _get_async_client()
    if name == "sync_client":
        return _get_sync_client()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_async_client: AsyncOpenAI | None = None
_sync_client: OpenAI | None = None
_local_embed_fn = None  # chromadb ONNX MiniLM (USE_LOCAL_EMBEDDINGS)

# Collection metadata keys recording which embedding backend + dimension a
# Chroma collection was created with. Flipping USE_JINA_EMBEDDINGS /
# USE_LOCAL_EMBEDDINGS after data exists used to write mismatched vectors
# into the same collection silently — the guard below fails loud instead.
EMBED_BACKEND_META_KEY = "orivory_embed_backend"
EMBED_DIM_META_KEY = "orivory_embed_dim"


class EmbeddingDimensionMismatch(ValueError):
    """Raised when embeddings don't match the collection's recorded backend.

    Either the embedding config changed after data was indexed (fix: reindex
    into a fresh collection or restore the previous backend), or vectors from
    two backends are being mixed (fix: don't).
    """


def active_backend_name() -> str:
    """Which embedding backend the current settings select."""
    if settings.USE_LOCAL_EMBEDDINGS:
        return "local"
    if settings.USE_JINA_EMBEDDINGS and settings.JINA_API_KEY:
        return "jina"
    return "openai"


def check_collection_dim(collection, embedding_dim: int, *, backend: str | None = None) -> dict | None:
    """Verify an embedding fits the collection's recorded backend/dim.

    Returns a replacement metadata dict to stamp when the collection carries
    no stamp yet (caller applies it via ``collection.modify`` in its own
    sync/async style), None when the stamp matches, and raises
    :class:`EmbeddingDimensionMismatch` on a backend or dimension switch.

    Collections whose metadata is unreadable (unit-test doubles, exotic
    wrappers) are skipped silently — the guard must never break a path it
    cannot verify.
    """
    backend = backend or active_backend_name()
    # Unwrap the local-mode sync→async adapter (its __getattr__ turns every
    # attribute access into a coroutine factory — read the inner instead).
    inner = getattr(collection, "_inner", collection)
    if type(inner).__name__ == "MagicMock":
        return None
    try:
        meta = inner.metadata
    except Exception:
        return None
    if not isinstance(meta, dict):
        return None
    recorded_dim = meta.get(EMBED_DIM_META_KEY)
    recorded_backend = meta.get(EMBED_BACKEND_META_KEY)
    if recorded_dim is None:
        return {**meta, EMBED_BACKEND_META_KEY: backend, EMBED_DIM_META_KEY: embedding_dim}
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


def stamp_collection_dim(collection, embedding_dim: int, *, backend: str | None = None) -> None:
    """Check the dim guard and stamp an unstamped collection (sync caller).

    Raises :class:`EmbeddingDimensionMismatch` on a backend/dim switch.
    Stamping failures are best-effort (logged, never raised) — a missing
    stamp only defers detection, while a stamp error must never break writes.
    """
    stamp = check_collection_dim(collection, embedding_dim, backend=backend)
    if stamp is None:
        return
    inner = getattr(collection, "_inner", collection)
    if type(inner).__name__ in ("MagicMock", "AsyncMock", "Mock"):
        return
    try:
        inner.modify(metadata=stamp)
    except Exception:
        log.warning("Could not stamp collection embedding dim", exc_info=True)


async def astamp_collection_dim(collection, embedding_dim: int, *, backend: str | None = None) -> None:
    """Async variant of :func:`stamp_collection_dim`."""
    import inspect

    stamp = check_collection_dim(collection, embedding_dim, backend=backend)
    if stamp is None:
        return
    inner = getattr(collection, "_inner", collection)
    if type(inner).__name__ in ("MagicMock", "AsyncMock", "Mock"):
        return
    try:
        res = inner.modify(metadata=stamp)
        if inspect.isawaitable(res):
            await res
    except Exception:
        log.warning("Could not stamp collection embedding dim", exc_info=True)


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


def _embed_with_local(texts: list[str]) -> list[list[float]]:
    """Embed with chromadb's bundled ONNX MiniLM — fully local, no API key.

    384-dim vectors. Chosen explicitly via USE_LOCAL_EMBEDDINGS=true (or
    automatically when Jina/OpenAI are unconfigured): keeps lite mode and
    benchmarks self-contained and free. Different dim than Jina/OpenAI —
    fine for a fresh collection, do NOT mix backends in one store.
    """
    global _local_embed_fn
    if _local_embed_fn is None:
        import chromadb.utils.embedding_functions as ef

        _local_embed_fn = ef.ONNXMiniLM_L6_V2()
    return [list(map(float, v)) for v in _local_embed_fn(texts)]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    if settings.USE_LOCAL_EMBEDDINGS:
        return _embed_with_local(texts)
    if settings.USE_JINA_EMBEDDINGS and settings.JINA_API_KEY:
        return await _embed_with_jina(texts)
    else:
        return await _embed_with_openai(texts)


async def embed_query(query: str) -> list[float]:
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
