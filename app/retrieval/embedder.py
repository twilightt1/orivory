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
    """Embed texts using OpenAI-compatible API."""
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
        # Fallback to OpenAI sync
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
