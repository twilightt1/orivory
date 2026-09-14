"""Embedding contract fingerprint (P0). Same-dim swaps must still reindex."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from app.config import settings
from app.retrieval.e5_local import (
    _MAX_TOKENS,
    ARCTIC_MODEL_SHA256,
    ARCTIC_QUERY_PREFIX,
    ARCTIC_TOKENIZER_SHA256,
    MODEL_SHA256,
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    TOKENIZER_SHA256,
)

DOC_FORMAT = "title-content-v1"
# These are immutable upstream commits, not mutable ``resolve/main`` URLs.
ARCTIC_MODEL_REVISION = "d8c86521100d3556476a063fc2342036d45c106f"
E5_MODEL_REVISION = "761b726dd34fb83930e26aab4e9ac3899aa1fa78"
ARCTIC_ARTIFACT_SHA256 = ARCTIC_MODEL_SHA256
E5_ARTIFACT_SHA256 = MODEL_SHA256
E5_TOKENIZER_SHA256 = TOKENIZER_SHA256


def _contract(
    *,
    model_id: str,
    revision: str | None,
    artifact_digest: str | None,
    tokenizer_digest: str | None,
    pooling: str,
    query_prefix: str,
    passage_prefix: str,
    max_tokens: int | None,
    truncation: str,
    padding: str,
    normalize: bool | None,
    dim: int,
    precision: str,
    provider: str,
    graph_outputs: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "model_revision": revision,
        # Keep the short spelling for consumers that used the original design
        # vocabulary; both values are part of the canonical fingerprint.
        "revision": revision,
        "artifact_digest": artifact_digest,
        "tokenizer_digest": tokenizer_digest,
        "graph_outputs": graph_outputs or ["last_hidden_state"],
        "pooling": pooling,
        "query_prefix": query_prefix,
        "passage_prefix": passage_prefix,
        "max_tokens": max_tokens,
        "truncation": truncation,
        "padding": padding,
        "normalize": normalize,
        "dim": dim,
        "precision": precision,
        "provider": provider,
        "doc_format": DOC_FORMAT,
    }


LEGACY_MEAN_FINGERPRINT = _contract(
    model_id="Snowflake/snowflake-arctic-embed-xs",
    revision=ARCTIC_MODEL_REVISION,
    artifact_digest=ARCTIC_ARTIFACT_SHA256,
    tokenizer_digest=ARCTIC_TOKENIZER_SHA256,
    pooling="legacy-mean",  # current implementation mean-pools, not standard CLS
    query_prefix=ARCTIC_QUERY_PREFIX,
    passage_prefix="",
    max_tokens=_MAX_TOKENS,
    truncation="head",
    padding="batch-longest-zero",
    normalize=True,
    dim=384,
    precision="float32",
    provider="onnxruntime-cpu",
)

# P1b cutover: production arctic XS pools the [CLS] token — the pooling the
# model was trained with — instead of the legacy masked mean. Derived from the
# legacy contract so ``pooling`` is provably the only field that differs.
# ``LEGACY_MEAN_FINGERPRINT`` stays importable: the mean-vs-CLS ablation and
# the Chroma rollback tool both need to name the old generation.
ARCTIC_CLS_FINGERPRINT: dict[str, Any] = {**LEGACY_MEAN_FINGERPRINT, "pooling": "cls"}


def canonical_fingerprint(fingerprint: str | dict[str, Any]) -> str:
    """Serialize a fingerprint deterministically for metadata and cache keys."""
    if isinstance(fingerprint, str):
        return fingerprint
    if isinstance(fingerprint, dict):
        try:
            return json.dumps(
                fingerprint,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid embedding fingerprint") from exc
    raise ValueError("embedding fingerprint must be a string or dict")


def fingerprint_generation(fingerprint: str | dict[str, Any]) -> str:
    """Return a stable contract-generation token.

    This is deliberately a contract token, not the future SQLite active
    generation manifest. It prevents ``only_missing`` from treating a
    same-dimension contract change as fresh data.
    """
    return hashlib.sha256(canonical_fingerprint(fingerprint).encode()).hexdigest()


def current_fingerprint() -> dict[str, Any]:
    """Fingerprint for the active embedding contract.

    P1b puts local Arctic XS on CLS pooling (``ARCTIC_CLS_FINGERPRINT``); e5
    keeps mean. API providers expose null immutable artifact provenance
    because the provider does not publish a revision through this interface;
    the fields remain explicit instead of claiming one.
    """
    if settings.USE_LOCAL_EMBEDDINGS and settings.LOCAL_EMBED_MODEL == "arctic":
        return dict(ARCTIC_CLS_FINGERPRINT)
    if settings.USE_LOCAL_EMBEDDINGS and settings.LOCAL_EMBED_MODEL == "e5":
        return _contract(
            model_id="Xenova/multilingual-e5-small",
            revision=E5_MODEL_REVISION,
            artifact_digest=E5_ARTIFACT_SHA256,
            tokenizer_digest=E5_TOKENIZER_SHA256,
            pooling="mean",
            query_prefix=QUERY_PREFIX,
            passage_prefix=PASSAGE_PREFIX,
            max_tokens=_MAX_TOKENS,
            truncation="head",
            padding="batch-longest-zero",
            normalize=True,
            dim=384,
            precision="float32",
            provider="onnxruntime-cpu",
            graph_outputs=["last_hidden_state"],
        )
    if settings.USE_JINA_EMBEDDINGS and settings.JINA_API_KEY:
        return _contract(
            model_id=settings.JINA_EMBED_MODEL,
            revision=None,
            artifact_digest=None,
            tokenizer_digest=None,
            pooling="api",
            query_prefix="",
            passage_prefix="",
            max_tokens=None,
            truncation="provider-defined",
            padding="provider-defined",
            normalize=None,
            dim=settings.JINA_EMBED_DIMENSIONS,
            precision="float32",
            provider="jina-api",
            graph_outputs=["embedding"],
        )
    return _contract(
        model_id=settings.EMBED_MODEL,
        revision=None,
        artifact_digest=None,
        tokenizer_digest=None,
        pooling="api",
        query_prefix="",
        passage_prefix="",
        max_tokens=None,
        truncation="provider-defined",
        padding="provider-defined",
        normalize=None,
        dim=settings.EMBED_DIMENSIONS,
        precision="float32",
        provider="openai-compatible-api",
        graph_outputs=["embedding"],
    )


_GENERATION_FAMILIES = {"memory": "orivory_memories", "chunk": "orivory_chunks"}


def generation_name(kind: str, fingerprint: str | dict[str, Any] | None = None) -> str:
    """Name the vector generation/collection for ``kind`` and a contract.

    ``orivory_memories__<fp8>`` / ``orivory_chunks__<fp8>``, where ``fp8`` is
    the first 8 hex characters of :func:`fingerprint_generation` — the same
    token the payload and the ``index_generations`` manifest carry, shortened
    to a usable name. The migration/cutover tool names the NEW generation with
    this helper; the runtime fallbacks in ``app.retrieval.memory.outbox`` keep
    their old spellings for un-migrated databases.
    ``None`` means "the active contract"; a falsy-but-explicit fingerprint
    (``{}`` / ``""``) is a caller bug, never a synonym for ``None``.
    """
    try:
        family = _GENERATION_FAMILIES[kind]
    except KeyError:
        raise ValueError(
            f"unknown index kind {kind!r} — expected one of {sorted(_GENERATION_FAMILIES)}"
        ) from None
    if fingerprint is None:
        fingerprint = current_fingerprint()
    if not fingerprint:
        # Rollback (T6) and ablation (T8) name generations they built: silently
        # naming the ACTIVE one would point them at the wrong vectors.
        raise ValueError(
            f"empty embedding fingerprint {fingerprint!r} — pass None to name the active generation"
        )
    return f"{family}__{fingerprint_generation(fingerprint)[:8]}"


def cache_key(fingerprint: dict, kind: str, text: str) -> str:
    """Hash an embedding contract, input kind, and text into a cache key."""
    h = hashlib.sha256()
    h.update(canonical_fingerprint(fingerprint).encode())
    h.update(len(kind.encode()).to_bytes(8, "big"))
    h.update(kind.encode())
    h.update(len(text.encode()).to_bytes(8, "big"))
    h.update(text.encode())
    return h.hexdigest()
