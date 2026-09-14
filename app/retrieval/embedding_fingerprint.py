"""Embedding contract fingerprint (P0). Same-dim swaps must still reindex."""
from __future__ import annotations

import hashlib

from app.config import settings
from app.retrieval.e5_local import (
    _MAX_TOKENS,
    ARCTIC_QUERY_PREFIX,
    PASSAGE_PREFIX,
    QUERY_PREFIX,
)

DOC_FORMAT = "title-content-v1"
LEGACY_MEAN_FINGERPRINT = {
    "model_id": "Snowflake/snowflake-arctic-embed-xs",
    "pooling": "legacy-mean",  # current implementation mean-pools, not standard CLS
    "query_prefix": ARCTIC_QUERY_PREFIX,
    "passage_prefix": "",
    "max_tokens": _MAX_TOKENS,
    "normalize": True,
    "dim": 384,
    "doc_format": DOC_FORMAT,
}


def current_fingerprint() -> dict:
    """Fingerprint for the active embedding contract.

    P0 keeps local Arctic XS on legacy mean pooling until the separate CLS
    parity probe passes its gate.
    """
    if settings.USE_LOCAL_EMBEDDINGS and settings.LOCAL_EMBED_MODEL == "arctic":
        return dict(LEGACY_MEAN_FINGERPRINT)
    if settings.USE_LOCAL_EMBEDDINGS and settings.LOCAL_EMBED_MODEL == "e5":
        return {
            "model_id": "Xenova/multilingual-e5-small",
            "pooling": "mean",
            "query_prefix": QUERY_PREFIX,
            "passage_prefix": PASSAGE_PREFIX,
            "max_tokens": _MAX_TOKENS,
            "normalize": True,
            "dim": 384,
            "doc_format": DOC_FORMAT,
        }
    if settings.USE_JINA_EMBEDDINGS and settings.JINA_API_KEY:
        return {
            "model_id": "jina-embeddings-v5-text-small",
            "pooling": "api",
            "query_prefix": "",
            "passage_prefix": "",
            "max_tokens": _MAX_TOKENS,
            "normalize": True,
            "dim": 1024,
            "doc_format": DOC_FORMAT,
        }
    return {
        "model_id": "openai",
        "pooling": "api",
        "query_prefix": "",
        "passage_prefix": "",
        "max_tokens": _MAX_TOKENS,
        "normalize": True,
        "dim": 1536,
        "doc_format": DOC_FORMAT,
    }


def cache_key(fingerprint: dict, kind: str, text: str) -> str:
    """Hash an embedding contract, input kind, and text into a cache key."""
    h = hashlib.sha256()
    h.update(repr(sorted(fingerprint.items())).encode())
    h.update(b"\x00" + kind.encode() + b"\x00" + text.encode())
    return h.hexdigest()
