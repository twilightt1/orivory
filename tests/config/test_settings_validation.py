import os
import tempfile

import pytest
from pydantic import ValidationError

os.environ.setdefault(
    "DATABASE_URL", f"sqlite+aiosqlite:///{tempfile.mkdtemp(prefix='orivory-config-')}/config-test.db"
)

from app.config import Settings

_SQLITE_URL = "sqlite+aiosqlite:////tmp/orivory-config-test.db"


def _base_settings(**overrides):
    values = {
        "DATABASE_URL": _SQLITE_URL,
        "ALLOWED_ORIGINS": "http://localhost:3000,http://localhost:5173",
        "ENVIRONMENT": "development",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _production_settings(**overrides):
    values = {
        "DATABASE_URL": _SQLITE_URL,
        "OPENROUTER_API_KEY": "«redacted:sk-…»",
        "OPENAI_API_KEY": "«redacted:sk-…»",
        "JINA_API_KEY": "jina-production",
        "CONFIG_ENCRYPTION_KEY": "production-config-encryption-key-32chars",
        "ALLOWED_ORIGINS": "https://app.orivory.example",
        "ENVIRONMENT": "production",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_zerokey_embeddings_default_to_the_local_model():
    """No embedding key means the bundled local ONNX model — the zero-cost
    path a fresh self-host boots with."""
    settings = _base_settings(JINA_API_KEY="", OPENAI_API_KEY="")

    assert settings.USE_LOCAL_EMBEDDINGS is True


def test_local_owner_email_is_the_default_identity():
    """There is no account auth: the install's identity is one configured
    email, defaulted so a fresh self-host boots with an owner."""
    assert _base_settings().LOCAL_OWNER_EMAIL == "owner@orivory.local"
    assert _base_settings(LOCAL_OWNER_EMAIL="me@example.com").LOCAL_OWNER_EMAIL == "me@example.com"


def test_production_rejects_wildcard_cors():
    with pytest.raises(ValidationError, match="ALLOWED_ORIGINS"):
        _production_settings(ALLOWED_ORIGINS="*")


def test_production_rejects_missing_provider_keys():
    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        _production_settings(OPENAI_API_KEY="")


def test_production_rejects_missing_config_encryption_key():
    with pytest.raises(ValidationError, match="CONFIG_ENCRYPTION_KEY"):
        _production_settings(CONFIG_ENCRYPTION_KEY="")


def test_production_accepts_complete_safe_settings():
    settings = _production_settings()

    assert settings.ENVIRONMENT == "production"
    assert settings.is_production is True
    assert settings.ALLOWED_ORIGINS == "https://app.orivory.example"


def test_rejects_invalid_embedding_batch_size():
    with pytest.raises(ValidationError, match="EMBED_BATCH_SIZE"):
        _base_settings(EMBED_BATCH_SIZE=0)


def test_rejects_invalid_evaluator_failure_mode():
    with pytest.raises(ValidationError, match="EVALUATOR_FAILURE_MODE"):
        _base_settings(EVALUATOR_FAILURE_MODE="unsafe")


@pytest.mark.parametrize("rrf_k", [0, -1])
def test_rejects_invalid_rrf_k(rrf_k):
    """k <= 0 makes `1 / (k + rank + 1)` zero-divide at rank 0; the constant
    is validated at load like the other typed knobs, and the vector-outage
    fallback fuses with the hybrid flag OFF — an operator typo must be a boot
    failure, not an unhandled 500."""
    with pytest.raises(ValidationError, match="RETRIEVAL_RRF_K"):
        _base_settings(RETRIEVAL_RRF_K=rrf_k)


def test_accepts_rrf_k_at_the_boundary():
    assert _base_settings(RETRIEVAL_RRF_K=1).RETRIEVAL_RRF_K == 1


@pytest.mark.parametrize("workers", [0, -1])
def test_rejects_invalid_embed_executor_workers(workers):
    """A zero-width executor cannot embed at all — `ThreadPoolExecutor(0)`
    raises at construction, deep inside whichever request triggers the first
    embed. The typo must fail at load instead."""
    with pytest.raises(ValidationError, match="EMBED_EXECUTOR_WORKERS"):
        _base_settings(EMBED_EXECUTOR_WORKERS=workers)


@pytest.mark.parametrize("threads", [-1, -8])
def test_rejects_invalid_embed_ort_intra_op_threads(threads):
    """0 is the signed default (ORT's own all-cores width); a negative width
    is a typo, and ONNX Runtime refuses it far from config load."""
    with pytest.raises(ValidationError, match="EMBED_ORT_INTRA_OP_THREADS"):
        _base_settings(EMBED_ORT_INTRA_OP_THREADS=threads)


def test_accepts_ort_intra_op_boundary_values():
    assert _base_settings(EMBED_ORT_INTRA_OP_THREADS=0).EMBED_ORT_INTRA_OP_THREADS == 0
    assert _base_settings(EMBED_ORT_INTRA_OP_THREADS=2).EMBED_ORT_INTRA_OP_THREADS == 2


@pytest.mark.parametrize("cap", [0, -1])
def test_rejects_invalid_reranker_cap(cap):
    """The cap is the per-call `min(top_k, cap)`: a 0 cap asks the scorer
    for zero rows on every request — a typo, not a configuration."""
    with pytest.raises(ValidationError, match="RERANK_TOP_N"):
        _base_settings(RERANK_TOP_N=cap)


@pytest.mark.parametrize("multiplier", [0, -1.0])
def test_rejects_invalid_rerank_pool_multiplier(multiplier):
    """The pool feeds the count invariant: a non-positive multiplier is a typo
    that silently collapses the fetch, and the invariant would hide it."""
    with pytest.raises(ValidationError, match="RETRIEVAL_RERANK_POOL_MULTIPLIER"):
        _base_settings(RETRIEVAL_RERANK_POOL_MULTIPLIER=multiplier)


def test_environment_whitespace_is_normalized_before_the_production_gate():
    """`ENVIRONMENT='production '` is an operator typo, not a downgrade.

    Unstripped, the whole production block was skipped silently: no CORS
    check, no provider-key check, no CONFIG_ENCRYPTION_KEY check (and then
    the dev fallback key), and the dev MinIO credentials were installed.
    """
    settings = _production_settings(ENVIRONMENT="production ")

    assert settings.is_production is True
    assert settings.ENVIRONMENT == "production"


@pytest.mark.parametrize("spelling", ["production", " production", "PRODUCTION\t", "Production\n"])
def test_production_spellings_with_whitespace_still_validate(spelling):
    """Whatever the padding, production must still refuse an unsafe install."""
    assert _production_settings(ENVIRONMENT=spelling).is_production is True

    with pytest.raises(ValidationError, match="CONFIG_ENCRYPTION_KEY"):
        _production_settings(ENVIRONMENT=spelling, CONFIG_ENCRYPTION_KEY="")
    with pytest.raises(ValidationError, match="ALLOWED_ORIGINS"):
        _production_settings(ENVIRONMENT=spelling, ALLOWED_ORIGINS="")
    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        _production_settings(ENVIRONMENT=spelling, OPENAI_API_KEY="")


def test_normalizes_evaluator_failure_mode():
    settings = _base_settings(EVALUATOR_FAILURE_MODE="FAIL_CLOSED")

    assert settings.EVALUATOR_FAILURE_MODE == "fail_closed"
