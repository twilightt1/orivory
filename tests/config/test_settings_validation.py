import os

import pytest
from pydantic import ValidationError

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://postgres:password@localhost:5432/ragdb")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-change-in-production")

from app.config import Settings


def _base_settings(**overrides):
    values = {
        "DATABASE_URL": "postgresql+asyncpg://postgres:password@localhost:5432/ragdb",
        "REDIS_URL": "redis://localhost:6379/0",
        "JWT_SECRET_KEY": "dev-secret-key",
        "ALLOWED_ORIGINS": "http://localhost:3000,http://localhost:5173",
        "ENVIRONMENT": "development",
    }
    values.update(overrides)
    return Settings(**values)


def _production_settings(**overrides):
    values = {
        "DATABASE_URL": "postgresql+asyncpg://orivory:strong-db-password@postgres:5432/ragdb",
        "REDIS_URL": "redis://redis:6379/0",
        "JWT_SECRET_KEY": "production-secret-key-with-more-than-32-characters",
        "MINIO_ACCESS_KEY": "orivory-prod-minio",
        "MINIO_SECRET_KEY": "orivory-prod-minio-secret",
        "OPENROUTER_API_KEY": "sk-or-production",
        "OPENAI_API_KEY": "sk-production",
        "JINA_API_KEY": "jina-production",
        "CONFIG_ENCRYPTION_KEY": "production-config-encryption-key-32chars",
        "ALLOWED_ORIGINS": "https://app.orivory.example",
        "ENVIRONMENT": "production",
    }
    values.update(overrides)
    return Settings(**values)


def test_development_defaults_minio_credentials():
    settings = _base_settings(MINIO_ACCESS_KEY=None, MINIO_SECRET_KEY=None)

    assert settings.ENVIRONMENT == "development"
    assert settings.MINIO_ACCESS_KEY == "minioadmin"
    assert settings.MINIO_SECRET_KEY == "minioadmin"


def test_production_rejects_placeholder_jwt_secret():
    with pytest.raises(ValidationError, match="JWT_SECRET_KEY"):
        _production_settings(JWT_SECRET_KEY="change-me-to-a-random-256-bit-secret")


def test_production_rejects_short_jwt_secret():
    with pytest.raises(ValidationError, match="at least 32 characters"):
        _production_settings(JWT_SECRET_KEY="too-short")


def test_production_rejects_wildcard_cors():
    with pytest.raises(ValidationError, match="ALLOWED_ORIGINS"):
        _production_settings(ALLOWED_ORIGINS="*")


def test_production_rejects_missing_provider_keys():
    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        _production_settings(OPENAI_API_KEY="")


def test_production_rejects_default_minio_credentials():
    with pytest.raises(ValidationError, match="Default MinIO"):
        _production_settings(MINIO_ACCESS_KEY="minioadmin")


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


def test_normalizes_evaluator_failure_mode():
    settings = _base_settings(EVALUATOR_FAILURE_MODE="FAIL_CLOSED")

    assert settings.EVALUATOR_FAILURE_MODE == "fail_closed"
