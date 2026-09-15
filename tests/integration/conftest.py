from __future__ import annotations

import asyncio
import os

import pytest
import pytest_asyncio

_TEST_ENV_DEFAULTS = {
    "DATABASE_URL": "postgresql+asyncpg://postgres:password@localhost:55432/ragdb",
    "REDIS_URL": "redis://localhost:6379/0",
    "JWT_SECRET_KEY": "test-secret-key-change-in-production",
    "MINIO_ENDPOINT": "localhost:9000",
    "MINIO_ACCESS_KEY": "minioadmin",
    "MINIO_SECRET_KEY": "minioadmin",
    "MINIO_BUCKET": "rag-docs",
    "MINIO_SECURE": "false",
    "QDRANT_URL": "http://localhost:6333",
    "OPENROUTER_API_KEY": "test-openrouter-key",
    "OPENAI_API_KEY": "test-openai-key",
    "JINA_API_KEY": "test-jina-key",
    "ENVIRONMENT": "test",
}

for key, value in _TEST_ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

pytestmark = [pytest.mark.integration, pytest.mark.requires_infra]


@pytest_asyncio.fixture
async def db():
    """Session-scoped-style DB fixture for integration tests.

    Tables are created by the Alembic `migrate` step that runs before this
    suite in CI (`docker compose run --rm migrate`). Locally, run
    `docker compose up -d migrate` first.
    """
    from app.database import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        yield session
        await session.rollback()


def pytest_runtest_setup(item):
    if "requires_infra" in item.keywords and os.getenv("RUN_LIVE_INTEGRATION") != "1":
        pytest.skip("Set RUN_LIVE_INTEGRATION=1 to run live integration tests.")


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(autouse=True)
async def reset_async_resources_between_tests():
    yield

    from app.database import engine

    await engine.dispose()
