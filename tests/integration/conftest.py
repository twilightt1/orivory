from __future__ import annotations

import asyncio
import os

import pytest
import pytest_asyncio

_TEST_ENV_DEFAULTS = {
    # The shape these live modules run against: SQLite + embedded Qdrant +
    # filesystem storage, i.e. the one container `docker compose up -d` boots.
    "DATABASE_URL": "sqlite+aiosqlite:////tmp/orivory-live-integration.db",
    "STORAGE_BACKEND": "fs",
    "FS_STORAGE_PATH": "/tmp/orivory-live-storage",
    "QDRANT_MODE": "local",
    "QDRANT_LOCAL_PATH": "/tmp/orivory-live-qdrant",
    "OPENROUTER_API_KEY": "test-openrouter-key",
    "OPENAI_API_KEY": "test-openai-key",
    "ENVIRONMENT": "test",
}

for key, value in _TEST_ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

pytestmark = [pytest.mark.integration, pytest.mark.requires_infra]


@pytest_asyncio.fixture
async def db():
    """Session-scoped-style DB fixture for integration tests.

    Tables come from the app's own SQLite bootstrap (`bootstrap_sqlite`, run by
    the container at boot or by the first connection here); there is no
    migration step to wait for any more.
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
