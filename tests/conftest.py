"""Pytest configuration and shared fixtures."""
import asyncio
import os
import warnings

# Mock required environment variables BEFORE importing app modules
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://postgres:password@localhost:55432/ragdb_test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-testing-only")

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.database import Base, get_db
from app.main import app

# ─── Rate Limiter Mock ─────────────────────────────────────────────────────────
# Disable rate limiting in tests by mocking Redis calls.
# Each test should be fast enough that rate limits shouldn't apply.


class MockRedisPipeline:
    """Mock Redis pipeline that returns safe values for rate limiting."""

    def zremrangebyscore(self, *args):
        return self

    def zcard(self, *args):
        return self

    def zadd(self, *args, **kwargs):
        return self

    def expire(self, *args):
        return self

    async def execute(self):
        # Return (removed_count, current_count, added_count, ttl)
        # current_count=0 means we're under the limit
        return (0, 0, 1, 60)


class MockRedis:
    """Mock Redis for tests that returns empty pipeline results."""

    def __init__(self):
        self._counters = {}  # For incr() mocking

    def pipeline(self):
        """Return a mock pipeline (synchronous method, async execute)."""
        return MockRedisPipeline()

    async def incr(self, key: str) -> int:
        """Mock incr that always returns 1 (under limit)."""
        self._counters[key] = self._counters.get(key, 0) + 1
        return self._counters[key]

    async def expire(self, key: str, seconds: int) -> bool:
        """Mock expire - always succeeds."""
        return True

    async def get(self, key: str) -> str | None:
        """Mock get - always returns None."""
        return None

    async def set(self, key: str, value, ex: int | None = None, nx: bool = False) -> bool:
        """Mock SET with NX semantics mirroring the atomic-window pattern."""
        if nx:
            if key in self._counters:
                return False
            self._counters[key] = value
            return True
        self._counters[key] = value
        return True

    async def setex(self, key: str, seconds: int, value: str) -> bool:
        """Mock setex - always succeeds."""
        return True

    async def delete(self, key: str) -> int:
        """Mock delete - always succeeds."""
        return 1

    async def zcard(self, key: str) -> int:
        """Mock zcard for rate limiting - always returns 0."""
        return 0

    async def zadd(self, key: str, mapping: dict) -> int:
        """Mock zadd for rate limiting."""
        return 1

    async def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> int:
        """Mock zremrangebyscore for rate limiting."""
        return 0


_mock_redis = MockRedis()


_DB_AVAILABLE: bool | None = None
_DB_ERROR: str = ""


async def require_db_available() -> None:
    """Skip the calling test when Postgres is unreachable.

    For tests that manage their own engine (loop-local engines in the hub
    security / import suites) instead of using the ``db`` fixture. Probes
    live when the session fixture never ran (e.g. ``--confcutdir``
    isolation), otherwise reuses its recorded result.
    """
    global _DB_AVAILABLE, _DB_ERROR
    if _DB_AVAILABLE is None:
        try:
            from sqlalchemy import text

            probe = create_async_engine(TEST_DATABASE_URL, poolclass=NullPool)
            async with probe.begin() as conn:
                await conn.execute(text("SELECT 1"))
            await probe.dispose()
            _DB_AVAILABLE = True
        except Exception as e:
            _DB_AVAILABLE = False
            _DB_ERROR = str(e)
    if not _DB_AVAILABLE:
        pytest.skip(f"Database not available: {_DB_ERROR}")


@pytest.fixture(autouse=True)
def mock_redis_for_rate_limiter(monkeypatch):
    """Mock Redis client to bypass rate limiting in tests."""
    async def mock_get_redis():
        return _mock_redis

    # Patch Redis at all locations where it's used
    monkeypatch.setattr("app.middleware.rate_limiter.get_redis", mock_get_redis)
    monkeypatch.setattr("app.redis_client.get_redis", mock_get_redis)
    monkeypatch.setattr("app.services.auth_service.get_redis", mock_get_redis)
    monkeypatch.setattr("app.api.v1.auth.get_redis", mock_get_redis)

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://postgres:password@localhost:55432/ragdb_test",
)

test_engine = create_async_engine(TEST_DATABASE_URL, poolclass=NullPool)
TestSession = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_db():
    """Provision test tables once per session — WITHOUT skipping the suite.

    Historically this fixture called ``pytest.skip()`` when Postgres was
    unreachable, which silently skipped the ENTIRE suite (648 tests, exit 0):
    `make test` looked green while running nothing. Now the failure is
    recorded and only tests that actually need the database (via the ``db``
    / ``client`` fixtures below) skip; pure-unit tests run regardless.
    """
    global _DB_AVAILABLE, _DB_ERROR
    try:
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        _DB_AVAILABLE = True
        yield
    except Exception as e:
        _DB_AVAILABLE = False
        _DB_ERROR = str(e)
        warnings.warn(
            f"Postgres unavailable at {TEST_DATABASE_URL} ({e}); "
            f"DB-backed tests will skip, unit tests still run. "
            f"Start it with: docker compose up -d postgres",
            stacklevel=2,
        )
        yield
    finally:
        if _DB_AVAILABLE:
            try:
                async with test_engine.begin() as conn:
                    await conn.run_sync(Base.metadata.drop_all)
            except Exception:
                pass


@pytest_asyncio.fixture
async def db():
    if not _DB_AVAILABLE:
        pytest.skip(f"Database not available: {_DB_ERROR}")
    async with TestSession() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def client(db):
    async def override_db():
        yield db

    app.dependency_overrides[get_db] = override_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
