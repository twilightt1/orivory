"""Pytest configuration and shared fixtures."""
import asyncio
import os
import tempfile

# Point app import at a per-run SQLite file BEFORE any app module is imported.
# SQLite is the only supported DATABASE_URL and is always available, so the
# whole suite runs with NO environment overrides.
os.environ.setdefault(
    "DATABASE_URL", f"sqlite+aiosqlite:///{tempfile.mkdtemp(prefix='orivory-tests-')}/orivory-tests.db"
)
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-testing-only")

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app import database
from app.database import Base, get_db
from app.main import app

# The effective URL every DB-backed fixture and loop-local engine resolves to.
TEST_DATABASE_URL = os.environ["DATABASE_URL"]

# ─── Rate Limiter Mock ─────────────────────────────────────────────────────────
# Disable rate limiting in tests by mocking Redis calls.
# Each test should be fast enough that rate limits shouldn't apply.


class MockRedisPipeline:
    """Mock Redis pipeline.

    Rate-limit commands are no-ops (tests must never be throttled); the
    refresh-token commands the auth service writes through a pipeline are
    queued and applied to the mock on ``execute()``, so a later
    ``smembers``/``delete`` sees them.
    """

    def __init__(self, redis: "MockRedis"):
        self._redis = redis
        self._queued: list[tuple[str, tuple]] = []

    def zremrangebyscore(self, *args):
        return self

    def zcard(self, *args):
        return self

    def zadd(self, *args, **kwargs):
        return self

    def expire(self, *args):
        return self

    def setex(self, key: str, seconds: int, value: str):
        return self._queue("setex", key, seconds, value)

    def sadd(self, key: str, *members: str):
        return self._queue("sadd", key, *members)

    def _queue(self, name: str, *args):
        self._queued.append((name, args))
        return self

    async def execute(self):
        for name, args in self._queued:
            await getattr(self._redis, name)(*args)
        self._queued.clear()
        # Return (removed_count, current_count, added_count, ttl)
        # current_count=0 means we're under the limit
        return (0, 0, 1, 60)


class MockRedis:
    """Mock Redis for tests that returns empty pipeline results."""

    def __init__(self):
        self._counters = {}  # For incr() mocking
        self._values = {}  # Values written through set/setex
        self._sets = {}  # Sets written through sadd

    def pipeline(self):
        """Return a mock pipeline (synchronous method, async execute)."""
        return MockRedisPipeline(self)

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
        """Mock setex - stores the value, always succeeds."""
        self._values[key] = str(value)
        return True

    async def sadd(self, key: str, *members: str) -> int:
        """Mock sadd - remembers the set members (refresh-token index)."""
        target = self._sets.setdefault(key, set())
        added = len([member for member in members if member not in target])
        target.update(members)
        return added

    async def smembers(self, key: str) -> "set[str]":
        """Mock smembers - the members added through sadd()."""
        return set(self._sets.get(key, set()))

    async def delete(self, *keys: str) -> int:
        """Mock delete - always succeeds."""
        removed = 0
        for key in keys:
            for store in (self._values, self._sets, self._counters):
                if store.pop(key, None) is not None:
                    removed += 1
        return removed

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


test_engine = create_async_engine(
    TEST_DATABASE_URL, poolclass=NullPool, connect_args={"check_same_thread": False}
)
event.listen(test_engine.sync_engine, "connect", database._configure_sqlite_connection)
TestSession = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


def make_async_test_engine(url: str = TEST_DATABASE_URL):
    """A loop-local async engine on the suite's SQLite file.

    NullPool on purpose: pytest-asyncio hands every test a fresh loop, and a
    pooled aiosqlite connection would stay bound to whichever loop created it
    ("attached to a different loop"). Suites that override ``get_db`` with an
    engine of their own must build it inside the test coroutine.
    """
    engine = create_async_engine(
        url, poolclass=NullPool, connect_args={"check_same_thread": False}
    )
    event.listen(engine.sync_engine, "connect", database._configure_sqlite_connection)
    return engine


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
    """Provision the test schema once per session on the per-run SQLite file.

    Every DB-backed test in the suite resolves to THIS file (the ``db`` /
    ``client`` fixtures below, and the loop-local engines some suites build
    from ``TEST_DATABASE_URL``), so nothing skips: SQLite is always there.
    """
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


@pytest_asyncio.fixture
async def db():
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
