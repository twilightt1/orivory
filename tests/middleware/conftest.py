"""Conftest for middleware tests (DB-free unit tests).

Mirrors tests/services/conftest.py: provide env so app import works, and run
with `pytest --confcutdir=tests/middleware tests/middleware` to stay clear of
the root conftest's autouse Postgres fixture.
"""
import os

_MW_TEST_ENV_DEFAULTS = {
    "DATABASE_URL": "sqlite+aiosqlite:////tmp/orivory_middleware_test.db",
    "REDIS_URL": "",
    "JWT_SECRET_KEY": "test-secret-key-for-testing-only",
    "ENVIRONMENT": "test",
}

for key, value in _MW_TEST_ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)
