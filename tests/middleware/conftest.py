"""Conftest for middleware tests (DB-free unit tests).

Provide env so app import works, and run with
`pytest --confcutdir=tests/middleware tests/middleware` to stay clear of the
root conftest's shared session fixtures.
"""
import os
import tempfile

_MW_TEST_ENV_DEFAULTS = {
    "DATABASE_URL": f"sqlite+aiosqlite:///{tempfile.mkdtemp(prefix='orivory-middleware-')}/mw-test.db",
    "JWT_SECRET_KEY": "test-secret-key-for-testing-only",
    "ENVIRONMENT": "test",
}

for key, value in _MW_TEST_ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)
