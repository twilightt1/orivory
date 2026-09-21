"""API test environment defaults."""

import os
import tempfile

_API_TEST_ENV_DEFAULTS = {
    "DATABASE_URL": f"sqlite+aiosqlite:///{tempfile.mkdtemp(prefix='orivory-api-')}/api-test.db",
    "JWT_SECRET_KEY": "test-secret-key-change-in-production",
    "ENVIRONMENT": "test",
}

for key, value in _API_TEST_ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)
