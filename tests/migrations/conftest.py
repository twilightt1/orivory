"""Conftest for schema-fidelity tests.

These tests introspect alembic migrations against Base.metadata and need NO
database. The root conftest's autouse setup_db fixture (which skips the
entire suite when Postgres is absent) is deliberately not inherited: run with

    pytest --confcutdir=tests/migrations tests/migrations

mirroring how CI isolates the other DB-free suites.
"""
import os

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:////tmp/orivory_schema_test.db")
os.environ.setdefault("REDIS_URL", "")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-testing-only")
