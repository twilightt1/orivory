"""
Pytest configuration for smoke tests.

Provides fixtures for Docker services and service discovery.
"""
import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "smoke: marks tests as smoke tests (deselect with '-m \"not smoke\"')"
    )


@pytest.fixture(scope="session")
def docker_services():
    """
    Fixture that ensures Docker services are available.

    In CI, services are started via docker compose.
    In local development, assumes services are running.

    Returns a dict with service connection info.
    """
    return {
        "postgres": {
            "host": "localhost",
            "port": 5432,
            "database": "ragdb",
            "user": "postgres",
            "password": "password",
        },
        "redis": {
            "host": "localhost",
            "port": 6379,
            "db": 0,
        },
        "qdrant": {
            "host": "localhost",
            "port": 6333,
        },
        "minio": {
            "host": "localhost",
            "port": 9000,
            "access_key": "minioadmin",
            "secret_key": "minioadmin",
        },
        "api": {
            "host": "localhost",
            "port": 8000,
        },
    }
