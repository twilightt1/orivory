"""
Pytest configuration for smoke tests.

The lite stack is ONE container — the API (with its MCP hub, embedded Qdrant,
SQLite and filesystem storage) published on localhost:8000. There are no
internal services to discover any more.
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
    Fixture that ensures the lite compose stack is available.

    In CI, the stack is started via `docker compose up -d`.
    In local development, assumes `docker compose up -d` is already running.

    Returns a dict with the one endpoint the lite stack exposes.
    """
    return {
        "api": {
            "host": "localhost",
            "port": 8000,
        },
    }
