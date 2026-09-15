"""Shared plumbing for the P1b migration CLI tests (Task 5).

The CLI is a SCRIPT (``scripts/migrate_qdrant.py``), not an importable package
member, so it is loaded here by path once per session. Every test gets a
private SQLite file + a private embedded-Qdrant folder: the CLI resolves both
from ``settings``, so the ambient deployment is never touched.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPO_ROOT / "scripts" / "migrate_qdrant.py"


@pytest.fixture(scope="session")
def migrate_cli():
    """The migration CLI module, imported once (module-level app imports)."""
    spec = importlib.util.spec_from_file_location("migrate_qdrant", CLI_PATH)
    assert spec is not None and spec.loader is not None, f"cannot load {CLI_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["migrate_qdrant"] = module
    spec.loader.exec_module(module)
    return module
