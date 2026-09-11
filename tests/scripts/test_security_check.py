"""Tests for the prod-compose gate in scripts/security_check.py (stdlib-light).

Regression (full-repo review, verified with `docker compose config` on
Compose v5.5.0): `docker-compose.prod.yml` used `ports: []` / `volumes: []`
to strip dev mappings, but Compose MERGES sequences instead of replacing
them — postgres/redis/chroma/minio stayed host-published and dev bind-mounts
survived into prod. The old gate grepped the YAML string for "ports: []" and
reported PASS while the exposure persisted.

The gate must therefore validate BEHAVIOR (merged `docker compose config`
output), not YAML text. These tests drive the parsing helpers with canned
merged-config output — no docker needed.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "security_check",
    Path(__file__).resolve().parents[2] / "scripts" / "security_check.py",
)
assert _spec is not None and _spec.loader is not None
security_check = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = security_check
_spec.loader.exec_module(security_check)

MERGED_CLEAN = """
services:
  postgres:
    ports: []
    volumes:
      - type: volume
        source: pgdata
  redis:
    ports: []
  app:
    volumes:
      - type: volume
        source: appdata
"""

MERGED_LEAKY_PORTS = """
services:
  postgres:
    ports:
      - mode: ingress
        target: 5432
        published: "55432"
        protocol: tcp
    volumes:
      - type: volume
        source: pgdata
  redis:
    ports: []
  app:
    volumes:
      - type: volume
        source: appdata
"""

MERGED_LEAKY_BINDS = """
services:
  postgres:
    ports: []
  app:
    volumes:
      - type: bind
        source: /repo/app
        target: /app/app
      - type: volume
        source: appdata
"""


def test_clean_merged_config_passes():
    result = security_check.check_merged_prod_config(MERGED_CLEAN)
    assert result.status == "PASS", result.detail


def test_leaky_host_port_fails_with_service_name():
    result = security_check.check_merged_prod_config(MERGED_LEAKY_PORTS)
    assert result.status == "FAIL", "host-published postgres port must fail the gate"
    assert "postgres" in result.detail


def test_host_bind_mount_fails_with_service_name():
    result = security_check.check_merged_prod_config(MERGED_LEAKY_BINDS)
    assert result.status == "FAIL", "dev bind-mount surviving into prod must fail"
    assert "app" in result.detail


def test_prod_override_uses_replace_semantics():
    """The committed override file must use merge-replacing syntax, so the
    no-op `ports: []` regression cannot silently return."""
    prod = (Path(__file__).resolve().parents[2] / "docker-compose.prod.yml").read_text()
    assert "!override []" in prod or "!reset []" in prod, (
        "docker-compose.prod.yml must use `!override []` (Compose 2.24+) — "
        "plain `ports: []` is a merge no-op and leaves services exposed"
    )
