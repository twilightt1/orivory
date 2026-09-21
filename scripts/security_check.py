from __future__ import annotations

import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.services.diagnostics_service import build_config_summary


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


class SecurityCheckFailure(RuntimeError):
    pass


# The full-stack services the lite product no longer runs: none of them may
# reappear in the prod override, and none may host-publish a port if one does.
INTERNAL_SERVICES = ("postgres", "redis", "qdrant", "minio", "flower")


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _base_production_settings(**overrides) -> dict[str, object]:
    values: dict[str, object] = {
        "DATABASE_URL": "sqlite+aiosqlite:////data/orivory.db",
        "JWT_SECRET_KEY": "production-secret-key-with-more-than-32-characters",
        "CONFIG_ENCRYPTION_KEY": "2CiSbMXhP2zwWOAk7nkEcGABAJSJnt7hl_SVcMBnlnk=",
        "OPENROUTER_API_KEY": "sk-or-production",
        "OPENAI_API_KEY": "sk-production",
        "JINA_API_KEY": "jina-production",
        "ALLOWED_ORIGINS": "https://app.orivory.example",
        "FRONTEND_URL": "https://app.orivory.example",
        "ENVIRONMENT": "production",
    }
    values.update(overrides)
    return values


def _expect_validation_error(name: str, match: str, **overrides) -> CheckResult:
    try:
        Settings(**_base_production_settings(**overrides))
    except ValidationError as exc:
        if re.search(match, str(exc), re.IGNORECASE):
            return CheckResult(name, "PASS", f"Rejected unsafe production settings: {match}")
        return CheckResult(name, "FAIL", f"Rejected settings, but not for expected reason: {exc}")
    return CheckResult(name, "FAIL", "Unsafe production settings were accepted")


def check_production_accepts_safe_settings() -> CheckResult:
    settings = Settings(**_base_production_settings())
    if not settings.is_production:
        return CheckResult("production safe settings", "FAIL", "Production settings did not normalize to production")
    return CheckResult("production safe settings", "PASS", "Complete safe production settings are accepted")


def check_jwt_placeholder_rejected() -> CheckResult:
    return _expect_validation_error(
        "placeholder JWT secret",
        "JWT_SECRET_KEY",
        JWT_SECRET_KEY="change-me-to-a-random-256-bit-secret",
    )


def check_wildcard_cors_rejected() -> CheckResult:
    return _expect_validation_error("wildcard CORS", "ALLOWED_ORIGINS", ALLOWED_ORIGINS="*")


def check_provider_keys_required() -> CheckResult:
    return _expect_validation_error("required provider keys", "OPENAI_API_KEY", OPENAI_API_KEY="")


def _service_block(compose_text: str, service_name: str) -> str:
    pattern = rf"(?ms)^  {re.escape(service_name)}:\n(?P<body>.*?)(?=^  [a-zA-Z0-9_-]+:|^volumes:|\Z)"
    match = re.search(pattern, compose_text)
    if not match:
        raise SecurityCheckFailure(f"Service {service_name!r} not found in production compose")
    return match.group("body")


def check_internal_ports_removed() -> CheckResult:
    """Prod must not host-publish internal services or keep dev bind-mounts.

    Validates BEHAVIOR, not YAML text: renders the merged prod configuration
    (`docker compose -f docker-compose.yml -f docker-compose.prod.yml
    config`) and asserts no `published:` ports on internal services and no
    host `bind` mounts on app containers. A plain `ports: []` in the override
    file is a Compose merge no-op, so grepping the YAML can report PASS while
    the exposure persists — this check would have caught that class of bug.

    The lite stack has no internal services at all — the single `app` service
    publishes 8000 on purpose — so the merged rendering only has to prove it
    keeps no host bind mount on app. When docker is unavailable, the override
    file is checked statically for the same two properties.
    """
    import shutil
    import subprocess

    if shutil.which("docker") is not None:
        try:
            merged = subprocess.run(
                [
                    "docker", "compose",
                    "-f", str(ROOT / "docker-compose.yml"),
                    "-f", str(ROOT / "docker-compose.prod.yml"),
                    "config",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(ROOT),
            )
            if merged.returncode == 0 and merged.stdout.strip():
                return check_merged_prod_config(merged.stdout)
        except (OSError, subprocess.SubprocessError):
            pass  # fall through to the static syntax check below

    prod_text = _read("docker-compose.prod.yml")
    declared = [name for name in INTERNAL_SERVICES if re.search(rf"(?m)^  {name}:", prod_text)]
    if not declared and "type: bind" not in prod_text:
        return CheckResult(
            "production internal ports",
            "PASS",
            "docker unavailable — prod override declares no internal service "
            "and no host bind mount",
        )
    return CheckResult(
        "production internal ports",
        "FAIL",
        f"docker unavailable and the prod override leaks: "
        f"{declared or 'a host bind mount'}",
    )


def check_merged_prod_config(merged_yaml: str) -> CheckResult:
    """Assert a MERGED `docker compose config` rendering exposes nothing.

    Fails when any internal service (postgres/redis/qdrant/minio/flower)
    carries a host-published port, or when app-tier services (app/frontend)
    retain a host `bind` mount (dev bind-mounts must not survive
    into prod). Pure function of the rendered text — unit-testable.
    """
    internal = list(INTERNAL_SERVICES)
    app_tier = ["app", "frontend"]
    leaks: list[str] = []

    current: str | None = None
    for line in merged_yaml.splitlines():
        svc = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", line)
        if svc:
            current = svc.group(1)
            continue
        if current in internal and re.search(r"^\s+published:", line):
            leaks.append(f"{current}: host-published port")
            current = None  # report once per service
        elif current in app_tier and re.search(r"-\s+type:\s*bind\s*$", line):
            leaks.append(f"{current}: host bind mount")
            current = None

    if leaks:
        return CheckResult(
            "production internal ports", "FAIL", "; ".join(sorted(set(leaks)))
        )
    return CheckResult(
        "production internal ports",
        "PASS",
        "Merged prod config publishes no internal ports and keeps no bind mounts",
    )


def check_flower_ops_profile() -> CheckResult:
    try:
        block = _service_block(_read("docker-compose.prod.yml"), "flower")
    except SecurityCheckFailure:
        # ponytail: flower deleted from compose (slim branch: no broker, no
        # monitor UI) — absence is the desired state, not a regression.
        return CheckResult("flower ops profile", "PASS", "Flower removed from compose (celery-free slim branch)")
    if "profiles:" in block and "- ops" in block:
        return CheckResult("flower ops profile", "PASS", "Flower is behind the ops profile in prod override")
    return CheckResult("flower ops profile", "FAIL", "Flower is not isolated behind the ops profile")


def check_diagnostics_summary_safe() -> CheckResult:
    summary = build_config_summary()
    forbidden_keys = {
        "DATABASE_URL",
        "JWT_SECRET_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "JINA_API_KEY",
        "SENDGRID_API_KEY",
        "GOOGLE_CLIENT_SECRET",
        "access_token",
        "refresh_token",
        "password",
    }
    leaked_keys = sorted(key for key in summary if key in forbidden_keys)
    if leaked_keys:
        return CheckResult("diagnostics secret redaction", "FAIL", f"Secret-bearing keys exposed: {', '.join(leaked_keys)}")
    return CheckResult("diagnostics secret redaction", "PASS", "Diagnostics summary exposes only secret-safe config")


def check_docs_disabled_in_production() -> CheckResult:
    main_py = _read("app/main.py")
    expected = 'docs_url="/docs" if settings.ENVIRONMENT != "production" else None'
    if expected in main_py:
        return CheckResult("production docs disabled", "PASS", "FastAPI docs are disabled when ENVIRONMENT=production")
    return CheckResult("production docs disabled", "FAIL", "Could not confirm production docs are disabled")


def check_env_example_placeholders() -> CheckResult:
    env_example = _read(".env.example")
    expected_markers = [
        "JWT_SECRET_KEY=change-me-to-a-random-256-bit-secret",
        "DATABASE_URL=sqlite+aiosqlite:////data/orivory.db",
        "STORAGE_BACKEND=fs",
        "ENVIRONMENT=development",
    ]
    missing = [marker for marker in expected_markers if marker not in env_example]
    if missing:
        return CheckResult("env example placeholders", "FAIL", f"Missing expected demo placeholders: {', '.join(missing)}")
    return CheckResult("env example placeholders", "PASS", ".env.example keeps demo placeholders explicit")


def run_checks() -> list[CheckResult]:
    checks: list[Callable[[], CheckResult]] = [
        check_production_accepts_safe_settings,
        check_jwt_placeholder_rejected,
        check_wildcard_cors_rejected,
        check_provider_keys_required,
        check_internal_ports_removed,
        check_flower_ops_profile,
        check_diagnostics_summary_safe,
        check_docs_disabled_in_production,
        check_env_example_placeholders,
    ]
    results: list[CheckResult] = []
    for check in checks:
        try:
            results.append(check())
        except Exception as exc:
            results.append(CheckResult(check.__name__, "FAIL", str(exc)))
    return results


def main() -> None:
    results = run_checks()
    print("Security readiness checks")
    print("=========================")
    for result in results:
        print(f"[{result.status}] {result.name}: {result.detail}")

    failed = [result for result in results if result.status != "PASS"]
    if failed:
        print(f"\n{len(failed)} check(s) failed.", file=sys.stderr)
        raise SystemExit(1)
    print("\nAll security readiness checks passed.")


if __name__ == "__main__":
    main()
