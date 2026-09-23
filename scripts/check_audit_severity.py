#!/usr/bin/env python3
"""Fail CI when an advisory pip-audit reported is rated CRITICAL by OSV.

``pip-audit -f json`` carries ids, descriptions, fix versions and aliases — and
NO severity (``pip_audit._service.VulnerabilityResult`` has no such field). A
``grep 'CRITICAL'`` over that JSON therefore matches advisory prose at best: it
misses real criticals and can fail on a word in a description. Severity lives in
the advisory database, so every id pip-audit reported is resolved against OSV
(free, no key) and the advisory's own severity decides the verdict.

Fail-closed by design: an id OSV does not know (404) counts as unknown and does
not block, but a request that fails for any other reason exits non-zero — a gate
that cannot reach its source has not run. The unknown count is always printed,
so the gate never reads greener than it is.

Usage: check_audit_severity.py <pip-audit.json>
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

OSV_URL = "https://api.osv.dev/v1/vulns/{}"
TIMEOUT_SECONDS = 20.0


def load_advisories(payload: dict | list) -> list[str]:
    """Every advisory id pip-audit reported: ids AND aliases, de-duplicated."""
    entries = payload.get("dependencies", []) if isinstance(payload, dict) else payload
    ids: list[str] = []
    for dependency in entries:
        for vuln in dependency.get("vulns") or []:
            for value in (vuln.get("id"), *(vuln.get("aliases") or [])):
                if value and value not in ids:
                    ids.append(str(value))
    return ids


def severity_of(record: dict) -> str | None:
    """The advisory's own severity word (GHSA-style), when the record carries one."""
    severity = (record.get("database_specific") or {}).get("severity")
    return str(severity).upper() if severity else None


def fetch(url: str) -> dict:
    """The one network call; patched by the tests."""
    with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode())


def resolve(advisory_id: str) -> str | None:
    """OSV's severity for one id — a word, or None when OSV has no record."""
    try:
        return severity_of(fetch(OSV_URL.format(advisory_id)))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:  # not in OSV: counted unknown, never assumed safe
            return None
        raise


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: check_audit_severity.py <pip-audit.json>", file=sys.stderr)
        return 2
    with open(argv[0], encoding="utf-8") as handle:
        advisory_ids = load_advisories(json.load(handle))

    counts: dict[str, int] = {}
    critical: list[str] = []
    for advisory_id in advisory_ids:
        try:
            severity = resolve(advisory_id)
        except Exception as exc:  # no source, no verdict — never a silent pass
            print(f"could not resolve {advisory_id} against OSV: {exc!r}", file=sys.stderr)
            return 1
        counts[severity or "UNKNOWN"] = counts.get(severity or "UNKNOWN", 0) + 1
        if severity == "CRITICAL":
            critical.append(advisory_id)

    summary = ", ".join(f"{name} {count}" for name, count in sorted(counts.items()))
    summary = summary or "no advisories"
    if critical:
        print(f"critical advisories by OSV ({len(critical)}): {', '.join(critical)}")
        print(f"severity across {len(advisory_ids)} advisory id(s): {summary}")
        return 1
    print(f"no critical advisories; severity across {len(advisory_ids)} advisory id(s): {summary}")
    return 0


if __name__ == "__main__":  # pragma: no cover - main() is what the tests drive
    raise SystemExit(main())
