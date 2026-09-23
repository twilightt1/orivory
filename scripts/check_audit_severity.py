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

Some OSV records carry no severity word at all — only a top-level CVSS v3 base
vector. Those are scored with the CVSS v3.1 base formula (spec §7.1) and mapped
to the v3.1 rating scale; vectors this script cannot score (v2, v4) stay
UNKNOWN, which is printed, never silently treated as safe.

Usage: check_audit_severity.py <pip-audit.json>
"""
from __future__ import annotations

import json
import math
import sys
import urllib.error
import urllib.request

OSV_URL = "https://api.osv.dev/v1/vulns/{}"
TIMEOUT_SECONDS = 20.0

# CVSS v3.1 base metric weights (spec §7.4). PR depends on scope, so it is keyed
# by scope; everything else is a single lookup.
_V3_WEIGHT = {
    "AV": {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2},
    "AC": {"L": 0.77, "H": 0.44},
    "UI": {"N": 0.85, "R": 0.62},
    "PR": {"U": {"N": 0.85, "L": 0.62, "H": 0.27}, "C": {"N": 0.85, "L": 0.68, "H": 0.5}},
    "C": {"H": 0.56, "L": 0.22, "N": 0.0},
    "I": {"H": 0.56, "L": 0.22, "N": 0.0},
    "A": {"H": 0.56, "L": 0.22, "N": 0.0},
}


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


def word_for_score(score: float) -> str:
    """CVSS v3.1 qualitative severity rating scale."""
    if score >= 9.0:
        return "CRITICAL"
    if score >= 7.0:
        return "HIGH"
    if score >= 4.0:
        return "MEDIUM"
    if score > 0:
        return "LOW"
    return "NONE"  # CVSS v3.1 rates 0.0 as NONE; LOW starts at 0.1


def cvss_v3_base_score(vector: str) -> float | None:
    """Base score for a CVSS v3.x base vector, or None when it cannot be scored.

    Only base metrics count (temporal/environmental metrics after them are
    ignored); a missing or unknown metric yields None, never a guess.
    """
    parts = vector.split("/")
    version = parts[0].upper() if parts else ""
    if len(parts) < 2 or not version.startswith("CVSS:3."):
        return None  # v2 vectors have no prefix, v4 is a different formula
    metrics = dict(part.split(":", 1) for part in parts[1:] if ":" in part)
    try:
        scope = metrics["S"].upper()
        weight = {key: metric[metrics[key].upper()] for key, metric in _V3_WEIGHT.items() if key != "PR"}
        weight["PR"] = _V3_WEIGHT["PR"][scope][metrics["PR"].upper()]
        exploitability = 8.22 * weight["AV"] * weight["AC"] * weight["PR"] * weight["UI"]
        impact_sub = 1 - (1 - weight["C"]) * (1 - weight["I"]) * (1 - weight["A"])
    except KeyError:
        return None
    if impact_sub <= 0:
        return 0.0
    if scope == "U":
        impact = 6.42 * impact_sub
        multiplier = 1.0
    else:
        # The changed-scope impact expression is the one thing v3.1 changed in
        # the base formula: v3.0 raises (ISS - 0.02) to 15, v3.1 raises
        # (ISS * 0.9731 - 0.02) to 13. Scoring a v3.1 vector with v3.0's
        # expression moves 5 of the 1296 Scope:Changed vectors across a band
        # boundary (HIGH <-> MEDIUM, never CRITICAL), so the vector's own
        # version decides which expression is used.
        if version == "CVSS:3.0":
            impact = 7.52 * (impact_sub - 0.029) - 3.25 * (impact_sub - 0.02) ** 15
        else:
            impact = 7.52 * (impact_sub - 0.029) - 3.25 * (impact_sub * 0.9731 - 0.02) ** 13
        if impact <= 0:
            return 0.0
        multiplier = 1.08  # the changed-scope multiplier, unchanged since v3.0
    # Spec §7.1: round up to one decimal, capped at 10.
    scaled = round(min((impact + exploitability) * multiplier, 10.0) * 100000)
    return scaled / 100000 if scaled % 10000 == 0 else (math.floor(scaled / 10000) + 1) / 10


def severity_of(record: dict) -> str | None:
    """The advisory's severity — its own word, else scored from a CVSS v3 vector."""
    severity = (record.get("database_specific") or {}).get("severity")
    if severity:
        return str(severity).upper()
    for entry in record.get("severity") or []:
        if str(entry.get("type") or "").upper() == "CVSS_V3":
            score = cvss_v3_base_score(str(entry.get("score") or ""))
            if score is not None:
                return word_for_score(score)
    return None  # no word, no scorable vector: UNKNOWN, which is printed


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
