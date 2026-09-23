"""The pip-audit severity gate: what it decides, and that CI actually runs it.

`scripts/check_audit_severity.py` exists because the step it replaces grepped
`pip-audit -f json` for the word CRITICAL — but that JSON has no severity field
at all (``pip_audit._service.VulnerabilityResult``: id, description,
fix_versions, aliases, published), so the grep matched advisory prose: it missed
real criticals and could fail on a description. These tests drive the decision
with a canned OSV responder (no network) and pin the CI step to the script.

The last test is the one that keeps the fix from rotting: the fake gate must not
come back.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import urllib.error
from pathlib import Path

import pytest
import yaml

_spec = importlib.util.spec_from_file_location(
    "check_audit_severity",
    Path(__file__).resolve().parents[2] / "scripts" / "check_audit_severity.py",
)
assert _spec is not None and _spec.loader is not None
gate = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = gate
_spec.loader.exec_module(gate)

CI_YML = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"
AUDIT_STEP = "Check for critical vulnerabilities"


def _payload(*vulns: dict) -> dict:
    return {"dependencies": [{"name": "pkg", "version": "1.0", "vulns": list(vulns)}]}


def _responder(records: dict[str, dict]):
    """A fake OSV: ids in ``records`` answer, everything else 404s."""

    def fetch(url: str) -> dict:
        advisory_id = url.rsplit("/", 1)[-1]
        if advisory_id not in records:
            raise urllib.error.HTTPError(url, 404, "not found", {}, None)  # type: ignore[arg-type]
        return records[advisory_id]

    return fetch


def _run(tmp_path: Path, payload: dict, monkeypatch, responder) -> int:
    monkeypatch.setattr(gate, "fetch", responder)
    target = tmp_path / "pip-audit.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    return gate.main([str(target)])


def test_ids_come_from_ids_and_aliases_without_duplicates():
    payload = _payload(
        {"id": "GHSA-aaaa-bbbb-cccc", "aliases": ["CVE-2026-0001"]},
        {"id": "GHSA-aaaa-bbbb-cccc", "aliases": []},
    )
    assert gate.load_advisories(payload) == ["GHSA-aaaa-bbbb-cccc", "CVE-2026-0001"]


def test_a_critical_advisory_fails_the_gate(tmp_path, monkeypatch, capsys):
    record = {"id": "GHSA-aaaa-bbbb-cccc", "database_specific": {"severity": "CRITICAL"}}
    exit_code = _run(tmp_path, _payload({"id": "GHSA-aaaa-bbbb-cccc"}), monkeypatch,
                     _responder({"GHSA-aaaa-bbbb-cccc": record}))
    assert exit_code == 1
    assert "critical advisories by OSV (1)" in capsys.readouterr().out


def test_a_high_advisory_passes_but_is_still_counted(tmp_path, monkeypatch, capsys):
    record = {"id": "GHSA-aaaa-bbbb-cccc", "database_specific": {"severity": "HIGH"}}
    exit_code = _run(tmp_path, _payload({"id": "GHSA-aaaa-bbbb-cccc"}), monkeypatch,
                     _responder({"GHSA-aaaa-bbbb-cccc": record}))
    assert exit_code == 0
    assert "HIGH 1" in capsys.readouterr().out


def test_an_advisory_osv_does_not_know_is_unknown_not_safe(tmp_path, monkeypatch, capsys):
    exit_code = _run(tmp_path, _payload({"id": "GHSA-unknown-0000"}), monkeypatch, _responder({}))
    assert exit_code == 0
    assert "UNKNOWN 1" in capsys.readouterr().out  # never a silent pass


def test_an_unreachable_source_fails_closed(tmp_path, monkeypatch, capsys):
    def dead(url: str) -> dict:
        raise urllib.error.URLError("network is down")

    exit_code = _run(tmp_path, _payload({"id": "GHSA-aaaa-bbbb-cccc"}), monkeypatch, dead)
    assert exit_code == 1
    assert "could not resolve" in capsys.readouterr().err


def test_no_advisories_at_all_passes(tmp_path, monkeypatch, capsys):
    assert _run(tmp_path, _payload(), monkeypatch, _responder({})) == 0
    assert "no advisories" in capsys.readouterr().out


@pytest.mark.parametrize("bad", [[], ["a.json", "b.json"]])
def test_usage_error_is_not_a_pass(bad, capsys):
    assert gate.main(bad) == 2
    assert "usage:" in capsys.readouterr().err


def test_ci_runs_the_severity_script_and_not_a_grep():
    """The gate CI calls is the script — and the string grep must not return."""
    jobs = yaml.safe_load(CI_YML.read_text())["jobs"]
    steps = [step for job in jobs.values() for step in job.get("steps", [])]
    audit = [step for step in steps if step.get("name") == AUDIT_STEP]
    assert len(audit) == 1, f"expected exactly one {AUDIT_STEP!r} step, found {len(audit)}"
    run = audit[0]["run"]
    assert "scripts/check_audit_severity.py pip-audit.json" in run
    assert "grep -q 'CRITICAL'" not in run


# --- vectors, for the records OSV sends with no severity word ---------------
# Some OSV records carry severity only as a top-level CVSS v3 vector, so a gate
# that reads database_specific.severity alone calls them UNKNOWN and lets a 9.8
# through. The vector below is the real one on CVE-2020-14343, which has no
# database_specific.severity (verified live against api.osv.dev).
CVE_2020_14343_VECTOR = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"


@pytest.mark.parametrize(
    ("vector", "score"),
    [
        (CVE_2020_14343_VECTOR, 9.8),
        ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H", 10.0),  # scope-changed branch (log4shell)
        ("CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H", 7.8),
        ("CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:L/I:L/A:N", 5.4),
        ("CVSS:3.1/AV:N/AC:H/PR:N/UI:R/S:U/C:L/I:N/A:N", 3.1),
    ],
)
def test_cvss_v3_base_scores_match_published_values(vector, score):
    assert gate.cvss_v3_base_score(vector) == pytest.approx(score)


def test_a_vector_with_a_critical_score_fails_the_gate(tmp_path, monkeypatch, capsys):
    """The CVE-2020-14343 shape end to end: no severity word, only the 9.8 vector."""
    record = {"id": "CVE-2020-14343", "severity": [{"type": "CVSS_V3", "score": CVE_2020_14343_VECTOR}]}
    exit_code = _run(tmp_path, _payload({"id": "CVE-2020-14343"}), monkeypatch,
                     _responder({"CVE-2020-14343": record}))
    assert exit_code == 1
    assert "critical advisories by OSV (1): CVE-2020-14343" in capsys.readouterr().out


def test_a_vector_with_a_medium_score_is_counted_but_does_not_block(tmp_path, monkeypatch, capsys):
    record = {"severity": [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:L/I:L/A:N"}]}
    exit_code = _run(tmp_path, _payload({"id": "GHSA-aaaa-bbbb-cccc"}), monkeypatch,
                     _responder({"GHSA-aaaa-bbbb-cccc": record}))
    assert exit_code == 0
    assert "MEDIUM 1" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("vtype", "vector"),
    [
        ("CVSS_V4", "CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N"),
        ("CVSS_V2", "AV:N/AC:L/Au:N/C:P/I:P/A:P"),
    ],
)
def test_a_v2_or_v4_vector_stays_unknown(tmp_path, monkeypatch, capsys, vtype, vector):
    """Unscoreable vectors must surface as UNKNOWN, never as a silent pass."""
    record = {"severity": [{"type": vtype, "score": vector}]}
    exit_code = _run(tmp_path, _payload({"id": "GHSA-aaaa-bbbb-cccc"}), monkeypatch,
                     _responder({"GHSA-aaaa-bbbb-cccc": record}))
    assert exit_code == 0
    assert "UNKNOWN 1" in capsys.readouterr().out


def test_a_malformed_v3_vector_stays_unknown(tmp_path, monkeypatch, capsys):
    record = {"severity": [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:Z/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"}]}
    exit_code = _run(tmp_path, _payload({"id": "GHSA-aaaa-bbbb-cccc"}), monkeypatch,
                     _responder({"GHSA-aaaa-bbbb-cccc": record}))
    assert exit_code == 0
    assert "UNKNOWN 1" in capsys.readouterr().out
