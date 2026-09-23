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
