"""The committed R26 dry-run artifact (`dry-run-report.json`).

The report is produced by `tests/migration/dry_run_p1b.py` against a COPY of
`eval/benchmarks/results/.system_run.db` (the original is never opened) with the
real production CLS embeddings. This test keeps the committed evidence honest:
it must describe a real run, and it must still show the migration fitting the
60-minute maintenance window the spec assumes.
"""
from __future__ import annotations

import json
from pathlib import Path

REPORT = Path(__file__).parent / "dry-run-report.json"


def test_dry_run_report_is_committed_evidence_of_the_window():
    report = json.loads(REPORT.read_text())

    assert report["source"]["opened"] is False, "the dry run must never open the source DB"
    assert report["runtime"]["embeddings"]["pooling"] == "cls"
    assert report["runtime"]["embeddings"]["dim"] == 384
    assert report["verify"]["memory"]["ok"] is True
    assert report["verify"]["chunk"]["ok"] is True
    assert report["backfill"]["memory"]["upserted"] == (
        report["dataset"]["eligible_memory_rows"]), "every eligible row was indexed"
    assert "blocked_intents" in report["cutover"]
    assert report["calibration"]["rows_per_sec"] > 0
    assert report["phase_totals"]["migration_seconds"] < (
        report["phase_totals"]["budget_minutes"] * 60)
    assert any(phase["name"] == "backfill_memory" for phase in report["phases"])
    assert all(tier["within_budget"] for tier in report["extrapolation"]["tiers"]), (
        "the measured throughput must hold the 60-minute window at every listed tier"
    )
