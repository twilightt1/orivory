"""The committed R26 dry-run artifact (`dry-run-report.json`).

The report is produced by `tests/migration/dry_run_p1b.py` against a COPY of
`eval/benchmarks/results/.system_run.db` (the original is never opened) with the
real production CLS embeddings. These tests keep the committed evidence honest:
it must describe a real run, and it must keep stating where the 60-minute
maintenance window STOPS holding — the 100k-row tier at a realistic ~1000 chars
per row does not fit, and that bound must not be softened away (F4).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPORT = Path(__file__).parent / "dry-run-report.json"


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(REPORT.read_text())


def test_dry_run_report_is_committed_evidence_of_a_real_run(report):
    assert report["source"]["opened"] is False, "the dry run must never open the source DB"
    assert report["runtime"]["embeddings"]["pooling"] == "cls"
    assert report["runtime"]["embeddings"]["dim"] == 384
    assert report["verify"]["memory"]["ok"] is True
    assert report["verify"]["chunk"]["ok"] is True
    assert report["backfill"]["memory"]["upserted"] == (
        report["dataset"]["eligible_memory_rows"]), "every eligible row was indexed"
    assert "blocked_intents" in report["cutover"]
    assert report["phase_totals"]["migration_seconds"] < (
        report["phase_totals"]["budget_minutes"] * 60)
    assert any(phase["name"] == "backfill_memory" for phase in report["phases"])
    # The flip is a real pointer move: the old generation served until cutover.
    assert report["cutover"]["active"]["memory"] == report["backfill"]["memory"]["generation"]


def test_dry_run_report_pins_the_honest_bound(report):
    """The measured rate, and the tier where it stops fitting the budget."""
    budget_minutes = report["phase_totals"]["budget_minutes"]
    extrapolation = report["extrapolation"]
    calibration = report["calibration"]

    # Headline calibration: the per-character rate the projection rests on, and
    # the row count it holds inside the window at a realistic memory size.
    assert calibration["rows"] >= 200 and calibration["seconds"] > 0
    assert extrapolation["seconds_per_row"] == pytest.approx(
        1.0 / calibration["rows_per_sec"], abs=1e-5)
    at_1000 = extrapolation["max_eligible_rows_within_budget_at_1000_chars"]
    assert at_1000 == pytest.approx(
        budget_minutes * 60 * calibration["chars_per_sec"] / 1000, abs=1.5
    ), "the headline figure is the measured rate, not a wish"
    assert 0 < at_1000 < 100_000

    # The char-blind tiers all fit at the copy's tiny rows...
    assert all(tier["within_budget"] for tier in extrapolation["tiers"])
    # ...and that is NOT the useful tier: at ~1000 chars per row, 100k eligible
    # rows blow the 60-minute window. Pinned so the artifacts cannot quietly
    # start claiming otherwise.
    tiers = extrapolation["tiers_at_1000_chars"]
    assert [tier["eligible_rows"] for tier in tiers] == [100, 1_000, 10_000, 100_000]
    biggest = tiers[-1]
    assert biggest["within_budget"] is False
    assert biggest["projected_minutes"] > budget_minutes
    assert biggest["projected_minutes"] == pytest.approx(
        100_000 * extrapolation["seconds_per_1000_chars"] / 60, abs=1e-3)


def test_dataset_char_counts_are_labelled_honestly(report):
    """`chars_in_eligible_memories` counts the eligible rows only (F4)."""
    dataset = report["dataset"]
    assert 0 < dataset["chars_in_eligible_memories"] <= dataset["chars_in_all_memories"]
    if dataset["eligible_memory_rows"] == dataset["memory_rows"]:
        assert dataset["chars_in_eligible_memories"] == dataset["chars_in_all_memories"]
    else:
        assert dataset["chars_in_eligible_memories"] < dataset["chars_in_all_memories"]
