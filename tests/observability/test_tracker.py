"""Tests for experiment tracking, cost tracking, and artifact store."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from app.observability.artifacts import ArtifactStore, sha256_text
from app.observability.cost import (
    CallCost,
    CostTracker,
    budget_window_iso,
    calculate_cost,
    estimate_max_budget_alert,
)
from app.observability.experiments import Experiment, Variant
from app.observability.tracker import RunTracker, stopwatch

# ---------------------------------------------------------------------------
# RunTracker
# ---------------------------------------------------------------------------


class TestRunTracker:
    def test_start_end_run(self, tmp_path: Path) -> None:
        db = tmp_path / "runs.db"
        tracker = RunTracker(db_path=db)
        with tracker.start_run("test_run", tags={"phase": "smoke"}) as run:
            tracker.log_params({"top_k": 5, "model": "gpt-4o-mini"})
            tracker.log_metrics({"source_hit_rate": 0.9, "latency_ms": 420.0})
        run_after = tracker.get_run(run["run_id"])
        assert run_after is not None
        assert run_after["status"] == "finished"
        assert tracker.get_params(run["run_id"]) == {"top_k": 5, "model": "gpt-4o-mini"}
        metrics = tracker.get_metrics(run["run_id"])
        assert metrics["source_hit_rate"] == pytest.approx(0.9)
        assert metrics["latency_ms"] == pytest.approx(420.0)

    def test_list_runs(self, tmp_path: Path) -> None:
        tracker = RunTracker(db_path=tmp_path / "r.db")
        for i in range(3):
            with tracker.start_run(f"run_{i}"):
                tracker.log_metrics({"x": float(i)})
        runs = tracker.list_runs()
        assert len(runs) == 3
        names = {r["name"] for r in runs}
        assert names == {"run_0", "run_1", "run_2"}

    def test_compare_runs(self, tmp_path: Path) -> None:
        tracker = RunTracker(db_path=tmp_path / "r.db")
        ids = []
        for i, score in enumerate([0.7, 0.85, 0.95]):
            with tracker.start_run(f"v_{i}") as run:
                tracker.log_params({"top_k": i + 3})
                tracker.log_metrics({"source_hit_rate": score})
                ids.append(run["run_id"])
        comp = tracker.compare_runs(ids, metrics=["source_hit_rate", "latency_ms"])
        assert len(comp) == 3
        for rid in ids:
            assert "source_hit_rate" in comp[rid]["metrics"]
            # latency_ms not logged in any run, but ensure key exists with None
            assert comp[rid]["metrics"]["latency_ms"] is None

    def test_failed_run_marked(self, tmp_path: Path) -> None:
        tracker = RunTracker(db_path=tmp_path / "r.db")
        with pytest.raises(RuntimeError):
            with tracker.start_run("bad") as run:
                raise RuntimeError("boom")
        run = tracker.get_run(run["run_id"])
        assert run["status"] == "failed"

    def test_metrics_step(self, tmp_path: Path) -> None:
        tracker = RunTracker(db_path=tmp_path / "r.db")
        with tracker.start_run("stepped") as run:
            tracker.log_metrics({"loss": 0.5}, step=0)
            tracker.log_metrics({"loss": 0.3}, step=1)
        with sqlite3.connect(tmp_path / "r.db") as conn:
            rows = conn.execute(
                "SELECT step, value FROM metrics WHERE run_id=? ORDER BY step",
                (run["run_id"],),
            ).fetchall()
        assert [(r[0], r[1]) for r in rows] == [(0, 0.5), (1, 0.3)]


class TestStopwatch:
    def test_measures_time(self) -> None:
        with stopwatch() as sw:
            sum(range(10_000))
        assert "total_ms" in sw
        assert sw["total_ms"] >= 0


# ---------------------------------------------------------------------------
# ArtifactStore
# ---------------------------------------------------------------------------


class TestArtifactStore:
    def test_save_and_load_text(self, tmp_path: Path) -> None:
        store = ArtifactStore(root=tmp_path / "art")
        meta = store.save_text("hello.txt", "hello world", kind="text")
        assert meta["name"] == "hello.txt"
        assert meta["sha256"] == sha256_text("hello world")
        assert Path(meta["path"]).exists()
        assert store.load_text(meta["sha256"]) == "hello world"

    def test_save_json(self, tmp_path: Path) -> None:
        store = ArtifactStore(root=tmp_path / "art")
        meta = store.save_json("config", {"k": 1, "v": 2})
        loaded = json.loads(store.load_text(meta["sha256"]))
        assert loaded == {"k": 1, "v": 2}

    def test_copy_file(self, tmp_path: Path) -> None:
        src = tmp_path / "src.txt"
        src.write_text("payload", encoding="utf-8")
        store = ArtifactStore(root=tmp_path / "art")
        meta = store.copy_file(src, kind="file")
        assert Path(meta["path"]).read_text(encoding="utf-8") == "payload"

    def test_list_artifacts(self, tmp_path: Path) -> None:
        store = ArtifactStore(root=tmp_path / "art")
        store.save_text("a", "1", kind="text")
        store.save_text("b", "2", kind="json")
        items = store.list_artifacts()
        assert len(items) == 2
        text_items = store.list_artifacts(kind="text")
        assert len(text_items) == 1


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


class TestCostCalculation:
    def test_known_model(self) -> None:
        cost = calculate_cost("openai/gpt-4o-mini", 1000, 500)
        # 0.001 * 0.15 + 0.0005 * 0.60 = 0.00015 + 0.0003 = 0.00045
        assert cost == pytest.approx(0.00045)

    def test_unknown_model_is_zero(self) -> None:
        assert calculate_cost("unknown/model", 1000, 500) == 0.0


class TestCostTracker:
    def test_record_and_total(self, tmp_path: Path) -> None:
        tracker = CostTracker(db_path=tmp_path / "c.db")
        c1 = tracker.record("router", "openai/gpt-4o-mini", 1000, 500, user_id="u1")
        c2 = tracker.record("answer", "openai/gpt-4o-mini", 2000, 800, user_id="u1")
        assert isinstance(c1, CallCost)
        assert tracker.total() == pytest.approx(c1.cost_usd + c2.cost_usd)

    def test_breakdown_by_agent(self, tmp_path: Path) -> None:
        tracker = CostTracker(db_path=tmp_path / "c.db")
        tracker.record("router", "openai/gpt-4o-mini", 1000, 500)
        tracker.record("router", "openai/gpt-4o-mini", 1000, 500)
        tracker.record("answer", "openai/gpt-4o-mini", 2000, 800)
        breakdown = tracker.breakdown_by_agent()
        assert breakdown["router"]["calls"] == 2
        assert breakdown["answer"]["calls"] == 1
        assert breakdown["router"]["total_cost_usd"] > 0

    def test_recent(self, tmp_path: Path) -> None:
        tracker = CostTracker(db_path=tmp_path / "c.db")
        for _i in range(5):
            tracker.record("agent", "openai/gpt-4o-mini", 100, 50)
        recent = tracker.recent(limit=3)
        assert len(recent) == 3

    def test_total_since(self, tmp_path: Path) -> None:
        tracker = CostTracker(db_path=tmp_path / "c.db")
        tracker.record("agent", "openai/gpt-4o-mini", 100, 50)
        # 1 hour ago — should include the record
        assert tracker.total(since_iso=budget_window_iso(hours=1)) > 0
        # 1 day in the future — should include nothing
        future = budget_window_iso(hours=-24)
        assert tracker.total(since_iso=future) == 0.0

    def test_wal_mode_and_busy_timeout(self, tmp_path: Path) -> None:
        # The ledger must use WAL + busy_timeout: a threading.Lock only works
        # in-process; multi-worker servers would hit 'database is locked'.
        tracker = CostTracker(db_path=tmp_path / "costs.db")
        tracker.record(agent="t", model="openai/gpt-4o-mini", tokens_in=10, tokens_out=5)
        conn = sqlite3.connect(tmp_path / "costs.db")
        try:
            journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
            busy_timeout = conn.execute("PRAGMA busy_timeout;").fetchone()[0]
        finally:
            conn.close()
        assert journal_mode.lower() == "wal"
        assert busy_timeout >= 5000


class TestBudgetAlert:
    def test_under_budget(self) -> None:
        assert estimate_max_budget_alert(0.5, 1.0) is None

    def test_80_percent(self) -> None:
        msg = estimate_max_budget_alert(0.85, 1.0)
        assert msg is not None
        assert "80%" in msg

    def test_exceeded(self) -> None:
        msg = estimate_max_budget_alert(1.5, 1.0)
        assert msg is not None
        assert "exceeded" in msg.lower()

    def test_zero_budget_no_alert(self) -> None:
        assert estimate_max_budget_alert(100, 0) is None


# ---------------------------------------------------------------------------
# Experiment (integration with offline eval) — uses a tiny dataset
# ---------------------------------------------------------------------------


class TestExperiment:
    def test_run_sweep(self, tmp_path: Path) -> None:
        # Use the existing eval/sample_docs + a mini dataset
        from pathlib import Path as P

        root = P(__file__).resolve().parents[2]
        sample_docs = root / "sample_docs"
        if not sample_docs.exists():  # pragma: no cover - skip if env lacks docs
            pytest.skip("sample_docs not available")
        dataset = root / "eval" / "orivory_eval_dataset.json"
        exp = Experiment(
            name="topk_sweep_test",
            dataset_path=dataset,
            sample_docs_dir=sample_docs,
            output_dir=tmp_path / "out",
            tracker_db=tmp_path / "exp.db",
            primary_metric="source_hit_rate",
        )
        exp.add_variants(
            [
                Variant(name="topk_3", params={"top_k": 3}),
                Variant(name="topk_5", params={"top_k": 5}),
            ]
        )
        result = exp.run(enable_ragas=False)
        assert len(result.runs) == 2
        assert result.best_run_id is not None
        # best_run_id corresponds to one of our variants
        assert any(r["run_id"] == result.best_run_id for r in result.runs)
