"""Reliability options for Celery tasks + cost tracker.

Regression net for the reliability review:
- LLM-touching tasks must have time limits (a hung LLM stream must not
  pin a worker slot forever).
- All retrying tasks must use backoff+jitter (fixed delays cause retry
  storms when a dependency flaps).
- The cost ledger must use WAL + busy_timeout (threading.Lock only works
  in-process; multi-worker gunicorn/uvicorn would hit 'database is locked').
"""
import sqlite3

import pytest


class TestTaskTimeLimits:
    def test_process_document_has_time_limits(self):
        from app.tasks import ingestion_tasks

        task = ingestion_tasks.process_document
        assert task.time_limit is not None
        assert task.soft_time_limit is not None
        assert task.soft_time_limit < task.time_limit

    def test_build_memory_graph_has_time_limits(self):
        from app.tasks import graph_tasks

        task = graph_tasks.build_memory_graph_task
        assert task.time_limit is not None
        assert task.soft_time_limit is not None
        assert task.soft_time_limit < task.time_limit


class TestRetryBackoffJitter:
    @pytest.mark.parametrize(
        "module,task_name",
        [
            ("app.tasks.ingestion_tasks", "process_document"),
            ("app.tasks.graph_tasks", "build_memory_graph_task"),
            ("app.tasks.reindex_tasks", "reindex_user_memories"),
            ("app.tasks.email_tasks", "send_verification_email"),
            ("app.tasks.email_tasks", "send_password_reset_email"),
        ],
    )
    def test_task_retries_with_backoff_and_jitter(self, module, task_name):
        import importlib

        task = getattr(importlib.import_module(module), task_name)
        assert task.retry_backoff is True
        assert task.retry_jitter is True


class TestCostTrackerConcurrency:
    def test_wal_mode_and_busy_timeout(self, tmp_path):
        from app.observability.cost import CostTracker

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
