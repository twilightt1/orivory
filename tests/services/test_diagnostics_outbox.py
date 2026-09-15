"""Task 6 — the outbox is OPERABLE: diagnostics summary, fallback counter, CI.

Three faces, one file:

- ``get_index_outbox_summary`` — counts by status AND by kind, the stuck-pending
  bucket and the oldest pending intent. ``blocked`` is part of the summary on
  purpose (carried item C2): a terminally blocked intent never lands, and it is
  not pending either, so the recall freshness barrier has nothing to wait for —
  a recall can answer ``200 []`` for a write that will never be indexed. A
  pending-only number would read as "still in flight" forever.
- the drain-failure fallback counter (ruling R24): a ``run_drain_loop`` round
  that raises counts ``index.outbox_drain_failed`` — once per failed round — in
  the existing ``app/observability/fallbacks.py`` registry.
- the CI gap (ruling R25): the P3 suites the brief lists run in a step of
  ``.github/workflows/ci.yml``. The gate reads the workflow as YAML and checks
  the step's name and the modules it runs, so deleting the step — or dropping
  one suite from it — fails here instead of silently narrowing CI.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from app.models.index_outbox import IndexOutbox
from app.observability import fallbacks
from app.retrieval.memory import drain_loop
from app.services import diagnostics_service
from app.services.diagnostics_service import build_diagnostics, get_index_outbox_summary

pytestmark = pytest.mark.service

CI_YML = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"
CI_STEP_NAME = "Run P3 background-indexing suites (temp SQLite, no services)"
DRAIN_FAILED = "index.outbox_drain_failed"
# The suites that were outside CI before Task 6, plus this file.
P3_SUITES = (
    "tests/services/test_erasure_service.py",
    "tests/services/test_diagnostics_outbox.py",
    "tests/services/test_erasure_reconcile.py",
    "tests/lite/test_sqlite_schema_v3.py",
    "tests/lite/test_sqlite_sync_parity.py",
    "tests/lite/test_correction_roundtrip.py",
    "tests/lite/test_sqlite_bootstrap.py",
    "tests/retrieval/test_drain_loop.py",
    "tests/retrieval/test_freshness_barrier.py",
    "tests/retrieval/test_drain_races.py",
    "tests/rag/test_pipeline_purge_fence.py",
)


def _intent(*, tenant: uuid.UUID, kind: str = "memory", status: str = "pending",
            created_at: datetime | None = None, last_error: str | None = None) -> IndexOutbox:
    row = IndexOutbox(
        kind=kind,
        entity_id=uuid.uuid4().hex,
        tenant_id=tenant.hex,
        revision=1,
        operation="upsert",
        target_generation="orivory_memories",
        status=status,
        last_error=last_error,
    )
    if created_at is not None:
        row.created_at = created_at
    return row


# ── the summary: every class is visible, including the terminal one ─────────


async def test_the_summary_counts_statuses_kinds_the_stuck_bucket_and_the_oldest(db):
    """3 pending — one older than the stuck threshold — and 1 terminally blocked."""
    tenant = uuid.uuid4()
    oldest_pending = datetime.now(UTC) - timedelta(minutes=20)
    db.add_all(
        [
            _intent(tenant=tenant, created_at=oldest_pending),
            _intent(tenant=tenant),
            _intent(tenant=tenant, kind="chunk"),
            _intent(tenant=tenant, status="blocked", last_error="contract mismatch"),
        ]
    )
    await db.commit()

    summary = await get_index_outbox_summary(db)

    assert summary["by_status"] == {"pending": 3, "done": 0, "blocked": 1}
    assert summary["by_kind"] == {"memory": 3, "chunk": 1}
    assert summary["stuck_pending"] == 1  # the 20-minute-old one, not the two fresh
    returned = datetime.fromisoformat(summary["oldest_pending_at"])
    assert returned.replace(tzinfo=UTC) == oldest_pending


async def test_the_summary_of_an_empty_outbox_is_zeros_not_missing_keys(db):
    summary = await get_index_outbox_summary(db)

    assert summary["by_status"] == {"pending": 0, "done": 0, "blocked": 0}
    assert summary["by_kind"] == {"memory": 0, "chunk": 0}
    assert summary["stuck_pending"] == 0
    assert summary["oldest_pending_at"] is None


async def test_build_diagnostics_carries_the_summary_to_the_admin_payload(db, monkeypatch):
    """/admin/diagnostics returns ``build_diagnostics`` — the blocked class rides along."""

    async def fake_readiness(extra_checkers=None):
        return {"postgres": {"status": "ok", "latency_ms": 1.0}}

    monkeypatch.setattr(diagnostics_service, "run_readiness_checks", fake_readiness)
    db.add(_intent(tenant=uuid.uuid4(), status="blocked", last_error="generation superseded"))
    await db.commit()

    payload = await build_diagnostics(db)

    assert payload["index_outbox"]["by_status"]["blocked"] == 1
    assert payload["index_outbox"]["by_status"]["pending"] == 0


# ── the drain-failure fallback counter (R24) ────────────────────────────────


def test_the_drain_failure_path_is_named_in_the_fallback_registry():
    """Dashboards alert on the canonical path list — it is documented there."""
    registry = fallbacks.__doc__ or ""
    assert DRAIN_FAILED in registry


async def test_a_failing_drain_round_counts_the_fallback_each_round(monkeypatch):
    """One count per failed round — not one per loop, not zero."""
    before = fallbacks.fallback_counts().get(DRAIN_FAILED, 0)
    rounds: list[int] = []

    async def _boom(*, batch_size):
        rounds.append(batch_size)
        raise RuntimeError("outbox unavailable")

    monkeypatch.setattr(drain_loop, "drain_pending", _boom)
    stop = asyncio.Event()
    task = asyncio.create_task(drain_loop.run_drain_loop(interval=0.05, batch_size=9, stop=stop))
    try:
        deadline = asyncio.get_running_loop().time() + 5.0
        while len(rounds) < 2:
            assert asyncio.get_running_loop().time() < deadline, "the loop never ran a round"
            await asyncio.sleep(0.01)
        assert not task.done()  # a failed round does not end the loop
    finally:
        stop.set()
        await asyncio.wait_for(task, timeout=5)

    assert fallbacks.fallback_counts()[DRAIN_FAILED] - before == 2
    assert rounds == [9, 9]  # the batch size still travels with every round


# ── the CI gap (R25) ───────────────────────────────────────────────────────


def _ci_step() -> dict:
    steps = yaml.safe_load(CI_YML.read_text())["jobs"]["test"]["steps"]
    matches = [step for step in steps if step.get("name") == CI_STEP_NAME]
    assert len(matches) == 1, [step.get("name") for step in steps]
    return matches[0]


def test_ci_runs_the_p3_background_indexing_suites():
    step = _ci_step()
    run = step["run"]

    missing = [suite for suite in P3_SUITES if suite not in run]
    assert missing == []
    # Same env discipline as the neighbouring temp-SQLite steps.
    assert step["env"]["DATABASE_URL"].startswith("sqlite+aiosqlite:///")
