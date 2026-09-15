#!/usr/bin/env python
"""R26 dry run: time the P1b migration on a COPY of a real database.

Runs the migration CLI end to end — inventory, backup, keyset backfill for both
kinds, verify for both kinds, cutover — against a COPY of the benchmark
database that ships in the repo (``eval/benchmarks/results/.system_run.db``) and
a private embedded Qdrant folder, with the REAL production CLS embeddings
(arctic XS on onnxruntime). The original database is never opened: the copy,
its WAL and its SHM are the only inputs.

It writes ``dry-run-report.json`` next to this file: per-step wall-clock,
measured embedding throughput, and the extrapolation to other row counts that
proves (or refutes) the ≤60-minute maintenance window assumed by spec §6.2.

    .venv/bin/python tests/migration/dry_run_p1b.py [--source DB] [--out JSON]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

DEFAULT_SOURCE = REPO_ROOT / "eval" / "benchmarks" / "results" / ".system_run.db"
BUDGET_MINUTES = 60
TIER_ROWS = (100, 1_000, 10_000, 100_000)


def _pin_environment(source: Path, workdir: Path) -> Path:
    """Copy the database aside and point ``settings`` at the copy.

    Everything here happens BEFORE the app modules are imported: settings (and
    the engines built from them) are read at import time.
    """
    copy_dir = workdir / "copy"
    copy_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(f"{source}{suffix}")
        if candidate.exists():
            shutil.copy2(candidate, copy_dir / candidate.name)
    db_copy = copy_dir / source.name
    if not db_copy.exists():
        raise SystemExit(f"could not copy {source} into {copy_dir}")
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_copy}"
    os.environ["QDRANT_MODE"] = "local"
    os.environ["QDRANT_LOCAL_PATH"] = str(workdir / "qdrant")
    os.environ["USE_LOCAL_EMBEDDINGS"] = "true"
    os.environ["LOCAL_EMBED_MODEL"] = "arctic"
    os.environ["STORAGE_BACKEND"] = "fs"
    os.environ["FS_STORAGE_PATH"] = str(workdir / "uploads")
    os.environ["APP_PORT"] = "1"  # nothing listens on port 1: quiesce passes
    return db_copy


def _checkpoint_copy(db_copy: Path) -> None:
    """Fold the copy's WAL into the main file so the read-only audit sees it.

    Part of the dry run's PREPARE step, not of any migration phase: a real
    operator's backup does the same before handing the copy to ``inventory``.
    """
    conn = sqlite3.connect(db_copy)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P1b migration dry run (R26)")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="database to COPY (never opened)")
    parser.add_argument("--workdir", type=Path, default=None,
                        help="scratch directory (default: a fresh temp dir)")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "dry-run-report.json")
    parser.add_argument("--batch", type=int, default=200, help="backfill batch size")
    parser.add_argument("--calibration-rows", type=int, default=200,
                        help="rows to embed for a stable throughput figure")
    args = parser.parse_args(argv)

    source = Path(args.source)
    if not source.exists():
        print(f"source database not found: {source}", file=sys.stderr)
        return 2
    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="p1b-dryrun-"))
    workdir.mkdir(parents=True, exist_ok=True)
    db_copy = _pin_environment(source, workdir)

    import migrate_qdrant as cli

    from app.config import settings
    from app.retrieval import vector_backend
    from app.retrieval.embedder import embed_texts_sync

    phases: list[dict] = []

    def timed(name: str, fn):
        started = time.perf_counter()
        result = fn()
        phases.append({"name": name, "seconds": round(time.perf_counter() - started, 3)})
        return result

    started_at = datetime.now(UTC)
    prepare_started = time.perf_counter()
    _checkpoint_copy(db_copy)
    prepare_seconds = round(time.perf_counter() - prepare_started, 3)

    inventory = timed("inventory", lambda: cli.inventory(out=workdir / "inventory.json"))
    backup = timed("backup", lambda: cli.backup(dest_dir=workdir / "backup"))
    memory_backfill = timed("backfill_memory",
                            lambda: cli.backfill(kind="memory", batch=args.batch))
    chunk_backfill = timed("backfill_chunk",
                           lambda: cli.backfill(kind="chunk", batch=args.batch))
    memory_verify = timed("verify_memory", lambda: cli.verify(kind="memory"))
    chunk_verify = timed("verify_chunk", lambda: cli.verify(kind="chunk"))
    cutover = timed("cutover", lambda: cli.cutover(yes=True))

    # Calibration on the same corpus/contract the migration just used: the
    # per-row cost is what the extrapolation must rest on, and the benchmark
    # copy is far too small to time a rate from directly.
    with sqlite3.connect(f"file:{db_copy}?mode=ro", uri=True) as conn:
        texts_by_id = {
            row[0]: row[1] for row in conn.execute(
                "SELECT id, coalesce(title || '\n', '') || content FROM memories"
            ).fetchall()
        }
    if not texts_by_id:
        texts_by_id = {"calibration": "calibration text"}
    # Eligibility comes from the CLI's own definition — the read path's serve
    # set — never a second spelling of it: `chars_in_eligible_memories` must not
    # quietly count the rows the migration excludes. SQLite stores the ids as
    # 32-char hex, the CLI keys them as dashed UUIDs: compare dashless.
    def _key(value: str) -> str:
        return str(value).replace("-", "").lower()

    with cli._session(readonly=True) as session:
        eligible_ids = {
            _key(memory_id) for memory_id, record in cli.memory_rows(session).items()
            if record["reason"] is None
        }
    texts = list(texts_by_id.values())
    corpus = [texts[index % len(texts)] for index in range(args.calibration_rows)]
    calibration_started = time.perf_counter()
    embed_texts_sync(corpus)
    calibration_seconds = max(time.perf_counter() - calibration_started, 1e-9)
    calibration_chars = sum(len(text) for text in corpus)
    rows_per_sec = args.calibration_rows / calibration_seconds
    seconds_per_row = 1.0 / rows_per_sec
    chars_per_sec = calibration_chars / calibration_seconds
    # The benchmark copy's memory texts are tiny; a real memory is hundreds to
    # thousands of characters, so the honest extrapolation is per CHARACTER,
    # with rows/second shown for a few representative row sizes.
    long_row_chars = 1000
    long_row_seconds = long_row_chars / chars_per_sec
    budget_seconds = BUDGET_MINUTES * 60

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "started_at": started_at.isoformat(),
        "purpose": "R26 dry run: prove the P1b maintenance window from measured throughput",
        "source": {
            "path": str(source.resolve()),
            "opened": False,
            "note": "the original database was copied (db + -wal + -shm) and never opened",
            "copy": str(db_copy),
            "workdir": str(workdir),
        },
        "runtime": {
            "database_url": settings.DATABASE_URL,
            "qdrant_mode": settings.QDRANT_MODE,
            "qdrant_local_path": settings.QDRANT_LOCAL_PATH,
            "embeddings": {
                "model": settings.LOCAL_EMBED_MODEL,
                "backend": "onnxruntime-cpu",
                "dim": int(cli.current_fingerprint()["dim"]),
                "pooling": cli.current_fingerprint()["pooling"],
                "generation": cli.fingerprint_token(),
            },
        },
        "prepare": {
            "seconds": prepare_seconds,
            "note": "fold the copy's WAL into the main file (not a migration phase)",
        },
        "dataset": {
            "memory_rows": inventory["kinds"]["memory"]["sql"]["total"],
            "eligible_memory_rows": inventory["kinds"]["memory"]["sql"]["eligible"],
            "chunk_rows": inventory["kinds"]["chunk"]["sql"]["total"],
            "eligible_chunk_rows": inventory["kinds"]["chunk"]["sql"]["eligible"],
            "superseded": inventory["kinds"]["memory"]["sql"]["superseded"],
            "dirty": inventory["kinds"]["memory"]["sql"]["dirty"],
            # Eligible-only, computed with the CLI's own eligibility rule; the
            # all-rows figure is labelled as such (it is the calibration corpus).
            "chars_in_eligible_memories": sum(
                len(text) for memory_id, text in texts_by_id.items()
                if _key(memory_id) in eligible_ids
            ),
            "chars_in_all_memories": sum(len(text) for text in texts_by_id.values()),
        },
        "phases": phases,
        "phase_totals": {
            "migration_seconds": round(sum(phase["seconds"] for phase in phases), 3),
            "budget_minutes": BUDGET_MINUTES,
        },
        "backfill": {"memory": memory_backfill, "chunk": chunk_backfill},
        "verify": {"memory": memory_verify, "chunk": chunk_verify},
        "backup": {key: value for key, value in backup.items() if key != "files"},
        "cutover": cutover,
        "calibration": {
            "rows": args.calibration_rows,
            "seconds": round(calibration_seconds, 3),
            "rows_per_sec": round(rows_per_sec, 2),
            "chars_per_sec": round(chars_per_sec, 1),
            "avg_chars_per_row": round(calibration_chars / args.calibration_rows, 1),
        },
        "extrapolation": {
            "basis": "measured CLS embedding throughput on this machine, single process",
            "seconds_per_row": round(seconds_per_row, 5),
            "seconds_per_1000_chars": round(long_row_seconds, 5),
            "budget_minutes": BUDGET_MINUTES,
            "max_eligible_rows_within_budget": int(budget_seconds / seconds_per_row),
            "max_eligible_rows_within_budget_at_1000_chars": int(budget_seconds / long_row_seconds),
            "tiers": [
                {"eligible_rows": rows,
                 "projected_minutes": round(rows * seconds_per_row / 60, 3),
                 "within_budget": rows * seconds_per_row <= budget_seconds}
                for rows in TIER_ROWS
            ],
            "tiers_at_1000_chars": [
                {"eligible_rows": rows,
                 "projected_minutes": round(rows * long_row_seconds / 60, 3),
                 "within_budget": rows * long_row_seconds <= budget_seconds}
                for rows in TIER_ROWS
            ],
        },
    }
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    asyncio.run(vector_backend.close_clients())  # release the embedded folder lock
    print(json.dumps({
        "out": str(args.out),
        "phases": phases,
        "verify": {"memory": memory_verify["ok"], "chunk": chunk_verify["ok"]},
        "rows_per_sec": report["calibration"]["rows_per_sec"],
        "cutover_blocked_intents": cutover["blocked_intents"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
