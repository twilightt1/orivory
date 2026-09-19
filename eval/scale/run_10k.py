#!/usr/bin/env python
"""The 10K ops run: a real local stack, measured, honest about its seams.

What runs here
--------------
A DETERMINISTIC corpus (``eval/scale/gen_corpus.py``: real LongMemEval-S user
turns + seeded synthetic fill) is ingested into a FRESH install of the real
stack — a private SQLite file and a private embedded-Qdrant folder under
``/tmp``, opened by the app's own boot path (``app.database.bootstrap_sqlite``)
and driven through the app's own write/index machinery:

* **ingest**: rows + their durable ``index_outbox`` intent commit together
  (the API's own order), and the EMBEDDING is performed by the real outbox
  drain (``drain_loop.drain_once`` → ``drain_pending`` → ``upsert_memory``),
  i.e. the durable path a bulk import or a boot drain takes. On success the
  write path would ack its intent; here nothing acks it, so the drain is the
  only path to the index and a lost write could not hide;
* **correction**: the PATCH-shape correction the API uses (content edit →
  ``bump_revision`` → ``enqueue_upsert`` → commit → write-through embed →
  ``mark_done``);
* **forget**: the real durable erasure path (``erase_memories``: receipt +
  delete intent, applied by the same drain);
* **recall**: ``MemoryRetriever.recall`` with the real local arctic XS ONNX
  embedder — SQL hydration, visibility filter, scoring, trace included.

The substitutions (recorded, because an ops artifact is only as honest as its seams)
-----------------------------------------------------------------------------------
- **LLM query rewrite**: identity. It is a paid out-of-process call and the
  plan forbids paid APIs in any run; the number it could move (rewrite quality)
  is not what any number here claims. Counted in ``seams.rewrite_stubbed``.
- **Knowledge-graph build**: suppressed (also an LLM call). Counted in
  ``seams.graph_builds_suppressed``; the drain path never schedules one.
- **Concurrent phase**: ONE recall loop (sequential calls) while write/forget
  workers run on the same process — the P2 gate's shape, not a parallel-recall
  load test. Recorded as such.
- **Cold page cache is NOT dropped** (no root, macOS): "cold" means a FRESH
  PROCESS opening the store and answering its first recall — the script is run
  twice (run 1 = fresh install, run 2 = ``--reuse`` on the same store) and both
  numbers are recorded. The OS page cache is warm in run 2 by construction.
- **Backup drill**: a file-level copy of the checkpointed SQLite file plus the
  embedded-Qdrant folder, restored into a fresh directory and verified by
  re-opening and counting. It does not exercise a real backup product, an
  incremental/streaming path, or a restore onto another machine.

Run::

    .venv/bin/python eval/scale/run_10k.py --dataset /path/longmemeval_s_cleaned.json
    .venv/bin/python eval/scale/run_10k.py --reuse <workdir> --out <artifact>

The second call is the cold-process measurement; the first writes
``<workdir>/run1.json`` and an artifact you can read on its own.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.scale.gen_corpus import DEFAULT_DATASET, REAL_SHARE, SEED, generate, load_corpus  # noqa: E402

ARTIFACTS_DIR = ROOT / "eval" / "scale" / "artifacts"
RUN1_NAME = "run1.json"
GATE_P95_MS = 150.0
CONCURRENT_SECONDS = 240.0
WARM_CALLS = 200
QUERY_COUNT = 60
QUERY_WORDS = 8
TOP_K = 10
ENQUEUE_BATCH = 100
TAIL_ROWS = 1000  # rows held back for the concurrent phase (ingest under load)
CORRECT_INTERVAL = 8.0
FORGET_INTERVAL = 12.0
RECALL_WARMUP_DISCARD = 3
BASE_TIME = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)
EMBED_DIM = 384
SCALE_USER = "orivory/scale/user"

CONFIG_FLAG_KEYS = (
    "ENVIRONMENT", "DATABASE_URL", "QDRANT_MODE", "QDRANT_LOCAL_PATH", "USE_LOCAL_EMBEDDINGS",
    "USE_JINA_EMBEDDINGS", "LOCAL_EMBED_MODEL", "EMBED_BATCH_SIZE",
    "EMBED_EXECUTOR_WORKERS", "EMBED_ORT_INTRA_OP_THREADS", "RETRIEVAL_HYBRID_ENABLED",
    "RETRIEVAL_SEMANTIC_RERANK", "RETRIEVAL_RERANK_POOL_MULTIPLIER",
    "RECALL_FRESHNESS_BUDGET_SECONDS", "OUTBOX_DRAIN_ENABLED",
    "OUTBOX_DRAIN_INTERVAL_SECONDS", "OUTBOX_DRAIN_BATCH_SIZE",
)


# ── small helpers ───────────────────────────────────────────────────────────


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile (documented method, no interpolation claims)."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def latency_block(values: list[float]) -> dict:
    return {
        "calls": len(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "max_ms": max(values) if values else None,
        "mean_ms": (sum(values) / len(values)) if values else None,
        "method": "nearest-rank percentile over per-call wall time "
                  "(time.perf_counter around one MemoryRetriever.recall call, "
                  "SQL session acquisition excluded)",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dir_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def git_state() -> dict:
    def _run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    return {
        "head": _run("rev-parse", "HEAD"),
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": _run("status", "--porcelain") or None,
    }


def rss_block(pid: int) -> dict:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS reports ru_maxrss in bytes, Linux in kibibytes.
    peak = int(usage.ru_maxrss) * (1 if sys.platform == "darwin" else 1024)
    current = None
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                             capture_output=True, text=True).stdout.strip()
        current = int(out) * 1024 if out else None
    except (OSError, ValueError):
        pass
    return {
        "peak_rss_bytes": peak,
        "current_rss_bytes": current,
        "ru_maxrss_raw": int(usage.ru_maxrss),
        "method": "resource.getrusage(RUSAGE_SELF).ru_maxrss (peak, process-wide, "
                  "CONTAMINATED by everything the process loaded before the stores) "
                  "+ ps -o rss= at measurement time (current)",
    }


def hardware_note() -> dict:
    def _sysctl(name: str) -> str | None:
        try:
            return subprocess.run(["sysctl", "-n", name], capture_output=True,
                                  text=True).stdout.strip() or None
        except OSError:
            return None

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _sysctl("machdep.cpu.brand_string") or platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_bytes": int(_sysctl("hw.memsize") or 0) or None,
        "note": "one local macOS box; single process; NO page-cache drops (no root)",
    }


# ── the stack ───────────────────────────────────────────────────────────────


def configure_environment(workdir: Path) -> dict:
    """Point every app-level store at this run's private /tmp install.

    Set BEFORE ``app.*`` is imported: ``app.database`` builds its engine from
    ``settings.DATABASE_URL`` at import time and both the outbox drain and the
    freshness barrier commit through that global sessionmaker, so a late env
    change would silently measure a different database.
    """
    env = {
        "DATABASE_URL": f"sqlite+aiosqlite:///{workdir / 'memories.db'}",
        "QDRANT_MODE": "local",
        "QDRANT_LOCAL_PATH": str(workdir / "qdrant"),
        "USE_LOCAL_EMBEDDINGS": "true",
        "USE_JINA_EMBEDDINGS": "false",
        "LOCAL_EMBED_MODEL": "arctic",
        "RETRIEVAL_HYBRID_ENABLED": "false",
        "RETRIEVAL_SEMANTIC_RERANK": "false",
        "OUTBOX_DRAIN_ENABLED": "true",
        # Only to silence the development-mode SQL echo (the first run of this
        # script wrote 129MB of echoed statements, which distorts nothing but
        # buries the numbers). Nothing else on this path reads ENVIRONMENT:
        # `is_production` gates JWT validation and docs URL, and no HTTP server
        # is booted here.
        "ENVIRONMENT": "staging",
    }
    os.environ.update(env)
    return env


def install_seams() -> dict:
    """Replace ONLY the two paid/out-of-process seams; count every suppression."""
    from app.retrieval.memory import retriever as retriever_module
    from app.retrieval.memory import write_back

    counters = {"rewrite_stubbed": 0, "graph_builds_suppressed": 0}

    async def _identity_rewrite(query, context=None):
        counters["rewrite_stubbed"] += 1
        return {"rewritten_query": query, "entities": [], "reasoning": None,
                "_fallback_used": False}

    def _suppress_graph_build(memory_id):  # the extraction LLM call, off by policy
        counters["graph_builds_suppressed"] += 1

    retriever_module.rewrite_query = _identity_rewrite
    write_back.safe_enqueue_graph_build = _suppress_graph_build
    return counters


def fingerprint(settings, corpus_manifest: dict, corpus_sha: str, workdir: Path) -> dict:
    from app.retrieval import e5_local
    from app.retrieval.embedding_fingerprint import current_fingerprint

    model_path = e5_local.model_dir() / e5_local.ARCTIC_MODEL_FILE
    tokenizer_path = e5_local.model_dir() / e5_local.ARCTIC_TOKENIZER_FILE
    return {
        "git": git_state(),
        "corpus": {
            "sha256": corpus_sha,
            "seed": corpus_manifest["seed"],
            "rows": corpus_manifest["rows"],
            "parts": {name: part["count"] for name, part in corpus_manifest["parts"].items()},
            "dataset": corpus_manifest["dataset"],
        },
        "model": {
            "id": "arctic-xs",
            "file": str(model_path),
            "model_sha256": sha256_file(model_path) if model_path.is_file() else None,
            "declared_model_sha256": e5_local.ARCTIC_MODEL_SHA256,
            "tokenizer_sha256": sha256_file(tokenizer_path) if tokenizer_path.is_file() else None,
            "declared_tokenizer_sha256": e5_local.ARCTIC_TOKENIZER_SHA256,
            "dim": EMBED_DIM,
            "pooling": "cls",
            "query_prefix": e5_local.ARCTIC_QUERY_PREFIX,
            "embedding_contract": current_fingerprint(),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "onnxruntime": __import__("onnxruntime").__version__,
            "qdrant_client": importlib.metadata.version("qdrant-client"),
            "numpy": __import__("numpy").__version__,
            "sqlalchemy": __import__("sqlalchemy").__version__,
        },
        "hardware": hardware_note(),
        "config": {key: getattr(settings, key) for key in CONFIG_FLAG_KEYS},
        "measurement": {
            "top_k": TOP_K,
            "concurrency": "one recall loop + one drain worker + one ingest worker "
                           "+ one correct worker + one forget worker, in ONE process",
            "recall_warmup_discarded": RECALL_WARMUP_DISCARD,
            "workdir": str(workdir),
        },
    }


async def boot(workdir: Path) -> dict:
    """The app's own boot: schema ladder + embedder warmup + store touch."""
    t0 = time.perf_counter()
    from app import models  # noqa: F401 — registers every table on Base
    from app.database import bootstrap_sqlite

    await bootstrap_sqlite()
    schema_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    from app.retrieval.embedder import warmup_embedder

    await warmup_embedder()
    embedder_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    from app.retrieval.memory.outbox import active_generation
    from app.retrieval.vector_backend import get_async_client

    generation, manifest_fingerprint = await active_generation()
    client = get_async_client()
    try:
        points = (await client.count(collection_name=generation)).count
    except Exception:
        points = None  # a fresh install has no collection yet: created on first write
    store_seconds = time.perf_counter() - t0
    return {
        "schema_seconds": schema_seconds,
        "embedder_warmup_seconds": embedder_seconds,
        "store_open_seconds": store_seconds,
        "generation": generation,
        "generation_manifest_fingerprint": manifest_fingerprint,
        "points_at_boot": points,
    }


async def ensure_user() -> uuid.UUID:
    from sqlalchemy import select

    from app.database import AsyncSessionLocal
    from app.models.user import User

    user_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{SCALE_USER}/{SEED}")
    async with AsyncSessionLocal() as db:
        user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
        if user is None:
            db.add(User(id=user_id, email="scale@orivory.invalid", hashed_password="x",
                        display_name="scale", onboarding_done=True, is_verified=True,
                        is_active=True, is_deleted=False))
            await db.commit()
    return user_id


def memory_from_row(row: dict, user_id: uuid.UUID, index: int):
    from app.models.memory import Memory
    from app.retrieval.memory.namespaces import personal_namespace

    return Memory(
        id=uuid.UUID(row["memory_id"]),
        user_id=user_id,
        namespace=personal_namespace(user_id),
        title=None,
        content=row["text"],
        tags=[],
        salience=0.5,
        pinned=False,
        source_type="generic_import",
        captured_at=BASE_TIME + timedelta(seconds=index * 60),
        extra_metadata={"scale_part": row["part"]},
    )


async def enqueue_batch(rows: list[dict], user_id: uuid.UUID, start_index: int,
                        *, flush_every: int = 25) -> int:
    """Rows + their durable upsert intents, committed in small flush groups.

    The API's own order (row and intent in one transaction). The periodic
    ``flush`` bounds the ORM's multi-row INSERT statement: SQLAlchemy batches
    every pending insert at flush time, and one unbounded statement per batch
    is a giant compiled object whose cache key is built per execution — the
    measured cost is linear (200 rows in one commit: ~0.1s), but nothing here
    needs a statement bigger than the flush group.
    """
    from app.database import AsyncSessionLocal
    from app.retrieval.memory.outbox import bump_revision, enqueue_upsert

    async with AsyncSessionLocal() as db:
        for offset, row in enumerate(rows):
            memory = memory_from_row(row, user_id, start_index + offset)
            db.add(memory)
            bump_revision(memory)
            await enqueue_upsert(db, memory)
            if flush_every and (offset + 1) % flush_every == 0:
                await db.flush()
        await db.commit()
    return len(rows)


def _merge(report: dict, totals: dict) -> None:
    for key in ("claimed", "applied", "skipped", "blocked", "failed"):
        totals[key] += int(report.get(key, 0))
    totals["rounds"] += 1


async def drain_until_empty(*, deadline: float | None = None) -> dict:
    """Drain through the single-flight door until nothing is claimable.

    The stop condition is the PER-ROUND count, never the running total: a
    cumulative total stays above zero after the first round and the loop would
    never end (measured: ~1,900 claim queries/s forever).
    """
    from app.retrieval.memory.drain_loop import drain_once

    totals = {"rounds": 0, "claimed": 0, "applied": 0, "skipped": 0, "blocked": 0, "failed": 0}
    while True:
        report = await drain_once(batch_size=50)
        _merge(report, totals)
        if not report.get("claimed") or (deadline is not None and time.perf_counter() > deadline):
            break
    return totals


async def recall_once(user_id: uuid.UUID, query: str) -> dict:
    """One real recall, timed around the pipeline (session acquisition excluded)."""
    from app.database import AsyncSessionLocal
    from app.retrieval.memory.retriever import MemoryRetriever

    async with AsyncSessionLocal() as db:
        retriever = MemoryRetriever(db=db, user_id=user_id)
        t0 = time.perf_counter()
        try:
            response = await retriever.recall(query, top_k=TOP_K)
        except Exception as exc:  # typed 503s are real outcomes, recorded as such
            return {"ms": (time.perf_counter() - t0) * 1000.0, "error": type(exc).__name__,
                    "stage_ms": None, "returned": None}
        return {
            "ms": (time.perf_counter() - t0) * 1000.0,
            "error": None,
            "stage_ms": dict(response.trace.stage_ms),
            "returned": len(response.results),
        }


async def correct_one(user_id: uuid.UUID, memory_id: uuid.UUID, index: int) -> bool:
    """The API's PATCH shape: edit → bump → intent → commit → write-through → ack."""
    from app.database import AsyncSessionLocal
    from app.models.memory import Memory
    from app.retrieval.memory.outbox import bump_revision, enqueue_upsert, mark_done
    from app.retrieval.memory.write_back import safe_upsert_to_index

    async with AsyncSessionLocal() as db:
        memory = await db.get(Memory, memory_id)
        if memory is None:
            return False
        memory.content = f"{memory.content} [revised {index}]"
        bump_revision(memory)
        await enqueue_upsert(db, memory)
        await db.commit()
        await db.refresh(memory)
        indexed = await safe_upsert_to_index(memory)
        if indexed:
            await mark_done(db, entity_id=memory.id, revision=memory.revision)
        return bool(indexed)


async def forget_one(user_id: uuid.UUID, memory_id: uuid.UUID) -> str | None:
    """The real durable erasure (receipt + delete intent, applied by the drain)."""
    from app.database import AsyncSessionLocal
    from app.services.erasure_service import erase_memories

    async with AsyncSessionLocal() as db:
        receipt = await erase_memories(db, user_id, [memory_id], requested_by="scale_run")
    return getattr(receipt, "status", None)


async def counts(user_id: uuid.UUID) -> dict:
    """Canonical SQL rows vs vector points vs chunks — SEPARATELY (§10.2)."""
    from sqlalchemy import func, select

    from app.database import AsyncSessionLocal
    from app.models.document_chunk import DocumentChunk
    from app.models.index_outbox import IndexOutbox
    from app.models.memory import Memory
    from app.retrieval.memory.outbox import CHUNK_TARGET_GENERATION, active_generation
    from app.retrieval.vector_backend import get_async_client

    async with AsyncSessionLocal() as db:
        memories = (await db.execute(select(func.count()).select_from(Memory))).scalar_one()
        chunks = (await db.execute(select(func.count()).select_from(DocumentChunk))).scalar_one()
        outbox_rows = (await db.execute(
            select(IndexOutbox.status, func.count()).group_by(IndexOutbox.status)
        )).all()
    generation, _ = await active_generation()
    client = get_async_client()
    try:
        memory_points = (await client.count(collection_name=generation)).count
    except Exception:
        memory_points = None
    try:
        chunk_points = (await client.count(collection_name=CHUNK_TARGET_GENERATION)).count
    except Exception:
        chunk_points = None
    return {
        "memories_sql": int(memories),
        "memory_points": memory_points,
        "chunks": int(chunks),
        "chunk_points": chunk_points,
        "collection": generation,
        "outbox_by_status": {str(status): int(count) for status, count in outbox_rows},
        "note": "canonical memories (SQL) vs vector points (Qdrant) vs chunks are counted "
                "separately; the workload is memory-only, so the chunk family is 0 and its "
                "collection may not exist",
    }


async def backup_drill(workdir: Path, generation: str) -> dict:
    """Checkpoint → copy → restore into a fresh dir → verify by re-opening."""
    from sqlalchemy import text

    from app.database import AsyncSessionLocal

    db_path = workdir / "memories.db"
    qdrant_path = workdir / "qdrant"
    async with AsyncSessionLocal() as db:
        await db.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
        await db.commit()

    backup_dir = workdir / "backup"
    shutil.rmtree(backup_dir, ignore_errors=True)
    backup_dir.mkdir(parents=True)
    t0 = time.perf_counter()
    shutil.copy2(db_path, backup_dir / db_path.name)
    wal = db_path.with_name(db_path.name + "-wal")
    if wal.exists() and wal.stat().st_size:
        shutil.copy2(wal, backup_dir / wal.name)
    shutil.copytree(qdrant_path, backup_dir / "qdrant")
    backup_seconds = time.perf_counter() - t0
    backup_bytes = dir_bytes(backup_dir)

    restored = workdir / "restored"
    shutil.rmtree(restored, ignore_errors=True)
    restored.mkdir(parents=True)
    t0 = time.perf_counter()
    shutil.copy2(backup_dir / db_path.name, restored / db_path.name)
    if (backup_dir / wal.name).exists():
        shutil.copy2(backup_dir / wal.name, restored / wal.name)
    shutil.copytree(backup_dir / "qdrant", restored / "qdrant")
    with sqlite3.connect(restored / db_path.name) as conn:
        restored_memories = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    from qdrant_client import QdrantClient

    client = QdrantClient(path=str(restored / "qdrant"))
    try:
        restored_points = client.count(collection_name=generation).count
    finally:
        client.close()
    restore_seconds = time.perf_counter() - t0
    return {
        "backup_seconds": backup_seconds,
        "restore_seconds": restore_seconds,
        "backup_bytes": backup_bytes,
        "restored_memories": int(restored_memories),
        "restored_points": int(restored_points),
        "verified": None,  # filled by the caller, which knows the live counts
        "method": "PRAGMA wal_checkpoint(TRUNCATE) → file copy of the SQLite db + the "
                  "embedded-Qdrant folder → copy back into a fresh dir → re-open "
                  "(sqlite3 + QdrantClient) and count. Restore timing INCLUDES the "
                  "verification reads.",
    }


def disk_block(workdir: Path) -> dict:
    db_path = workdir / "memories.db"
    wal = db_path.with_name(db_path.name + "-wal")
    return {
        "workdir_bytes": dir_bytes(workdir),
        "memories_db_bytes": db_path.stat().st_size if db_path.exists() else 0,
        "wal_bytes": wal.stat().st_size if wal.exists() else 0,
        "qdrant_dir_bytes": dir_bytes(workdir / "qdrant"),
    }


# ── the phases ──────────────────────────────────────────────────────────────


def build_queries(rows: list[dict], count: int = QUERY_COUNT) -> list[str]:
    """Deterministic queries: the opening words of real corpus rows."""
    queries: list[str] = []
    seen: set[str] = set()
    for row in rows:
        query = " ".join(row["text"].split()[:QUERY_WORDS])
        if query and query not in seen:
            seen.add(query)
            queries.append(query)
        if len(queries) >= count:
            break
    return queries


async def run_full(opts) -> dict:
    workdir: Path = opts.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    configure_environment(workdir)
    seams = install_seams()

    if opts.corpus:
        corpus_path = Path(opts.corpus).resolve()
        rows, manifest = load_corpus(corpus_path)
        corpus_sha = manifest["corpus_sha256"]
    else:
        corpus_path = workdir / "corpus.jsonl"
        manifest = generate(rows=opts.rows, seed=opts.seed, dataset=Path(opts.dataset),
                            out=corpus_path, real_share=opts.real_share)
        rows, manifest = load_corpus(corpus_path)
        corpus_sha = manifest["corpus_sha256"]
    if len(rows) != opts.rows:
        raise SystemExit(f"corpus has {len(rows)} rows but --rows is {opts.rows}")

    boot_info = await boot(workdir)
    user_id = await ensure_user()

    from app.config import settings

    print(f"corpus: rows={len(rows)} sha256={corpus_sha[:12]} "
          f"real={manifest['parts']['real']['count']} "
          f"synthetic={manifest['parts']['synthetic']['count']}")

    # ── phase 1: bulk ingest (rows + intents, the drain embeds) ─────────────
    tail_size = min(TAIL_ROWS, max(1, len(rows) // 10))
    if len(rows) - tail_size < 1:  # tiny runs: keep at least one bulk row
        tail_size = max(0, len(rows) - 1)
    tail = rows[len(rows) - tail_size:] if tail_size else []
    bulk = rows[: len(rows) - len(tail)]
    ingest_totals = {"rounds": 0, "claimed": 0, "applied": 0, "skipped": 0,
                     "blocked": 0, "failed": 0, "batches": 0}
    t0 = time.perf_counter()
    for start in range(0, len(bulk), ENQUEUE_BATCH):
        batch = bulk[start:start + ENQUEUE_BATCH]
        await enqueue_batch(batch, user_id, start)
        ingest_totals["batches"] += 1
        _merge(await drain_until_empty(), ingest_totals)
        done = start + len(batch)
        print(f"  ingest {done}/{len(rows)} rows "
              f"({time.perf_counter() - t0:.1f}s, applied={ingest_totals['applied']})")
    ingest_seconds = time.perf_counter() - t0
    ingest = {
        "memories": len(bulk),
        "tail_held_for_concurrent_phase": len(tail),
        "batches": ingest_totals["batches"],
        "seconds": ingest_seconds,
        "memories_per_min": len(bulk) / (ingest_seconds / 60.0),
        "drain": ingest_totals,
        "method": "rows + durable upsert intents committed in batches of "
                  f"{ENQUEUE_BATCH}; embedding performed by the real outbox drain "
                  "(drain_once → drain_pending, batch 50); nothing acks the intents "
                  "on the write path, so every point in the store came through it",
    }
    print(f"ingest: {ingest['memories']} memories in {ingest_seconds:.1f}s "
          f"({ingest['memories_per_min']:.1f}/min)")

    # ── phase 2: first recall + warm pass (in-process, page cache warm) ─────
    queries = build_queries(rows)
    first = await recall_once(user_id, queries[0])
    warm_latencies: list[float] = []
    warm_stages: dict[str, list[float]] = defaultdict(list)
    warm_errors: dict[str, int] = {}
    for index in range(WARM_CALLS):
        result = await recall_once(user_id, queries[index % len(queries)])
        if result["error"]:
            warm_errors[result["error"]] = warm_errors.get(result["error"], 0) + 1
        else:
            warm_latencies.append(result["ms"])
            for stage, value in result["stage_ms"].items():
                warm_stages[stage].append(value)
    warm = {
        "calls": WARM_CALLS,
        "latency_ms": latency_block(warm_latencies),
        "errors": warm_errors,
        "stage_ms_median": {key: percentile(values, 0.5) for key, values in warm_stages.items()},
        "note": "second pass in the SAME process: page cache warm by construction",
    }
    print(f"warm recall: p50={warm['latency_ms']['p50_ms']:.1f}ms "
          f"p95={warm['latency_ms']['p95_ms']:.1f}ms")

    # ── phase 3: concurrent phase ───────────────────────────────────────────
    mixed = await mixed_phase(user_id=user_id, tail=tail, start_index=len(bulk),
                              correction_ids=[uuid.UUID(row["memory_id"]) for row in bulk[:40]],
                              queries=queries, seconds=opts.concurrent_seconds)
    print(f"mixed: recalls={mixed['recall_calls']} p50={mixed['latency_ms']['p50_ms']:.1f}ms "
          f"p95={mixed['latency_ms']['p95_ms']:.1f}ms "
          f"gate={'PASS' if mixed['gate']['p95_le_150ms'] else 'FAIL'}")

    # ── phase 4: quiesce + backup drill + final numbers ─────────────────────
    final_drain = await drain_until_empty()
    final_counts = await counts(user_id)
    backup = await backup_drill(workdir, final_counts["collection"])
    backup["verified"] = (
        backup["restored_memories"] == final_counts["memories_sql"]
        and backup["restored_points"] == final_counts["memory_points"]
    )

    startup = {
        "boot": boot_info,
        "first_recall_after_ingest_ms": first["ms"],
        "first_recall_error": first["error"],
        "note": "this run's own open+first recall: the same store just wrote every point, "
                "so the page cache is warm. The cold-PROCESS measurement is the --reuse run "
                "(recorded under cold_process).",
    }
    run1 = {
        "status": "run1-complete",
        "rss_ceiling": "pending-user",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "role": "run-1 (fresh install, full ingest)",
        "rss_ceiling_note": "D4: the RSS ceiling is the user's to set after these numbers; "
                            "this artifact deliberately does not choose one",
        "fingerprint": fingerprint(settings, manifest, corpus_sha, workdir),
        "corpus": {**{k: manifest[k] for k in ("corpus_file", "corpus_sha256", "rows", "seed",
                                               "real_share", "parts", "dataset")},
                   "path": str(corpus_path)},
        "queries": queries,
        "counts": {**final_counts, "corpus_rows": len(rows)},
        "ingest": ingest,
        "startup": startup,
        "warm_recall": warm,
        "concurrent": mixed,
        "backup": backup,
        "final_drain": final_drain,
        "disk": disk_block(workdir),
        "rss": rss_block(os.getpid()),
        "outbox": {
            "pending": final_counts["outbox_by_status"].get("pending", 0),
            "acked": final_counts["outbox_by_status"].get("done", 0),
            "blocked": final_counts["outbox_by_status"].get("blocked", 0),
            "failed": final_counts["outbox_by_status"].get("failed", 0),
            "by_status": final_counts["outbox_by_status"],
            "note": "acked = status 'done'; the write path acks its own intent, this run's "
                    "ingest never did, so 'acked' here is the drain's own accounting",
        },
        "seams": {
            **seams,
            "llm_rewrite": "identity (paid API forbidden in this run) — counted above",
            "graph_builds": "suppressed (LLM extraction) — counted above",
            "cold_page_cache": "NOT dropped (no root): the cold number is a fresh PROCESS "
                               "on the same store (--reuse), not a cold page cache",
            "backup": "file-level copy + re-open verify, not a backup product, not a "
                      "cross-machine restore",
            "recall_concurrency": "ONE sequential recall loop against concurrent writers "
                                  "(the P2 gate's shape), not a parallel-recall load test",
        },
    }
    (workdir / RUN1_NAME).write_text(json.dumps(run1, indent=2, ensure_ascii=False) + "\n",
                                     encoding="utf-8")
    await _close_clients()
    return run1


async def _close_clients() -> None:
    """Release the embedded store's folder lock before the process exits."""
    from app.retrieval.vector_backend import close_clients

    try:
        await close_clients()
    except Exception:  # closing is best-effort; the process is about to end
        pass


async def mixed_phase(*, user_id, tail, correction_ids, start_index, queries, seconds) -> dict:
    """Recall loop WHILE ingest/correct/forget run, all through the real paths."""
    from app.retrieval.memory.drain_loop import drain_once

    deadline = time.perf_counter() + seconds
    stop = asyncio.Event()
    latencies: list[float] = []
    errors: dict[str, int] = {}
    stages: dict[str, list[float]] = defaultdict(list)
    ops = {"ingested": 0, "corrected": 0, "forgotten": 0}
    drain = {"rounds": 0, "claimed": 0, "applied": 0, "skipped": 0, "blocked": 0, "failed": 0}
    forget_status: dict[str, int] = {}

    async def drain_worker():
        while not stop.is_set():
            report = await drain_once(batch_size=50)
            _merge(report, drain)
            if not report.get("claimed"):
                await asyncio.sleep(0.05)

    async def ingest_worker():
        index = start_index
        for start in range(0, len(tail), ENQUEUE_BATCH):
            if stop.is_set():
                return
            batch = tail[start:start + ENQUEUE_BATCH]
            await enqueue_batch(batch, user_id, index + start)
            ops["ingested"] += len(batch)
            await asyncio.sleep(2.0)

    candidates = [uuid.UUID(row["memory_id"]) for row in tail[: min(len(tail), 40)]]
    corrections = correction_ids or candidates
    forget_start_delay = min(5.0, max(0.5, seconds / 10.0))

    async def correct_worker():
        index = 0
        while not stop.is_set():
            ok = await correct_one(user_id, corrections[index % len(corrections)], index)
            ops["corrected"] += 1 if ok else 0
            index += 1
            await sleep_or_stop(stop, CORRECT_INTERVAL)

    async def forget_worker():
        # The tail rows exist only after the ingest worker's first commit: wait
        # a moment (scaled to the phase) so a forget targets a real row.
        await sleep_or_stop(stop, forget_start_delay)
        index = 0
        while not stop.is_set():
            status = await forget_one(user_id, candidates[index % len(candidates)])
            if status:
                forget_status[status] = forget_status.get(status, 0) + 1
                ops["forgotten"] += 1
            index += 1
            await sleep_or_stop(stop, FORGET_INTERVAL)

    async def recall_worker():
        index = 0
        measured = 0
        while not stop.is_set():
            result = await recall_once(user_id, queries[index % len(queries)])
            index += 1
            if measured < RECALL_WARMUP_DISCARD:
                measured += 1
                continue
            if result["error"]:
                errors[result["error"]] = errors.get(result["error"], 0) + 1
            else:
                latencies.append(result["ms"])
                for stage, value in result["stage_ms"].items():
                    stages[stage].append(value)

    tasks = [asyncio.create_task(coro()) for coro in
             (drain_worker, ingest_worker, correct_worker, forget_worker, recall_worker)]
    while time.perf_counter() < deadline:
        await asyncio.sleep(0.1)
    stop.set()
    await asyncio.gather(*tasks, return_exceptions=True)

    latency = latency_block(latencies)
    return {
        "seconds": seconds,
        "recall_calls": len(latencies),
        "recall_errors": errors,
        "recall_errors_note": "typed failures (e.g. IndexFreshnessTimeout) are real 503s "
                              "from the freshness barrier, recorded, never dropped",
        "latency_ms": latency,
        "stage_ms_median": {key: percentile(values, 0.5) for key, values in stages.items()},
        "ops": ops,
        "forget_statuses": forget_status,
        "drain": drain,
        "gate": {
            "metric": "p95 of sequential recall calls while ingest/correct/forget run",
            "threshold_ms": GATE_P95_MS,
            "p95_ms": latency["p95_ms"],
            "p95_le_150ms": (latency["p95_ms"] is not None and latency["p95_ms"] <= GATE_P95_MS),
            "recorded_not_asserted": True,
            "authority": "D4 (signed): p95 <= 150ms — RECORDED here, no assert, no code "
                         "path depends on it",
        },
        "feed": {
            "ingest_rows": ops["ingested"],
            "queries": len(queries),
        },
    }


async def sleep_or_stop(stop: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
    except TimeoutError:
        pass


async def run_reuse(opts) -> dict:
    """Run 2: a FRESH process opening the same store (the cold-process number)."""
    workdir: Path = Path(opts.reuse)
    run1 = json.loads((workdir / RUN1_NAME).read_text(encoding="utf-8"))
    configure_environment(workdir)
    seams = install_seams()

    t0 = time.perf_counter()
    boot_info = await boot(workdir)
    user_id = await ensure_user()
    open_seconds = time.perf_counter() - t0

    queries = run1["queries"]
    t0 = time.perf_counter()
    first = await recall_once(user_id, queries[0])
    first_recall_seconds = time.perf_counter() - t0

    latencies: list[float] = []
    errors: dict[str, int] = {}
    stages: dict[str, list[float]] = defaultdict(list)
    for index in range(WARM_CALLS):
        result = await recall_once(user_id, queries[index % len(queries)])
        if result["error"]:
            errors[result["error"]] = errors.get(result["error"], 0) + 1
        else:
            latencies.append(result["ms"])
            for stage, value in result["stage_ms"].items():
                stages[stage].append(value)

    warm = {
        "calls": WARM_CALLS,
        "latency_ms": latency_block(latencies),
        "errors": errors,
        "stage_ms_median": {key: percentile(values, 0.5) for key, values in stages.items()},
    }
    current = await counts(user_id)


    run2 = {
        "role": "run-2 (--reuse: fresh process, same store — the cold-process proxy)",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "cold_process": {
            "open_seconds": open_seconds,
            "open_boot": boot_info,
            "first_recall_seconds": first_recall_seconds,
            "first_recall_ms": first["ms"],
            "first_recall_error": first["error"],
            "open_plus_first_recall_ms": first_recall_seconds * 1000.0,
            "note": "open = app boot on the EXISTING store (schema ladder check + embedder "
                    "warmup + store touch) + the first real recall. Fresh process, warm page "
                    "cache: the honest border of what this box can measure without root.",
        },
        "warm_recall": warm,
        "counts": current,
        "disk": disk_block(workdir),
        "rss": rss_block(os.getpid()),
        "seams": {"llm_rewrite": "identity", "graph_builds": "suppressed", **seams},
    }

    artifact = dict(run1)
    artifact["status"] = "complete"
    artifact["cold_process"] = run2["cold_process"]
    artifact["runs"] = {
        "run1": {
            "role": run1["role"], "generated_at": run1["generated_at"],
            "counts": run1["counts"], "disk": run1["disk"], "rss": run1["rss"],
            "warm_recall": run1["warm_recall"],
        },
        "run2": run2,
    }
    artifact["counts_run2"] = current
    artifact["fingerprint"]["measurement"]["run2_open_seconds"] = open_seconds
    await _close_clients()
    return artifact


def default_artifact_path(corpus_sha: str) -> Path:
    return ARTIFACTS_DIR / f"10k_{corpus_sha[:12]}.json"


def write_artifact(payload: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summarize(artifact: dict) -> None:
    counts = artifact["counts"]
    print("")
    print(f"status: {artifact['status']}  rss_ceiling: {artifact['rss_ceiling']}")
    print(f"counts: memories_sql={counts['memories_sql']} memory_points={counts['memory_points']} "
          f"chunks={counts['chunks']} corpus_rows={counts['corpus_rows']}")
    ingest = artifact["ingest"]
    print(f"ingest: {ingest['memories']} in {ingest['seconds']:.1f}s "
          f"({ingest['memories_per_min']:.1f} memories/min)")
    concurrent = artifact["concurrent"]
    lat = concurrent["latency_ms"]
    print(f"concurrent: calls={concurrent['recall_calls']} errors={concurrent['recall_errors']} "
          f"p50={lat['p50_ms']:.1f} p95={lat['p95_ms']:.1f} p99={lat['p99_ms']:.1f} ms "
          f"gate(p95<={GATE_P95_MS:.0f})="
          f"{'PASS' if concurrent['gate']['p95_le_150ms'] else 'FAIL'}")
    print(f"ops: {concurrent['ops']}")
    backup = artifact["backup"]
    print(f"backup: {backup['backup_seconds']:.2f}s restore: {backup['restore_seconds']:.2f}s "
          f"verified={backup['verified']} ({backup['restored_memories']} rows / "
          f"{backup['restored_points']} points)")
    rss = artifact["rss"]
    print(f"rss peak={rss['peak_rss_bytes'] / 1e6:.0f}MB current={rss['current_rss_bytes']}B "
          f"db={artifact['disk']['memories_db_bytes'] / 1e6:.1f}MB "
          f"wal={artifact['disk']['wal_bytes']}B qdrant={artifact['disk']['qdrant_dir_bytes'] / 1e6:.1f}MB")
    outbox = artifact["outbox"]
    print(f"outbox: pending={outbox['pending']} acked={outbox['acked']} "
          f"blocked={outbox['blocked']} failed={outbox['failed']}")
    if "cold_process" in artifact:
        cold = artifact["cold_process"]
        print(f"cold process: open={cold['open_seconds']:.2f}s first_recall={cold['first_recall_ms']:.1f}ms")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="10K ops run against the real local stack")
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--real-share", type=float, default=REAL_SHARE)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--corpus", default=None, help="reuse a corpus jsonl (+ manifest)")
    parser.add_argument("--workdir", default=None, help="fresh dir for db + qdrant + corpus")
    parser.add_argument("--reuse", default=None,
                        help="run 2: open the workdir of a finished run 1 and measure "
                             "cold-process open + first recall")
    parser.add_argument("--concurrent-seconds", type=float, default=CONCURRENT_SECONDS)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    if args.reuse:
        artifact = asyncio.run(run_reuse(args))
        default_out = default_artifact_path(artifact["fingerprint"]["corpus"]["sha256"])
    else:
        if not args.workdir:
            stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            args.workdir = Path(f"/tmp/orivory-scale-{stamp}-{os.getpid()}")
        args.workdir = Path(args.workdir)
        artifact = asyncio.run(run_full(args))
        default_out = default_artifact_path(artifact["fingerprint"]["corpus"]["sha256"])
    out = Path(args.out) if args.out else default_out

    write_artifact(artifact, out)
    summarize(artifact)
    print(f"\nartifact: {out}")
    if not args.reuse:
        print(f"workdir: {args.workdir}  (re-run with --reuse {args.workdir} for the "
              f"cold-process number)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
