#!/usr/bin/env python
"""Offline P1b migration CLI: inventory -> backup -> backfill -> verify -> cutover.

Implements spec §6.2 (expand -> backfill -> verify -> cutover) for the
SQLite/Qdrant deployments, and the maintenance discipline of §6.2 step 2 under
the controller rulings R24-R29:

- ``inventory`` is STRICTLY read-only (R28): a read-only SQLite connection, no
  write anywhere, and no collection creation. It reports counts per tenant/kind
  plus the missing/stale/orphan/excluded/unknown-fingerprint/malformed lists.
- ``backup`` takes a VACUUM INTO snapshot (committed WAL frames included) plus a
  sha256 manifest, copies the upload directory and the legacy Chroma directory
  when they exist, and never overwrites an existing snapshot. It refuses an
  empty or unmanifested file instead of silently reusing one.
- ``backfill`` builds the contract generation (``generation_name(kind)``) with a
  keyset scan over the primary key — never OFFSET — embedding only the ELIGIBLE
  SQL rows. It writes into the generation by NAME (the one ``cutover`` will
  later activate), never "whatever is active". A checkpoint keyed
  ``(generation, kind, tenant, last_id)`` is written after every acked batch, so
  a crash re-embeds at most one batch. It finishes with a GC pass that DELETES
  points whose SQL row is not eligible (R29b).
- ``verify`` is a real read-side audit of the live collection: full scan, count
  vs eligible SQL rows, ID-set equality, per-point revision + fingerprint +
  tenant, and the ABSENCE of points for deleted/superseded/dirty/suppressed/
  unowned rows. An empty dataset is accepted only when the manifest row names
  the generation.
- ``cutover`` requires ``--yes`` and a green verify for BOTH kinds, then FLIPS
  the pointer for both kinds in ONE transaction (the expand only WRITES the new
  rows: until this flip the install keeps serving its old generation, so an
  un-migrated one fails loud instead of returning zero hits) — it also blocks
  the intents that still target a retired generation (R27) — and writes a
  rollback marker.

Quiesce (R25): every command except ``inventory`` refuses to run while the app
answers on ``settings.APP_PORT`` or another migration holds ``migrate.lock``
(stale locks are detected by pid liveness). No middleware, no zero-downtime
promise — stop the app, migrate, start the app.

Exit codes: 0 the command did what it says; 1 the command ran but a gate failed
(verify findings, a cutover the findings blocked, a backfill that stopped before
every batch was acked); 2 refused before doing anything (quiesce, lock, --yes,
bad input).

    python scripts/migrate_qdrant.py inventory --out report.json
    python scripts/migrate_qdrant.py backup --dir /backups
    python scripts/migrate_qdrant.py backfill --kind memory --batch 200 [--resume]
    python scripts/migrate_qdrant.py verify --kind memory
    python scripts/migrate_qdrant.py cutover --yes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import socket
import sqlite3
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qdrant_client import models as qm  # noqa: E402
from sqlalchemy import create_engine, event, select, text, update  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402
from sqlalchemy.exc import OperationalError  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402
from sqlalchemy.pool import NullPool  # noqa: E402

from app import database  # noqa: E402
from app.config import settings  # noqa: E402
from app.models.conversation import Conversation  # noqa: E402
from app.models.document_chunk import DocumentChunk  # noqa: E402
from app.models.index_outbox import IndexGeneration, IndexOutbox  # noqa: E402
from app.models.memory import Memory, MemorySuppression  # noqa: E402
from app.retrieval import vector_backend, vector_retriever  # noqa: E402
from app.retrieval.embedder import embed_texts_sync  # noqa: E402
from app.retrieval.embedding_fingerprint import (  # noqa: E402
    canonical_fingerprint,
    current_fingerprint,
    fingerprint_generation,
    generation_name,
)
from app.retrieval.memory import outbox, vector_store  # noqa: E402
from app.retrieval.memory.correction import state_of  # noqa: E402
from app.services.erasure_service import ERASURE_STATUS_COMPLETED  # noqa: E402

log = logging.getLogger("migrate_qdrant")

KIND_MEMORY = outbox.KIND_MEMORY
KIND_CHUNK = outbox.KIND_CHUNK
KINDS = (KIND_MEMORY, KIND_CHUNK)

MIGRATE_LOCK_NAME = "migrate.lock"
CHECKPOINT_NAME = "backfill-checkpoint.json"
ROLLBACK_MARKER_NAME = "p1b-rollback-marker.json"
EXPAND_RECORD_NAME = "p1b-expand-record.json"
BACKUP_SUFFIX = "pre-p1b.bak"

APP_PROBE_HOST = "127.0.0.1"
APP_PROBE_TIMEOUT_S = 0.5
VECTOR_SCROLL_PAGE = 128
DEFAULT_BATCH = 200

# Payload revision/contract keys per kind (the P0/P1a spellings, ruling R9).
_REVISION_KEY = {KIND_MEMORY: "orivory_memory_revision", KIND_CHUNK: "revision"}
_CONTRACT_KEY = {KIND_MEMORY: "orivory_embed_generation", KIND_CHUNK: "fingerprint"}


class MigrationRefused(RuntimeError):
    """The CLI refused to run — quiesce, a lock, or a bad invocation. Exit code 2."""


class VerifyFailed(MigrationRefused):
    """A read-side gate found findings, so the command did not run. Exit code 1.

    Distinct from a refusal on purpose: "the store is not ready" is a finding
    the operator reads out of the report, not a misuse of the command.
    """


# ── where things live (settings are the one source of truth) ────────────────


def is_sqlite() -> bool:
    return settings.DATABASE_URL.startswith("sqlite")


def db_file() -> Path | None:
    """The SQLite file this CLI migrates, or None for a server database."""
    if not is_sqlite():
        return None
    return Path(make_url(settings.DATABASE_URL).database)


def _sidecar(name: str) -> Path:
    path = db_file()
    if path is None:
        return Path.cwd() / name
    return Path(f"{path}.{name}")


def lock_path() -> Path:
    """``migrate.lock`` beside the SQLite file (pid inside), cwd for servers."""
    return _sidecar(MIGRATE_LOCK_NAME)


def checkpoint_path() -> Path:
    return _sidecar(CHECKPOINT_NAME)


def rollback_marker_path() -> Path:
    return _sidecar(ROLLBACK_MARKER_NAME)


def expand_record_path() -> Path:
    return _sidecar(EXPAND_RECORD_NAME)


def fingerprint_token() -> str:
    """The 64-char contract token every generation row / payload must carry."""
    return fingerprint_generation(current_fingerprint())


def data_generation(kind: str) -> str:
    """The generation this migration builds for ``kind`` (the live contract)."""
    return generation_name(kind)


# ── sessions (sync engine; the CLI never uses the app's async engine) ───────


@contextmanager
def _session(readonly: bool = False) -> Iterator[Session]:
    url = settings.DATABASE_URL
    if url.startswith("sqlite"):
        path = make_url(url).database
        if readonly:
            # Strictly read-only (R28): the file cannot be written, not even a
            # journal — `inventory` must never touch the operator's data.
            engine = create_engine(
                f"sqlite:///file:{path}?mode=ro&uri=true", poolclass=NullPool
            )
            try:
                with engine.connect() as probe:
                    probe.execute(text("SELECT 1"))
            except OperationalError as exc:
                engine.dispose()
                raise MigrationRefused(
                    "cannot open the database read-only: a non-empty WAL needs its "
                    "`-shm` sidecar, which this file does not have. Checkpoint the "
                    "copy first (`sqlite3 <db> 'PRAGMA wal_checkpoint(TRUNCATE)'`), or "
                    f"run `backup` first and inventory the snapshot ({exc})"
                ) from exc
        else:
            engine = create_engine(
                url.replace("+aiosqlite", ""),
                connect_args={"check_same_thread": False},
                poolclass=NullPool,
            )
            event.listen(engine, "connect", database._configure_sqlite_connection)
    else:
        engine = create_engine(url.replace("+asyncpg", "+psycopg2"), pool_pre_ping=True)
    session = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


# ── quiesce: the app must be down, and one migration at a time (R25) ────────


def app_is_listening() -> bool:
    """True when something answers on the app's port — the app is still alive."""
    sock = socket.socket()
    sock.settimeout(APP_PROBE_TIMEOUT_S)
    try:
        sock.connect((APP_PROBE_HOST, int(settings.APP_PORT)))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _lock_pid(path: Path) -> int | None:
    try:
        return int(path.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def require_quiesced() -> None:
    """Refuse while the app is running (R25: stop app -> migrate -> start app)."""
    if app_is_listening():
        raise MigrationRefused(
            f"the app is still listening on {APP_PROBE_HOST}:{settings.APP_PORT} — "
            "stop it before migrating (the migration needs a quiesced store)"
        )


@contextmanager
def _maintenance_lock() -> Iterator[None]:
    """One migration owner at a time; a dead owner's lock is taken over."""
    path = lock_path()
    if path.exists():
        pid = _lock_pid(path)
        if pid is not None and _pid_alive(pid):
            raise MigrationRefused(f"another migration holds {path} (pid {pid})")
        log.warning("removing stale %s (pid %s is not running)", path, pid)
        path.unlink(missing_ok=True)
    path.write_text(f"{os.getpid()}\n")
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


@contextmanager
def _offline() -> Iterator[None]:
    """Quiesce probe + maintenance lock: the guard every mutating command uses."""
    require_quiesced()
    with _maintenance_lock():
        yield


# ── SQL side: eligibility, the one definition per kind ──────────────────────


def _suppressed_sources(session: Session) -> set[tuple[str, str]]:
    return {
        (row.user_id.hex, row.source_ref)
        for row in session.execute(select(MemorySuppression)).scalars()
    }


def memory_rows(session: Session) -> dict[str, dict]:
    """``{memory_id: {...}}`` for EVERY memory row, eligible or not.

    Eligible = the read path's serve set: ``state_of`` current or needs-check
    (superseded and dirty are history/never-served), and NOT a projection whose
    source identity the user suppressed (a forgotten source must not keep a
    servable vector, R28).
    """
    suppressed = _suppressed_sources(session)
    rows: dict[str, dict] = {}
    for memory in session.execute(select(Memory)).scalars():
        state = state_of(memory)
        reason = None
        if state == "superseded":
            reason = "superseded"
        elif state == "dirty":
            reason = "dirty"
        elif memory.source_ref and (memory.user_id.hex, memory.source_ref) in suppressed:
            reason = "suppressed"
        rows[str(memory.id)] = {
            "revision": int(memory.revision or 0),
            "reason": reason,
            "tenant": str(memory.user_id),
            "state": state,
        }
    return rows


def chunk_rows(session: Session) -> dict[str, dict]:
    """``{chunk_id: {...}}`` for EVERY chunk row, eligible or not.

    Eligible = the drain's own owner rule (``outbox._chunk_owner``): the chunk's
    ``chunk_metadata.conversation_id`` resolves to a live conversation, which
    yields the tenant the payload is scoped to. No owner means no indexable
    point — the same call the durable drain makes.
    """
    chunk_data = session.execute(
        select(DocumentChunk.id, DocumentChunk.chunk_metadata, DocumentChunk.revision)
    ).all()
    conversations: dict[uuid.UUID, uuid.UUID] = {}
    parsed: dict[str, uuid.UUID | None] = {}
    conversation_ids: set[uuid.UUID] = set()
    for chunk_id, metadata, _revision in chunk_data:
        raw = (metadata or {}).get("conversation_id")
        try:
            conversation = uuid.UUID(str(raw))
        except (TypeError, ValueError, AttributeError):
            conversation = None
        parsed[str(chunk_id)] = conversation
        if conversation is not None:
            conversation_ids.add(conversation)
    if conversation_ids:
        conversations = {
            row.id: row.user_id
            for row in session.execute(
                select(Conversation.id, Conversation.user_id).where(
                    Conversation.id.in_(conversation_ids)
                )
            )
        }
    rows: dict[str, dict] = {}
    for chunk_id, _metadata, revision in chunk_data:
        conversation = parsed[str(chunk_id)]
        owner = conversations.get(conversation) if conversation is not None else None
        rows[str(chunk_id)] = {
            "revision": int(revision or 0),
            "reason": None if owner is not None else "unowned",
            "tenant": str(owner) if owner is not None else None,
            "conversation_id": str(conversation) if conversation is not None else None,
        }
    return rows


def correction_chains(session: Session) -> list[dict]:
    """Supersede chains an operator needs to read the inventory (spec §6.2.1)."""
    chains: list[dict] = []
    for memory in session.execute(select(Memory)).scalars():
        metadata = dict(memory.extra_metadata or {})
        superseded_by = metadata.get("cm_superseded_by")
        supersedes = metadata.get("cm_supersedes")
        if not superseded_by and not supersedes:
            continue
        if supersedes and not isinstance(supersedes, list):
            supersedes = [supersedes]
        chains.append(
            {
                "memory_id": str(memory.id),
                "superseded_by": str(superseded_by) if superseded_by else None,
                "supersedes": [str(value) for value in (supersedes or [])],
            }
        )
    return chains


def sql_rows(session: Session, kind: str) -> dict[str, dict]:
    return memory_rows(session) if kind == KIND_MEMORY else chunk_rows(session)


def manifest_rows(session: Session, kind: str) -> list[IndexGeneration]:
    return list(
        session.execute(
            select(IndexGeneration).where(IndexGeneration.kind == kind)
        ).scalars()
    )


def active_generation_name(session: Session, kind: str) -> str | None:
    """The generation the runtime serves for ``kind`` right now (or None)."""
    rows = manifest_rows(session, kind)
    for row in rows:
        if row.is_active:
            return row.generation
    return None


# ── vector side: read, audit, write ─────────────────────────────────────────


def _client():
    return vector_backend.get_sync_client()


def scroll_points(generation: str):
    """Every point of ``generation``, paginated. Missing collection = nothing."""
    client = _client()
    if not client.collection_exists(generation):
        return
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=generation, limit=VECTOR_SCROLL_PAGE, offset=offset,
            with_payload=True, with_vectors=False,
        )
        if not points:
            return
        yield from points
        if offset is None:
            return


def _payload_contract_ok(kind: str, payload: dict, canonical: str, token: str) -> bool:
    value = payload.get(_CONTRACT_KEY[kind])
    if not isinstance(value, str) or not value:
        return False
    if kind == KIND_MEMORY:
        # Memory payloads carry BOTH spellings (canonical contract + token).
        recorded = payload.get("orivory_embed_fingerprint")
        return value == token and recorded == canonical
    return value == canonical


def _payload_identity_ok(kind: str, point_id: str, payload: dict) -> bool:
    raw = payload.get("memory_id") if kind == KIND_MEMORY else payload.get("chunk_id")
    try:
        return str(uuid.UUID(str(raw))) == point_id
    except (TypeError, ValueError, AttributeError):
        return False


def audit_collection(kind: str, generation: str, rows: dict[str, dict]) -> tuple[dict, dict]:
    """One full read-side pass over ``generation`` against the SQL rows.

    Returns ``(audit, point_tenants)``. Read-only by construction: it scrolls,
    it never creates and never writes.
    """
    canonical = canonical_fingerprint(current_fingerprint())
    token = fingerprint_token()
    audit = {
        "exists": True,
        "points": 0,
        "missing": [],
        "stale": [],
        "orphan": [],
        "excluded_present": [],
        "excluded_reasons": {},
        "unknown_fingerprint": [],
        "malformed_id": [],
        "tenant_mismatch": [],
    }
    seen: set[str] = set()
    tenants: dict[str, str] = {}
    for point in scroll_points(generation):
        point_id = str(point.id)
        payload = dict(point.payload or {})
        seen.add(point_id)
        tenants[point_id] = str(payload.get("user_id") or "unknown")
        if not _payload_identity_ok(kind, point_id, payload):
            # Quarantine: this point cannot even be named, so it is NOT
            # classified further (calling it an orphan would claim knowledge of
            # a row we cannot identify). It must simply not be present.
            audit["malformed_id"].append(point_id)
            continue
        if not _payload_contract_ok(kind, payload, canonical, token):
            audit["unknown_fingerprint"].append(point_id)
        record = rows.get(point_id)
        if record is None:
            audit["orphan"].append(point_id)
        elif record["reason"] is not None:
            audit["excluded_present"].append(point_id)
            audit["excluded_reasons"][point_id] = record["reason"]
        else:
            if int(payload.get(_REVISION_KEY[kind]) or -1) != int(record["revision"]):
                audit["stale"].append(point_id)
            if str(payload.get("user_id") or "") != str(record.get("tenant") or ""):
                # The payload tenant IS the read path's filter (the security
                # boundary): a drifted point is invisible to its owner and
                # visible to someone else. Reported independently of the
                # revision check — it is the more dangerous finding.
                audit["tenant_mismatch"].append(point_id)
    audit["points"] = len(seen)
    # Missing = an ELIGIBLE SQL row with no point. The ineligible rows are the
    # absence check instead: a point of ours that they must not have.
    audit["missing"] = sorted(
        entity_id for entity_id, record in rows.items()
        if record["reason"] is None and entity_id not in seen
    )
    return audit, tenants


def _tenant_report(rows: dict[str, dict], audit: dict, tenants: dict[str, str]) -> dict:
    report: dict[str, dict] = {}

    def bucket(tenant: str) -> dict:
        return report.setdefault(
            tenant, {"eligible": 0, "total": 0, "points": 0, "missing": 0, "stale": 0, "orphan": 0}
        )

    for record in rows.values():
        entry = bucket(record.get("tenant") or "unknown")
        entry["total"] += 1
        if record["reason"] is None:
            entry["eligible"] += 1
    for point_id in tenants:
        bucket(tenants[point_id])["points"] += 1
    for point_id in audit["missing"]:
        bucket(rows[point_id].get("tenant") or "unknown")["missing"] += 1
    for point_id in audit["stale"]:
        bucket(rows[point_id].get("tenant") or "unknown")["stale"] += 1
    for point_id in audit["orphan"]:
        bucket(tenants.get(point_id, "unknown"))["orphan"] += 1
    return report


# ── expand (spec §6.2 step 4): schema v3 + the two generation rows ──────────


def _record_pre_expand_pointer() -> None:
    """Record ONCE which generation the install served before the expand.

    The rollback marker is only useful with the origin of the migration, and the
    first expand is the only moment it can be read: a rolled-back install (T6)
    reads the marker, and the pointer it restores is the one recorded here. On
    the current SQLite ladder the expand no longer moves the pointer (F1), so
    this value normally equals the live pointer at cutover time — recording it
    keeps the marker meaningful for installs expanded by an earlier build and
    for the Postgres path, where the expand is the only pointer write.
    """
    path = expand_record_path()
    if path.exists():
        return
    previous = {kind: _fallback_generation(kind) for kind in KINDS}
    try:
        with _session(readonly=is_sqlite()) as session:
            previous = {
                kind: active_generation_name(session, kind) or _fallback_generation(kind)
                for kind in KINDS
            }
    except Exception as exc:
        log.warning("no index_generations table yet (%s): recording the transitional names", exc)
    path.write_text(json.dumps(
        {"recorded_at": datetime.now(UTC).isoformat(), "previous": previous}, indent=2
    ))


def _expand() -> None:
    """Bring the database to the expand state: ladder v3, the two new rows.

    The expand NEVER moves the pointer (F1): the rows are written inactive, the
    install keeps serving its old generation, and ``cutover`` is what flips
    them. The SQLite ladder is shared with the app
    (``database.upgrade_sqlite_schema``) so the CLI can never upgrade
    differently; Postgres has no ladder, so the rows are written directly
    (ruling R10).
    """
    _record_pre_expand_pointer()
    if is_sqlite():
        with _session() as session:
            database.upgrade_sqlite_schema(session.connection())
            session.commit()
        return
    with _session() as session:
        with session.begin():
            database.activate_generations(session, activate=False)


# ── inventory ───────────────────────────────────────────────────────────────


def inventory(*, out: Path | None = None) -> dict:
    """Strictly read-only counts + quarantine lists (R28). Writes only ``out``."""
    report: dict = {
        "generated_at": datetime.now(UTC).isoformat(),
        "db": {"dialect": "sqlite" if is_sqlite() else "postgresql",
               "path": str(db_file()) if db_file() else None},
        "fingerprint": {"token": fingerprint_token(),
                        "canonical": canonical_fingerprint(current_fingerprint()),
                        "dim": int(current_fingerprint()["dim"])},
        "kinds": {},
    }
    with _session(readonly=True) as session:
        version = None
        if is_sqlite():
            version = int(session.execute(text("PRAGMA user_version")).scalar_one())
        report["db"]["user_version"] = version
        for kind in KINDS:
            rows = sql_rows(session, kind)
            active = active_generation_name(session, kind) or _fallback_generation(kind)
            target = data_generation(kind)
            names = [active] + ([target] if target != active else [])
            collections = {}
            audits = {}
            for name in names:
                if not _client().collection_exists(name):
                    collections[name] = {
                        "exists": False, "points": 0,
                        "missing": sorted(
                            entity_id for entity_id, record in rows.items()
                            if record["reason"] is None
                        ),
                        "stale": [], "orphan": [], "excluded_present": [],
                        "excluded_reasons": {}, "unknown_fingerprint": [],
                        "malformed_id": [], "tenant_mismatch": [],
                    }
                    audits[name] = (collections[name], {})
                    continue
                audit, tenants = audit_collection(kind, name, rows)
                collections[name] = audit
                audits[name] = (audit, tenants)
            counts = {"total": len(rows), "eligible": 0, "superseded": 0, "dirty": 0,
                      "suppressed": 0, "unowned": 0}
            for record in rows.values():
                if record["reason"] is None:
                    counts["eligible"] += 1
                else:
                    counts[record["reason"]] = counts.get(record["reason"], 0) + 1
            primary = audits.get(target) or audits.get(active)
            kind_report = {
                "sql": counts,
                "per_tenant": _tenant_report(rows, primary[0], primary[1]) if primary else {},
                "generations": {"active": active, "target": target},
                "collections": collections,
            }
            if kind == KIND_MEMORY:
                kind_report["correction_chains"] = correction_chains(session)
            report["kinds"][kind] = kind_report
    if out is not None:
        Path(out).write_text(json.dumps(report, indent=2, default=str))
    return report


def _fallback_generation(kind: str) -> str:
    """The transitional spelling an un-migrated install still serves."""
    return outbox.TARGET_GENERATION if kind == KIND_MEMORY else outbox.CHUNK_TARGET_GENERATION


# ── backup (spec §6.3) ──────────────────────────────────────────────────────


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_tree(source: Path, dest: Path) -> list[dict]:
    files: list[dict] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        target = dest / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        files.append({"path": str(target.relative_to(dest.parent)), "sha256": _sha256(target),
                      "bytes": target.stat().st_size})
    return files


def _backup_sources() -> dict[str, Path]:
    """The real directories ``backup`` copies, keyed by the manifest label.

    A blank/unset setting is NOT ``Path(".")``: ``Path("")`` passes ``is_dir()``,
    so ``_copy_tree`` would copy the WHOLE working directory into the backup
    (sha256-ing every file of it — and recursing into its own destination when
    ``--dir`` sits inside the CWD).
    """
    configured = {
        "uploads": settings.FS_STORAGE_PATH,
        "chroma": getattr(settings, "CHROMA_LOCAL_PATH", ""),
    }
    sources: dict[str, Path] = {}
    for label, value in configured.items():
        text = str(value or "").strip()
        if text and Path(text).is_dir():
            sources[label] = Path(text).resolve()
    return sources


# ── the backup's own records: contract stamp + deletion ledger ──────────────


def _norm_id(value: object) -> str:
    """Compare ids the way SQLite stores them: hex, no dashes, lowercase."""
    return str(value or "").replace("-", "").strip().lower()


def _json_list(value: object) -> list:
    """A JSON column read straight from the driver comes back as text.

    Iterating a str yields characters, so an unparsed ``requested_memory_ids``
    would silently compare nothing and make the absence check pass — the one
    direction a delete ledger must never fail in.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return list(value) if isinstance(value, (list, tuple)) else []


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def _active_contract_rows(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """``[(kind, fingerprint)]`` for every ACTIVE generation row, or nothing."""
    if "index_generations" not in _table_names(conn):
        return []
    return [
        (str(kind), str(fingerprint or ""))
        for kind, fingerprint in conn.execute(
            "SELECT kind, fingerprint FROM index_generations WHERE is_active = 1"
        )
    ]


def _active_tokens(conn: sqlite3.Connection) -> dict[str, str | None]:
    """The token of the ACTIVE generation per kind — ``None`` when there is none.

    Recorded at backup time and reproduced from the restore, so a restore that
    silently changed (or lost) the pointer is a finding, not a guess.
    """
    tokens: dict[str, str | None] = {kind: None for kind in KINDS}
    for kind, token in _active_contract_rows(conn):
        if kind in tokens and token and tokens[kind] is None:
            tokens[kind] = token
    return tokens


def _ledger_digest(db_path: str | Path) -> dict:
    """Digest of the deletion/suppression ledger the backup carries.

    The ledger — not the vectors — is what stops a restore from resurrecting
    content the user deleted or forgot (§5.4/§12.3), so the drill compares this
    record against the restored bytes and refuses on drift.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = _table_names(conn)
        digest: dict[str, dict] = {}
        if "memory_suppressions" in tables:
            rows = sorted(
                f"{_norm_id(user_id)}|{source_ref}|{reason}"
                for user_id, source_ref, reason in conn.execute(
                    "SELECT user_id, source_ref, reason FROM memory_suppressions"
                )
            )
            digest["suppressions"] = {
                "present": True, "count": len(rows),
                "sha256": hashlib.sha256("\n".join(rows).encode()).hexdigest(),
            }
        else:
            digest["suppressions"] = {"present": False, "count": 0,
                                      "sha256": hashlib.sha256(b"").hexdigest()}
        if "erasure_receipts" in tables:
            rows = sorted(
                f"{receipt_id}|{status}|"
                f"{','.join(sorted(_norm_id(value) for value in _json_list(ids)))}"
                for receipt_id, status, ids in conn.execute(
                    "SELECT id, status, requested_memory_ids FROM erasure_receipts"
                )
            )
            digest["erasure_receipts"] = {
                "present": True, "count": len(rows),
                "sha256": hashlib.sha256("\n".join(rows).encode()).hexdigest(),
            }
        else:
            digest["erasure_receipts"] = {"present": False, "count": 0,
                                          "sha256": hashlib.sha256(b"").hexdigest()}
        return digest
    finally:
        conn.close()


def _refresh_manifest(manifest_path: Path, snapshot: Path, recorded: dict) -> dict:
    """Add the drill's records to a manifest that predates them, from the bytes.

    Only derived records are ever added (they are recomputed from the verified
    snapshot), and only when they are missing: an operator's existing manifest
    is otherwise left exactly as it is.
    """
    conn = sqlite3.connect(f"file:{snapshot}?mode=ro", uri=True)
    try:
        recorded.setdefault("db", {})["fingerprint"] = _active_tokens(conn)
    finally:
        conn.close()
    recorded.setdefault("deletion_ledger", _ledger_digest(snapshot))
    recorded["refreshed_at"] = datetime.now(UTC).isoformat()
    manifest_path.write_text(json.dumps(recorded, indent=2, default=str))
    log.info("refreshed %s with the fingerprint and ledger records the drill reads",
             manifest_path)
    return recorded


def backup(*, dest_dir: Path) -> dict:
    """VACUUM INTO snapshot + checksum manifest; resumable, never overwriting."""
    if not is_sqlite():
        raise MigrationRefused("backup --dir snapshots a SQLite install; server mode uses Qdrant snapshots")
    db_path = db_file()
    assert db_path is not None
    dest_dir = Path(dest_dir).resolve()
    with _offline():
        sources = _backup_sources()
        for label, source in sources.items():
            if dest_dir == source or dest_dir.is_relative_to(source):
                raise MigrationRefused(
                    f"--dir {dest_dir} is inside the {label} source tree {source} — "
                    "the copy would recurse into its own destination; choose a "
                    "destination outside every source tree"
                )
        dest_dir.mkdir(parents=True, exist_ok=True)
        snapshot = dest_dir / f"{db_path.name}.{BACKUP_SUFFIX}"
        manifest_path = Path(f"{snapshot}.manifest.json")
        report = {
            "dir": str(dest_dir),
            "db_backup": str(snapshot),
            "manifest": str(manifest_path),
            "files": [],
            "missing": [],
            "reused": False,
        }
        if snapshot.exists():
            if snapshot.stat().st_size == 0 or not os.access(snapshot, os.R_OK):
                raise MigrationRefused(f"existing backup {snapshot} is empty or unreadable")
            if not manifest_path.exists():
                raise MigrationRefused(
                    f"existing backup {snapshot} has no checksum manifest — remove it to take a fresh one"
                )
            recorded = json.loads(manifest_path.read_text())
            if recorded.get("db", {}).get("sha256") != _sha256(snapshot):
                raise MigrationRefused(
                    f"existing backup {snapshot} does not match its manifest — refusing to reuse it"
                )
            if "deletion_ledger" not in recorded or "fingerprint" not in recorded.get("db", {}):
                # A manifest written before the restore drill learned to read
                # these: the bytes are verified, so the records are recomputed
                # from them instead of leaving the drill unable to pass.
                recorded = _refresh_manifest(manifest_path, snapshot, recorded)
            log.info("reusing the verified backup at %s", snapshot)
            report["reused"] = True
            report["files"] = recorded.get("files", [])
            report["missing"] = recorded.get("missing", [])
            return report

        # VACUUM INTO includes committed WAL frames: one file, self-contained.
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("VACUUM INTO ?", (str(snapshot),))
        finally:
            conn.close()

        files: list[dict] = []
        for label, source in sources.items():
            files.extend(_copy_tree(source, dest_dir / label))
        missing: list[str] = [
            label for label in ("uploads", "chroma") if label not in sources
        ]

        snapshot_conn = sqlite3.connect(f"file:{snapshot}?mode=ro", uri=True)
        try:
            user_version = snapshot_conn.execute("PRAGMA user_version").fetchone()[0]
            integrity = snapshot_conn.execute("PRAGMA integrity_check").fetchone()[0]
            memories = snapshot_conn.execute("SELECT count(*) FROM memories").fetchone()[0]
            fingerprint = _active_tokens(snapshot_conn)
        finally:
            snapshot_conn.close()
        manifest = {
            "created_at": datetime.now(UTC).isoformat(),
            "db": {
                "source": str(db_path),
                "name": snapshot.name,
                "sha256": _sha256(snapshot),
                "bytes": snapshot.stat().st_size,
                "user_version": int(user_version),
                "integrity_check": integrity,
                "memories": int(memories),
                # The contract the snapshot's ACTIVE generation served, per kind
                # (``None`` = no active row). The restore drill reproduces this
                # from the restored bytes and refuses on drift.
                "fingerprint": fingerprint,
            },
            # What the ledger held at backup time: a restore that cannot prove
            # forgotten content stays forgotten is not a restore (R33).
            "deletion_ledger": _ledger_digest(snapshot),
            "files": files,
            "missing": missing,
            "router_schema_version": database.SQLITE_SCHEMA_VERSION,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
        report["files"] = files
        report["missing"] = missing
        if integrity != "ok":
            raise MigrationRefused(f"backup {snapshot} failed integrity_check: {integrity}")
        return report


# ── backfill (spec §6.2 steps 5-6) ──────────────────────────────────────────


def _read_checkpoint() -> dict:
    path = checkpoint_path()
    if not path.exists():
        return {"version": 1, "checkpoints": {}}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        raise MigrationRefused(f"checkpoint {path} is not readable JSON") from None


def _write_checkpoint(state: dict) -> None:
    checkpoint_path().write_text(json.dumps(state, indent=2, default=str))


def _checkpoint_key(kind: str) -> str:
    return f"{data_generation(kind)}|{kind}"


def _eligible_ids(session: Session, kind: str) -> set[str]:
    return {
        entity_id for entity_id, record in sql_rows(session, kind).items()
        if record["reason"] is None
    }


def _backfill_page(session: Session, kind: str, last_id: uuid.UUID | None, batch: int) -> list:
    """One keyset page, ordered by primary key — never OFFSET."""
    model = Memory if kind == KIND_MEMORY else DocumentChunk
    stmt = select(model).order_by(model.id).limit(batch)
    if last_id is not None:
        stmt = stmt.where(model.id > last_id)
    return list(session.execute(stmt).scalars())


def backfill(*, kind: str, batch: int = DEFAULT_BATCH, resume: bool = False) -> dict:
    """Build the contract generation with a keyset scan; checkpoint per batch."""
    if kind not in KINDS:
        raise MigrationRefused(f"unknown kind {kind!r}; expected one of {list(KINDS)}")
    if batch < 1:
        raise MigrationRefused("--batch must be >= 1")
    started = time.monotonic()
    with _offline():
        _expand()
        generation = data_generation(kind)
        vector_backend.ensure_collection(kind, generation, int(current_fingerprint()["dim"]))
        key = _checkpoint_key(kind)
        state = _read_checkpoint()
        entry = state["checkpoints"].get(key) if resume else None
        last_id = uuid.UUID(entry["last_id"]) if entry else None
        resumed_from = entry["last_id"] if entry else None

        report = {
            "kind": kind, "generation": generation, "batch": batch,
            "resumed_from": resumed_from, "batches": 0, "upserted": 0,
            "excluded": 0, "skipped_unowned": 0, "gc_deleted": 0,
            "last_id": None, "errors": [], "complete": False, "elapsed_s": None,
            "rows_per_sec": None,
        }
        with _session() as session:
            # ponytail: the eligibility map is O(rows) in RAM; page-local
            # eligibility if a deployment ever backs up millions of rows.
            rows_meta = sql_rows(session, kind)
            while True:
                page = _backfill_page(session, kind, last_id, batch)
                if not page:
                    report["complete"] = True
                    break
                report["batches"] += 1
                eligible, skipped = [], 0
                for row in page:
                    record = rows_meta.get(str(row.id))
                    if record is None or record["reason"] is not None:
                        if record is not None and record["reason"] == "unowned":
                            skipped += 1
                        else:
                            report["excluded"] += 1
                        continue
                    eligible.append(row)
                report["skipped_unowned"] += skipped
                if eligible:
                    points = _points_for(kind, eligible, rows_meta)
                    result = _client().upsert(collection_name=generation, points=points)
                    status = str(getattr(result, "status", "completed")).lower()
                    if status.split(".")[-1] not in {"completed", "acknowledged", "updated"}:
                        report["errors"].append(f"batch at {last_id}: upsert status {status}")
                        break
                    report["upserted"] += len(points)
                last_id = page[-1].id
                report["last_id"] = str(last_id)
                entry = state["checkpoints"].setdefault(key, {})
                entry.update({
                    "generation": generation, "kind": kind,
                    "tenant": _page_tenant(kind, page[-1], rows_meta),
                    "last_id": str(last_id),
                })
                _write_checkpoint(state)
        with _session() as session:
            report["gc_deleted"] = _gc_stale(kind, generation, _eligible_ids(session, kind))
        report["elapsed_s"] = round(time.monotonic() - started, 3)
        report["rows_per_sec"] = round(
            report["upserted"] / report["elapsed_s"], 2) if report["elapsed_s"] else None
        return report


def _page_tenant(kind: str, row, rows_meta: dict[str, dict]) -> str:
    record = rows_meta.get(str(row.id)) or {}
    return record.get("tenant") or "unknown"


def _points_for(kind: str, rows: list, rows_meta: dict[str, dict]) -> list:
    """The production payload contract, never a second spelling of it (R9)."""
    if kind == KIND_MEMORY:
        documents = [vector_store._memory_to_document(memory) for memory in rows]
        vectors = embed_texts_sync(documents)
        return [
            vector_store._point(memory, vectors[index], documents[index])
            for index, memory in enumerate(rows)
        ]
    vectors = embed_texts_sync([chunk.content for chunk in rows])
    return [
        vector_retriever._point(  # the chunk payload contract, one definition
            chunk, vectors[index], user_id=rows_meta[str(chunk.id)]["tenant"])
        for index, chunk in enumerate(rows)
    ]


def _gc_stale(kind: str, generation: str, eligible: set[str]) -> int:
    """Delete points whose SQL row is not eligible (R29b) — and say how many."""
    client = _client()
    stale = [point_id for point_id in
             (str(point.id) for point in scroll_points(generation)) if point_id not in eligible]
    if not stale:
        return 0
    client.delete(collection_name=generation, points_selector=qm.PointIdsList(points=stale))
    log.info("GC deleted %d stale point(s) from %s", len(stale), generation)
    return len(stale)


# ── verify (spec §6.2 step 7, ruling R28) ───────────────────────────────────


def _verify(kind: str) -> dict:
    """The audit itself (no quiesce/lock): cutover reuses it in-process."""
    generation = data_generation(kind)
    report: dict = {
        "kind": kind, "generation": generation, "ok": False, "no_manifest_row": False,
        "sql_eligible": 0, "points": 0,
        "findings": {"missing": [], "stale_revision": [], "fingerprint": [], "absence": [],
                     "tenant_mismatch": [], "malformed_id": []},
    }
    with _session(readonly=True) as session:
        rows = sql_rows(session, kind)
        eligible = {entity_id for entity_id, record in rows.items() if record["reason"] is None}
        manifest = [
            row for row in manifest_rows(session, kind) if row.generation == generation
        ]
        active = active_generation_name(session, kind)
    report["sql_eligible"] = len(eligible)
    report["active_generation"] = active
    report["manifest_fingerprint"] = manifest[0].fingerprint if manifest else None
    if not manifest:
        # An empty dataset is only trustworthy when a manifest row names it.
        report["no_manifest_row"] = True
        return report
    if manifest[0].fingerprint != fingerprint_token():
        report["findings"]["fingerprint"].append(f"manifest:{manifest[0].fingerprint}")
    audit, _tenants = audit_collection(kind, generation, rows)
    report["points"] = audit["points"]
    report["findings"]["missing"] = audit["missing"]
    report["findings"]["stale_revision"] = audit["stale"]
    report["findings"]["fingerprint"] += [
        point_id for point_id in audit["unknown_fingerprint"] if point_id in eligible
    ]
    report["findings"]["absence"] = sorted(
        set(audit["orphan"]) | set(audit["excluded_present"])
    )
    report["findings"]["tenant_mismatch"] = audit["tenant_mismatch"]
    report["findings"]["malformed_id"] = audit["malformed_id"]
    report["ok"] = not any(report["findings"].values())
    return report


def verify(*, kind: str) -> dict:
    """Full read-side audit of the contract generation for ``kind``."""
    if kind not in KINDS:
        raise MigrationRefused(f"unknown kind {kind!r}; expected one of {list(KINDS)}")
    with _offline():
        return _verify(kind)


# ── restore drill (spec §12, ruling R33) ────────────────────────────────────

def _check(findings: list[str]) -> dict:
    return {"ok": not findings, "findings": findings}


def _drill_checksum(manifest: dict, backup_dir: Path, target: Path, restored_db: Path) -> dict:
    """Every byte the manifest recorded must be present and unchanged."""
    findings: list[str] = []
    recorded_db = manifest.get("db", {}).get("sha256")
    if not recorded_db:
        findings.append("the manifest records no database checksum")
    elif not restored_db.is_file():
        findings.append(f"the restored database {restored_db.name} is missing")
    elif _sha256(restored_db) != recorded_db:
        findings.append(f"the restored database {restored_db.name} does not match its manifest")
    for entry in manifest.get("files", []):
        source, dest = backup_dir / entry["path"], target / entry["path"]
        if not source.is_file():
            findings.append(f"{entry['path']} is recorded but missing from the backup")
        elif not dest.is_file():
            findings.append(f"{entry['path']} was not restored")
        elif _sha256(dest) != entry.get("sha256"):
            findings.append(f"{entry['path']} does not match its checksum")
    return _check(findings)


def _drill_integrity(path: Path) -> dict:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        result = conn.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        conn.close()
    return _check([] if result == "ok" else [f"PRAGMA integrity_check says {result!r}"])


def _drill_foreign_keys(path: Path) -> dict:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        conn.close()
    return _check([f"PRAGMA foreign_key_check: {row}" for row in violations[:5]])


def _drill_fingerprint(manifest: dict, path: Path) -> dict:
    """The contract stamp the backup recorded must survive the restore.

    A restore that silently changed the ladder version or the active generation
    pointer serves a different contract than the one the backup was taken
    under — the drill refuses to call that ready.
    """
    findings: list[str] = []
    recorded = manifest.get("db", {}).get("fingerprint")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        user_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        contract = _active_contract_rows(conn)
        live = _active_tokens(conn)
    finally:
        conn.close()
    if not isinstance(recorded, dict):
        findings.append("the manifest records no embedding fingerprint — re-run `backup`")
    else:
        for kind in KINDS:
            if live.get(kind) != recorded.get(kind):
                findings.append(
                    f"{kind}: the restore's active contract {live.get(kind)!r} does not "
                    f"match the recorded {recorded.get(kind)!r}"
                )
    for kind in KINDS:
        active = [token for row_kind, token in contract if row_kind == kind]
        if len(active) > 1:
            findings.append(f"{kind}: {len(active)} active generation rows — exactly one owner")
        if any(not token for token in active):
            findings.append(f"{kind}: an active generation row carries no fingerprint")
    recorded_version = manifest.get("db", {}).get("user_version")
    if recorded_version is not None and int(recorded_version) != user_version:
        findings.append(
            f"the restored schema is user_version={user_version}, the backup recorded "
            f"{recorded_version} — the restore is not the backup"
        )
    return _check(findings)


def _drill_ledger(manifest: dict, path: Path) -> dict:
    """The deletion/suppression ledger must be present AND unchanged (R33)."""
    findings: list[str] = []
    recorded = manifest.get("deletion_ledger")
    live = _ledger_digest(path)
    for name, record in live.items():
        if not record["present"]:
            findings.append(f"the {name} table is missing from the snapshot — a restore "
                            "cannot prove forgotten content stays forgotten")
    if not isinstance(recorded, dict):
        findings.append("the manifest records no deletion/suppression ledger — "
                        "re-run `backup` before trusting a restore")
    elif recorded != live:
        findings.append("the deletion/suppression ledger drifted from the record taken at "
                        f"backup time: {recorded} vs {live}")
    return _check(findings)


def _drill_absence(path: Path) -> dict:
    """GC/absence assertions: the restore must not resurrect what was erased.

    A receipt carrying the app's own ``completed`` status
    (``ERASURE_STATUS_COMPLETED``) is a hard claim that its memories were gone; a
    chunk whose document row no longer exists is GC residue that no reader can
    serve. Either one present in the restore means the bytes are not the state
    the receipt was written against.
    """
    findings: list[str] = []
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = _table_names(conn)
        if "erasure_receipts" in tables:
            present = {_norm_id(row[0]) for row in conn.execute("SELECT id FROM memories")}
            for receipt_id, status, ids in conn.execute(
                "SELECT id, status, requested_memory_ids FROM erasure_receipts"
            ):
                if status != ERASURE_STATUS_COMPLETED:
                    continue
                for memory_id in _json_list(ids):
                    if _norm_id(memory_id) in present:
                        findings.append(
                            f"erasure receipt {receipt_id} erased {memory_id}, which the "
                            "restore resurrects"
                        )
        if "document_chunks" in tables and "documents" in tables:
            orphans = conn.execute(
                "SELECT count(*) FROM document_chunks WHERE document_id NOT IN "
                "(SELECT id FROM documents)"
            ).fetchone()[0]
            if orphans:
                findings.append(f"{orphans} chunk row(s) have no document — GC residue")
    finally:
        conn.close()
    return _check(findings)


def restore_drill(*, backup_dir: Path, target: Path | None = None) -> dict:
    """Restore a ``backup --dir`` volume into a NEW directory and verify it.

    Read-only over the backup volume (checksums are recomputed, never written)
    and it never touches the live database, so it is safe to run with the app
    up. ``target`` defaults to ``<backup_dir>.restore`` beside the backup; it
    must be empty (or absent) and outside ``backup_dir``, because a target
    inside the volume would write into the bytes the drill is verifying.
    """
    if not is_sqlite():
        raise MigrationRefused(
            "restore-drill restores a SQLite backup; a server deployment drills its "
            "snapshot restore with Qdrant's own tooling"
        )
    backup_dir = Path(backup_dir).resolve()
    if not backup_dir.is_dir():
        raise MigrationRefused(f"no backup directory at {backup_dir}")
    snapshots = sorted(backup_dir.glob(f"*.{BACKUP_SUFFIX}"))
    if len(snapshots) != 1:
        raise MigrationRefused(
            f"expected exactly one *.{BACKUP_SUFFIX} snapshot in {backup_dir}, "
            f"found {len(snapshots)}"
        )
    snapshot = snapshots[0]
    manifest_path = Path(f"{snapshot}.manifest.json")
    if not manifest_path.is_file():
        raise MigrationRefused(
            f"{snapshot} has no checksum manifest — a restore that cannot be verified "
            "is not a restore"
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        raise MigrationRefused(f"manifest {manifest_path} is not readable JSON") from None
    target = Path(target).resolve() if target else backup_dir.parent / f"{backup_dir.name}.restore"
    if target.is_relative_to(backup_dir):
        raise MigrationRefused(
            f"target {target} is inside the backup directory {backup_dir} — the drill "
            "never writes into the volume it verifies; choose a target outside it"
        )
    if target.exists() and any(target.iterdir()):
        raise MigrationRefused(
            f"target {target} is not empty — the drill restores into a NEW directory "
            "(the original volume is never written)"
        )
    target.mkdir(parents=True, exist_ok=True)
    restored_db = target / snapshot.name
    shutil.copy2(snapshot, restored_db)
    for entry in manifest.get("files", []):
        source, dest = backup_dir / entry["path"], target / entry["path"]
        if source.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)

    checks = {
        "checksum": _drill_checksum(manifest, backup_dir, target, restored_db),
        "integrity": _drill_integrity(restored_db),
        "foreign_keys": _drill_foreign_keys(restored_db),
        "fingerprint": _drill_fingerprint(manifest, restored_db),
        "ledger": _drill_ledger(manifest, restored_db),
        "absence": _drill_absence(restored_db),
    }
    report = {
        "ok": all(check["ok"] for check in checks.values()),
        "backup_dir": str(backup_dir),
        "manifest": str(manifest_path),
        "target": str(target),
        "database": str(restored_db),
        "checks": checks,
    }
    report["ready"] = report["ok"]
    if not report["ok"]:
        print(
            "restore drill failed: " + "; ".join(
                f"{name}={len(check['findings'])}"
                for name, check in checks.items() if not check["ok"]
            ),
            file=sys.stderr,
        )
    return report


def cutover(*, yes: bool = False) -> dict:
    """FLIP the pointer for BOTH kinds in one transaction, gated on verify."""
    if not yes:
        raise MigrationRefused(
            "cutover flips the active generation for memory AND chunks; re-run with --yes"
        )
    started = time.monotonic()
    with _offline():
        reports = {kind: _verify(kind) for kind in KINDS}
        failed = {kind: report for kind, report in reports.items() if not report["ok"]}
        if failed:
            summary = "; ".join(
                f"{kind}: verify failed ({_finding_summary(report)})"
                for kind, report in failed.items()
            )
            raise VerifyFailed(f"refusing the cutover — {summary}")
        targets = {kind: data_generation(kind) for kind in KINDS}
        with _session(readonly=True) as session:
            # What was serving before the flip: the active row, or the
            # transitional spelling the runtime falls back to when the manifest
            # has no active row — which is exactly the un-migrated state the
            # expand leaves behind (F1). The rollback marker must name it.
            previous = {
                kind: active_generation_name(session, kind) or _fallback_generation(kind)
                for kind in KINDS
            }
        with _session() as session:
            # ONE transaction for both kinds: a crash leaves the old pointer or
            # the new one, never a half-flipped pair.
            with session.begin():
                database.activate_generations(session)
                blocked = _block_stale_intents(session, targets)
        with _session(readonly=True) as session:
            active = {kind: active_generation_name(session, kind) for kind in KINDS}
        marker = {
            "cutover_at": datetime.now(UTC).isoformat(),
            "previous": previous,
            "active": active,
            "pre_migration": _recorded_pre_expand_pointer(),
            "blocked_intents": sum(blocked.values()),
            "verify": {
                kind: {"points": report["points"], "sql_eligible": report["sql_eligible"]}
                for kind, report in reports.items()
            },
        }
        rollback_marker_path().write_text(json.dumps(marker, indent=2, default=str))
        return {
            "previous": previous,
            "active": active,
            "blocked": blocked,
            "blocked_intents": sum(blocked.values()),
            "marker": str(rollback_marker_path()),
            "elapsed_s": round(time.monotonic() - started, 3),
        }


def _recorded_pre_expand_pointer() -> dict | None:
    """The pointer recorded at expand time, when this install has one."""
    path = expand_record_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())["previous"]
    except (json.JSONDecodeError, KeyError):
        return None


def _block_stale_intents(session: Session, targets: dict[str, str]) -> dict[str, int]:
    """Terminal `blocked` for every pending intent a retired generation owns.

    Same transaction as the pointer flip (R27): their writes are already covered
    by the backfill, so leaving them pending would mean replaying them into a
    generation the app no longer serves.
    """
    blocked: dict[str, int] = {}
    for kind, generation in targets.items():
        result = session.execute(
            update(IndexOutbox)
            .where(
                IndexOutbox.kind == kind,
                IndexOutbox.status == "pending",
                IndexOutbox.target_generation != generation,
            )
            .values(
                status="blocked",
                last_error=f"superseded by the P1b cutover (active {generation!r})",
                updated_at=datetime.now(UTC),
            )
        )
        blocked[kind] = int(getattr(result, "rowcount", 0) or 0)
    return blocked


def _finding_summary(report: dict) -> str:
    return ", ".join(
        f"{name}={len(ids)}" for name, ids in report["findings"].items() if ids
    ) or ("no manifest row" if report["no_manifest_row"] else "clean")


# ── argv ────────────────────────────────────────────────────────────────────


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="migrate_qdrant.py",
        description="Offline P1b migration: inventory, backup, keyset backfill, verify, cutover.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inv = sub.add_parser("inventory", help="read-only counts + quarantine lists")
    inv.add_argument("--out", type=Path, default=None, help="write the JSON report here")

    bkp = sub.add_parser("backup", help="VACUUM INTO snapshot + checksum manifest")
    bkp.add_argument("--dir", type=Path, required=True, help="destination directory")

    bf = sub.add_parser("backfill", help="build the contract generation (keyset)")
    bf.add_argument("--kind", choices=list(KINDS), required=True)
    bf.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    bf.add_argument("--resume", action="store_true", help="continue from the checkpoint")

    vf = sub.add_parser("verify", help="full read-side audit against the live collection")
    vf.add_argument("--kind", choices=list(KINDS), default=None,
                    help="the kind to audit (required unless --restore-drill)")
    vf.add_argument("--restore-drill", action="store_true",
                    help="restore the backup into a NEW directory and verify the restore")
    vf.add_argument("--dir", type=Path, default=None,
                    help="the backup directory a restore drill reads (never writes)")
    vf.add_argument("--target", type=Path, default=None,
                    help="where the drill restores to (default: <dir>.restore)")

    cut = sub.add_parser("cutover", help="flip the pointer for BOTH kinds (one transaction)")
    cut.add_argument("--yes", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    try:
        if args.command == "inventory":
            report = inventory(out=args.out)
        elif args.command == "backup":
            report = backup(dest_dir=args.dir)
        elif args.command == "backfill":
            report = backfill(kind=args.kind, batch=args.batch, resume=args.resume)
        elif args.command == "verify":
            if args.restore_drill:
                if args.dir is None:
                    raise MigrationRefused("verify --restore-drill needs --dir <backup directory>")
                report = restore_drill(backup_dir=args.dir, target=args.target)
            elif args.kind is None:
                raise MigrationRefused("verify needs --kind <kind>, or --restore-drill --dir <dir>")
            else:
                report = verify(kind=args.kind)
        else:
            report = cutover(yes=args.yes)
    except VerifyFailed as exc:
        print(f"verify failed: {exc}", file=sys.stderr)
        return 1
    except MigrationRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, default=str))
    if args.command == "verify" and not report["ok"]:
        print(
            "verify failed: " + _finding_summary(report),
            file=sys.stderr,
        )
        return 1
    if args.command == "backfill" and not report["complete"]:
        # The store did not ack a batch: the run stopped early and the operator
        # must not read "exit 0" as "the generation is built".
        print(
            "backfill incomplete: " + ("; ".join(report["errors"]) or "stopped early"),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
