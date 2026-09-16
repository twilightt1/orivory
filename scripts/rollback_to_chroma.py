#!/usr/bin/env python
"""Isolated rollback: rebuild a pre-P1b Chroma store from the LIVE SQLite DB.

One release of escape hatch (spec §12): the P1b cutover is a pointer flip, so
getting back on the OLD stack needs a Chroma store the pre-P1b binary can read
— the migrated Qdrant generations are no substitute. This tool rebuilds that
store from the LIVE database (so post-cutover writes survive) and hands the
operator two artifacts:

    <chroma-path>/                     the rebuilt Chroma store
    <chroma-path>.parent/<db>.rollback.db   the matching SQLite copy

The copy is what the old binary opens: its ladder accepts ``user_version = 2``
and its readers resolve ``Orivory_memories`` + ``rag_conv_<conversation_id>``.

What the rebuild guarantees (rulings R30-R32):

- the eligible set is the LIVE reader's set, computed by the migration CLI's ONE
  eligibility definition (``migrate_qdrant.memory_rows`` / ``chunk_rows``) — a
  superseded, dirty, suppressed or unowned row is never resurrected, and the
  memory half is scoped to ONE namespace (P4a: ``personal``, the only one there
  is; ``--namespace`` overrides it);
- the vectors and the recorded contract are the LEGACY MEAN contract
  (``e5_local.arctic_embed_passages_mean``), never the CLS one the live Qdrant
  generation serves;
- ``index_generations`` ends with exactly ONE active row per kind naming the
  rebuilt store, and ``PRAGMA user_version`` is stamped back to 2;
- the fixture gates (tenant isolation, ID set, correction, forget, chunks) run
  BEFORE anything is reported ready, and a failing gate writes a
  ``NOT-READY.json`` marker into the store;
- the arctic-mean assumption is cross-checked against the INSTALL's own records
  (the expand record, and any row still naming the mean generation): a
  contradiction refuses, and an install that records nothing is reported as
  unverified instead of implied;
- neither the LIVE database nor the pre-P1b snapshot beside it is written — all
  work happens on the emitted copy.

Isolation (R30): ``chromadb`` is NOT a runtime dependency after P1b, so run this
from its own venv (and never install the pinned file into the runtime image):

    python -m venv .venv-rollback
    .venv-rollback/bin/pip install -r requirements.txt -r requirements-rollback.txt
    .venv-rollback/bin/python scripts/rollback_to_chroma.py \
        --db /data/orivory.db --chroma-path /data/chroma-rollback

Exit codes: 0 the store is rebuilt and every gate passed; 1 the rebuild ran but
a gate failed (the artifacts are on disk, the store is NOT ready); 2 refused
before doing anything (bad input, an existing target, the app still running).
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sqlite3
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import migrate_qdrant as migration  # noqa: E402 — the ONE eligibility definition
from sqlalchemy import create_engine, delete, event, insert, select, text, update  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402
from sqlalchemy.pool import NullPool  # noqa: E402

from app import database  # noqa: E402
from app.config import settings  # noqa: E402
from app.models.document_chunk import DocumentChunk  # noqa: E402
from app.models.index_outbox import IndexGeneration  # noqa: E402
from app.models.memory import Memory  # noqa: E402
from app.retrieval.e5_local import arctic_embed_passages_mean  # noqa: E402
from app.retrieval.embedding_fingerprint import (  # noqa: E402
    LEGACY_MEAN_FINGERPRINT,
    canonical_fingerprint,
    fingerprint_generation,
)
from app.retrieval.memory import outbox, vector_store  # noqa: E402
from app.retrieval.memory.namespaces import PERSONAL  # noqa: E402
from app.retrieval.memory.visibility import namespace_predicate  # noqa: E402

log = logging.getLogger("rollback_to_chroma")

MEAN = "legacy-mean"
FINGERPRINTS = {MEAN: LEGACY_MEAN_FINGERPRINT}
MEMORY_COLLECTION = vector_store.COLLECTION_NAME          # the old binary's name
CHUNK_COLLECTION_PREFIX = "rag_conv_"                      # per-conversation
NOT_READY_MARKER = "NOT-READY.json"
PRE_P1B_USER_VERSION = 2
CHROMA_SPACE = {"hnsw:space": "cosine"}
DEFAULT_BATCH = 64
# The backend name the old guard compares against its own ``active_backend_name()``
# for arctic (``{"arctic": "local-arctic", "e5": "local-e5"}``). Stamped from the
# CONTRACT being rebuilt, never from the ambient settings, which describe the
# post-P1b deployment.
MEAN_BACKEND = "local-arctic"

default_embed = arctic_embed_passages_mean


class RollbackRefused(RuntimeError):
    """The tool refused to run. Exit code 2."""


# ── where things live ───────────────────────────────────────────────────────


def sidecar(db_path: Path, name: str) -> Path:
    """``<db>.<name>`` — the migration CLI's sidecar spelling, reused."""
    return Path(f"{db_path}.{name}")


def mean_canonical() -> str:
    return canonical_fingerprint(LEGACY_MEAN_FINGERPRINT)


def mean_token() -> str:
    return fingerprint_generation(LEGACY_MEAN_FINGERPRINT)


def chunk_collection(conversation_id: str) -> str:
    return f"{CHUNK_COLLECTION_PREFIX}{conversation_id}"


def rollback_from(db_path: Path) -> dict:
    """What this rollback is rolling back FROM (marker first, R32 read-order).

    The rollback marker is written AFTER the cutover transaction commits, so it
    can be missing (crash window). The marker's ``active``/``cutover_at`` are
    preferred over ``previous`` alone: ``active`` is what the install serves
    now, and it is the thing being rolled back. Without a marker, the pointer
    recorded at expand time (``<db>.p1b-expand-record.json``) is the fallback.
    """
    marker = sidecar(db_path, migration.ROLLBACK_MARKER_NAME)
    if marker.is_file():
        try:
            data = json.loads(marker.read_text())
        except json.JSONDecodeError:
            log.warning("%s is not readable JSON — falling back to the expand record", marker)
        else:
            return {
                "source": "marker",
                "marker": str(marker),
                "active": data.get("active"),
                "previous": data.get("previous"),
                "cutover_at": data.get("cutover_at"),
            }
    record = sidecar(db_path, migration.EXPAND_RECORD_NAME)
    if record.is_file():
        try:
            data = json.loads(record.read_text())
        except json.JSONDecodeError:
            data = {}
        return {
            "source": "expand-record",
            "marker": str(record),
            "active": data.get("previous"),
            "recorded_at": data.get("recorded_at"),
        }
    return {"source": "none", "active": None,
            "hint": "no rollback marker and no expand record: the generation the "
                    "live rows serve was not recorded by the migration"}


def _mapping(value: object) -> dict:
    """A JSON field that may be anything: a dict or nothing usable."""
    return value if isinstance(value, dict) else {}


def mean_evidence(db_path: Path) -> dict:
    """What the INSTALL's own records say about the contract it served pre-P1b.

    Ambient settings describe the post-P1b deployment, so they cannot show that
    the store being rebuilt was the arctic mean one. Two records can, and both
    are read here: the pointer recorded at expand time
    (``<db>.p1b-expand-record.json``) and any ``index_generations`` row that
    still names the mean generation (the P1a transitional row), which carries
    the fingerprint token it was built at.

    ``mean`` is ``None`` when neither record names a contract: the assumption is
    then UNVERIFIED, and the report says so instead of implying a check that
    never ran. ``False`` is a contradiction the caller refuses on.
    """
    mean_names = {outbox.TARGET_GENERATION, outbox.CHUNK_TARGET_GENERATION}
    observed: dict[str, list[str]] = {}
    sources: list[str] = []
    record = sidecar(db_path, migration.EXPAND_RECORD_NAME)
    if record.is_file():
        sources.append(record.name)
        try:
            recorded = json.loads(record.read_text())
        except json.JSONDecodeError:
            recorded = {}
        for kind, name in sorted(_mapping(_mapping(recorded).get("previous")).items()):
            if name:
                observed.setdefault(str(kind), []).append(str(name))
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        if "index_generations" in migration._table_names(conn):
            rows = conn.execute(
                "SELECT kind, generation, fingerprint FROM index_generations "
                "WHERE generation IN (?, ?)",
                tuple(sorted(mean_names)),
            ).fetchall()
            if rows:
                sources.append("index_generations")
            for kind, generation, fingerprint in rows:
                observed.setdefault(str(kind), []).extend(
                    value for value in (str(generation), str(fingerprint or "")) if value
                )
    finally:
        conn.close()

    known = mean_names | {mean_token()}
    foreign = sorted({value for values in observed.values() for value in values
                      if value not in known})
    if not observed:
        note = ("no install record names a pre-P1b contract (no expand record, no row "
                "naming the mean generation): the arctic mean assumption is UNVERIFIED, "
                "not confirmed")
    elif foreign:
        note = (f"the install's records name a different contract: {', '.join(foreign)} "
                "— neither the mean generation names nor the mean token")
    else:
        note = "the install's records name the arctic mean generation and token"
    return {
        "mean": None if not observed else not foreign,
        "sources": sources,
        "observed": observed,
        "unexpected": foreign,
        "note": note,
    }


# ── reading: the live data, on a copy ───────────────────────────────────────


def _copy_database(db_path: Path, dest: Path) -> None:
    """A consistent copy of the LIVE database, committed WAL frames included.

    ``sqlite3.Connection.backup`` reads pages; it never journals, so the source
    stays byte-identical. A read-only open needs the ``-shm`` sidecar of a
    non-empty WAL, so a killed process can leave a file that only opens
    read-write — the fallback logs it (that path may checkpoint the WAL).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        source = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        source.execute("SELECT 1")
    except sqlite3.OperationalError as exc:
        log.warning("read-only open failed (%s); copying over a read-write handle", exc)
        source = sqlite3.connect(db_path)
    try:
        target = sqlite3.connect(dest)
        try:
            source.backup(target)
        finally:
            target.close()
    finally:
        source.close()


@contextmanager
def _session(db_path: Path) -> Iterator[Session]:
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}, poolclass=NullPool
    )
    event.listen(engine, "connect", database._configure_sqlite_connection)
    session = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _eligible_memories(db: Session, *, namespace: str = PERSONAL
                       ) -> tuple[list[tuple[Memory, dict]], dict[str, dict]]:
    """Live rows split into (served, excluded) by the CLI's own eligibility.

    ``namespace`` is the boundary this rebuild exports (P4a: ``personal``, the
    only one there is): the old binary has no notion of a namespace, and a
    copied row the LIVE read path would not serve must not be resurrected just
    because the artifact is a copy.
    """
    rows = migration.memory_rows(db, namespace=namespace)
    served: list[tuple[Memory, dict]] = []
    for memory in db.execute(select(Memory).where(namespace_predicate(namespace))).scalars():
        record = rows[str(memory.id)]
        if record["reason"] is None:
            served.append((memory, record))
    return served, rows


def _eligible_chunks(db: Session) -> dict[str, list[tuple[DocumentChunk, dict]]]:
    """Eligible chunks grouped by conversation — the old collection layout."""
    rows = migration.chunk_rows(db)
    grouped: dict[str, list[tuple[DocumentChunk, dict]]] = {}
    for chunk in db.execute(select(DocumentChunk)).scalars():
        record = rows[str(chunk.id)]
        if record["reason"] is None:
            grouped.setdefault(record["conversation_id"], []).append((chunk, record))
    return grouped


def counts_for(rows: dict[str, dict]) -> dict:
    """Eligible/excluded counts per reason, for the operator's report."""
    counts = {"total": len(rows), "eligible": 0,
              "excluded": {"superseded": 0, "dirty": 0, "suppressed": 0, "unowned": 0}}
    for record in rows.values():
        reason = record["reason"]
        if reason is None:
            counts["eligible"] += 1
        else:
            counts["excluded"][reason] = counts["excluded"].get(reason, 0) + 1
    return counts


# ── writing: the pre-P1b Chroma contract ────────────────────────────────────


def collection_metadata(dim: int) -> dict:
    """The collection-level contract stamp the old guard reads."""
    canonical = mean_canonical()
    return {
        **CHROMA_SPACE,
        "orivory_embed_backend": MEAN_BACKEND,
        "orivory_embed_dim": int(dim),
        "orivory_embed_fingerprint": canonical,
        "orivory_embed_generation": fingerprint_generation(canonical),
    }


def memory_metadata(memory: Memory, dim: int) -> dict:
    """The memory payload the PRE-P1b binary wrote and reads (its own spelling).

    Deliberately not ``vector_store._memory_to_metadata``: that one stamps the
    LIVE contract and the P1b additions. This is the old contract at the mean
    fingerprint, so the old guard's fingerprint check passes. An absent value
    is omitted, never null (Chroma metadata has no null).
    """
    canonical = mean_canonical()
    metadata = {
        "user_id": str(memory.user_id),
        "memory_id": str(memory.id),
        "source_type": memory.source_type,
        "salience": float(memory.salience),
        "pinned": bool(memory.pinned),
        "orivory_embed_backend": MEAN_BACKEND,
        "orivory_embed_dim": int(dim),
        "orivory_embed_fingerprint": canonical,
        "orivory_embed_generation": fingerprint_generation(canonical),
    }
    if memory.captured_at is not None:
        metadata["captured_at"] = memory.captured_at.isoformat()
    model_revision = (LEGACY_MEAN_FINGERPRINT.get("model_revision")
                      or LEGACY_MEAN_FINGERPRINT.get("revision"))
    metadata["orivory_embed_model_revision"] = model_revision or "unavailable"
    metadata["orivory_memory_revision"] = int(getattr(memory, "revision", 1) or 1)
    if memory.tags:
        metadata["tags"] = list(memory.tags)
    return metadata


def open_chroma(path: Path):
    """The Chroma client, with R30's isolated-venv story in the error."""
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover — the runtime has no chromadb
        raise RollbackRefused(
            "chromadb is not installed: run this tool from its own venv "
            "(python -m venv .venv-rollback; pip install -r requirements.txt "
            "-r requirements-rollback.txt) — never into the runtime image"
        ) from exc
    return chromadb.PersistentClient(path=str(path))


def _upsert(collection, ids: list[str], vectors: list[list[float]],
            documents: list[str], metadatas: list[dict]) -> None:
    collection.upsert(ids=ids, embeddings=vectors, documents=documents, metadatas=metadatas)


def rebuild_memory(client, rows: list[tuple[Memory, dict]], *, embed, batch: int) -> int:
    """Write the memory collection the old binary resolves by name."""
    dim = int(LEGACY_MEAN_FINGERPRINT["dim"])
    collection = client.get_or_create_collection(
        MEMORY_COLLECTION, metadata=collection_metadata(dim)
    )
    for start in range(0, len(rows), batch):
        page = rows[start : start + batch]
        documents = [vector_store._memory_to_document(memory) for memory, _ in page]
        vectors = embed(documents)
        _upsert(
            collection,
            ids=[str(memory.id) for memory, _ in page],
            vectors=vectors,
            documents=documents,
            metadatas=[memory_metadata(memory, len(vectors[index]))
                       for index, (memory, _) in enumerate(page)],
        )
    return int(collection.count())


def rebuild_chunks(client, grouped: dict[str, list[tuple[DocumentChunk, dict]]], *,
                   embed, batch: int) -> dict[str, int]:
    """Write one ``rag_conv_<id>`` collection per conversation (old layout)."""
    dim = int(LEGACY_MEAN_FINGERPRINT["dim"])
    written: dict[str, int] = {}
    for conversation_id, rows in sorted(grouped.items()):
        name = chunk_collection(conversation_id)
        collection = client.get_or_create_collection(name, metadata=collection_metadata(dim))
        for start in range(0, len(rows), batch):
            page = rows[start : start + batch]
            documents = [chunk.content for chunk, _ in page]
            vectors = embed(documents)
            _upsert(
                collection,
                ids=[str(chunk.id) for chunk, _ in page],
                vectors=vectors,
                documents=documents,
                metadatas=[dict(chunk.chunk_metadata or {}) for chunk, _ in page],
            )
        written[name] = int(collection.count())
    return written


def stamp_manifest(db: Session, targets: dict[str, str]) -> None:
    """Exactly ONE generation row per kind, naming the rebuilt store.

    Every other row for the two kinds is DELETED, not merely deactivated: the
    emitted database describes one store — the Chroma one — and a leftover row
    naming a Qdrant generation (at the CLS contract) would be a claim about
    data that is not in this store. Rolling forward again re-creates those rows
    (``database.activate_generations`` inserts what is missing).
    """
    token = mean_token()
    db.execute(
        delete(IndexGeneration).where(
            IndexGeneration.kind.in_(list(targets)),
            IndexGeneration.generation.notin_(list(targets.values())),
        )
    )
    for kind, generation in targets.items():
        row = db.execute(
            select(IndexGeneration).where(
                IndexGeneration.kind == kind, IndexGeneration.generation == generation
            )
        ).scalars().first()
        if row is None:
            db.execute(insert(IndexGeneration).values(
                id=uuid.uuid4().hex, kind=kind, generation=generation,
                fingerprint=token, is_active=True, created_at=datetime.now(UTC),
            ))
        else:
            db.execute(
                update(IndexGeneration)
                .where(IndexGeneration.kind == kind, IndexGeneration.generation == generation)
                .values(fingerprint=token, is_active=True)
            )


# ── gates: the fixtures, before anything is called ready (R32 f) ────────────


def _points(collection) -> list[tuple[str, dict]]:
    got = collection.get(include=["metadatas"])
    return [(str(point_id), dict(metadata or {}))
            for point_id, metadata in zip(got["ids"], got["metadatas"], strict=False)]


def _by_tenant(expected: dict[str, str]) -> dict[str, set[str]]:
    tenants: dict[str, set[str]] = {}
    for entity_id, tenant in expected.items():
        tenants.setdefault(str(tenant), set()).add(str(entity_id))
    return tenants


def _contract_findings(metadata: dict) -> list[str]:
    """Every way the OLD binary's own guard would reject this collection.

    ``check_collection_dim`` — the p1a code the pre-P1b binary runs before every
    query — requires the collection's recorded backend to equal its
    ``active_backend_name()``, its dim to equal the embedding dim, its
    fingerprint to equal the contract it embeds with, and its generation to be
    that fingerprint's own token; the space must be the cosine the vectors were
    compared under. A store that fails any of them is not ready, however clean
    its points are.
    """
    problems: list[str] = []
    canonical = str(metadata.get("orivory_embed_fingerprint") or "")
    if canonical != mean_canonical():
        problems.append("fingerprint")
    try:
        generation: str | None = fingerprint_generation(canonical)
    except ValueError:
        generation = None
    if not generation or metadata.get("orivory_embed_generation") != generation:
        problems.append("generation")
    if metadata.get("orivory_embed_backend") != MEAN_BACKEND:
        problems.append("backend")
    try:
        dim: int | None = int(str(metadata.get("orivory_embed_dim")))
    except (TypeError, ValueError):
        dim = None
    if dim != int(LEGACY_MEAN_FINGERPRINT["dim"]):
        problems.append("dim")
    if metadata.get("hnsw:space") != CHROMA_SPACE["hnsw:space"]:
        problems.append("space")
    return problems


def gate_findings(*, client, memory_collection: str, chunk_collections: dict, expected: dict,
                  excluded: dict, chunk_expected: dict, token: str) -> dict[str, list[str]]:
    """Read the rebuilt store back and report every fixture violation.

    Findings are grouped by the fixture they belong to: ``id_set`` (the served
    set must equal the eligible set, per tenant), ``fingerprint`` (the legacy
    mean contract: the collection stamp the old guard reads AND the per-point
    generation), ``tenant_isolation`` (no query may cross a tenant),
    ``correction`` (a superseded/dirty row is gone), ``forget`` (a suppressed
    row is gone and stays gone) and ``chunks``. An empty dict means ready.
    """
    findings: dict[str, list[str]] = {}

    def add(bucket: str, entity_id: str) -> None:
        findings.setdefault(bucket, []).append(str(entity_id))

    memory = client.get_collection(memory_collection)
    for problem in _contract_findings(dict(memory.metadata or {})):
        add("fingerprint", f"collection:{problem}")
    served: dict[str, set[str]] = {}
    served_ids: set[str] = set()
    absent_by_reason = {"superseded": "correction", "dirty": "correction", "suppressed": "forget"}
    for point_id, payload in _points(memory):
        tenant = str(payload.get("user_id") or "unknown")
        served.setdefault(tenant, set()).add(point_id)
        served_ids.add(point_id)
        want = expected.get(point_id)
        if want is None:
            add(absent_by_reason.get(str(excluded.get(point_id)), "id_set"), point_id)
            continue
        if tenant != str(want):
            add("tenant_isolation", point_id)
        if payload.get("orivory_embed_generation") != token:
            add("fingerprint", point_id)
    for entity_id, reason in excluded.items():
        if entity_id not in served_ids:
            continue
        add(absent_by_reason.get(str(reason), "id_set"), entity_id)
    for tenant, want_ids in _by_tenant(expected).items():
        have = served.get(tenant, set())
        for missing in sorted(want_ids - have):
            add("id_set", missing)
        for extra in sorted(have - want_ids):
            add("id_set", extra)
        # The store's own filter is the security boundary: a query scoped to one
        # tenant must never return another tenant's point.
        if not want_ids or not have:
            continue
        probe = memory.get(ids=[sorted(want_ids)[0]], include=["embeddings"])["embeddings"][0]
        hits = memory.query(query_embeddings=[probe], n_results=memory.count(),
                            where={"user_id": {"$eq": tenant}})["ids"][0]
        for hit in hits:
            if hit not in want_ids:
                add("tenant_isolation", hit)

    for name, want_ids in chunk_expected.items():
        collection = chunk_collections[name]
        for problem in _contract_findings(dict(collection.metadata or {})):
            add("fingerprint", f"{name}:{problem}")
        have = {point_id for point_id, _payload in _points(collection)}
        for missing in sorted(set(want_ids) - have):
            add("chunks", missing)
        for extra in sorted(have - set(want_ids)):
            add("chunks", extra)

    return {bucket: sorted(set(ids)) for bucket, ids in findings.items() if ids}


# ── the rollback itself ─────────────────────────────────────────────────────


def rollback(*, db_path: Path, chroma_path: Path, fingerprint: str = MEAN,
             out_db: Path | None = None, embed_passages=None, batch: int = DEFAULT_BATCH,
             namespace: str = PERSONAL) -> dict:
    """Rebuild the pre-P1b Chroma store + the SQLite copy the old binary opens.

    ``namespace`` scopes the memory export (P4a: ``personal``, the only one a
    P4a deployment holds) — passed to the eligibility definition, so the rebuilt
    store carries exactly the rows the live reader would serve.
    """
    started = time.monotonic()
    if not migration.is_sqlite():
        raise RollbackRefused(
            "rollback_to_chroma rebuilds a SQLite install's Chroma store; a server "
            "deployment rolls back with its Qdrant snapshot and its own pre-P1b path"
        )
    if fingerprint not in FINGERPRINTS:
        raise RollbackRefused(
            f"unknown fingerprint {fingerprint!r}: this tool rebuilds the "
            f"{MEAN!r} contract only (the one the pre-P1b binary served)"
        )
    if settings.USE_LOCAL_EMBEDDINGS and settings.LOCAL_EMBED_MODEL != "arctic":
        raise RollbackRefused(
            f"this install embeds with local {settings.LOCAL_EMBED_MODEL!r}; the only "
            f"contract that can be rebuilt here is the arctic {MEAN!r} one — an install "
            "that served a different local model needs its own rollback path"
        )
    db_path = Path(db_path)
    if not db_path.is_file():
        raise RollbackRefused(f"no such database: {db_path}")
    evidence = mean_evidence(db_path)
    if evidence["mean"] is False:
        raise RollbackRefused(
            f"refusing to rebuild: {evidence['note']}. This store would carry vectors "
            "for a contract the install never served"
        )
    chroma_path = Path(chroma_path)
    if chroma_path.exists() and not chroma_path.is_dir():
        raise RollbackRefused(
            f"{chroma_path} is a file — --chroma-path names the DIRECTORY the rebuilt "
            "store is written into"
        )
    if chroma_path.exists() and any(chroma_path.iterdir()):
        raise RollbackRefused(
            f"{chroma_path} already holds a store — a rollback builds a FRESH "
            "directory (never merges into an existing one); pass another "
            "--chroma-path or remove it. A re-run generally needs both: another "
            "--chroma-path AND --out-db for the emitted copy"
        )
    emitted = Path(out_db) if out_db else chroma_path.parent / f"{db_path.name}.rollback.db"
    if emitted.exists():
        raise RollbackRefused(
            f"{emitted} already exists — a rebuild never overwrites an emitted copy; "
            "remove it, or pass --out-db <path> (a second rollback also needs its own "
            "--chroma-path: the store beside it is taken)"
        )
    if emitted.resolve() == db_path.resolve():
        raise RollbackRefused("--out-db must not be the live database")
    migration.require_quiesced()
    embed = embed_passages or default_embed
    targets = {"memory": MEMORY_COLLECTION, "chunk": outbox.CHUNK_TARGET_GENERATION}

    try:
        _copy_database(db_path, emitted)
        client = open_chroma(chroma_path)
        with _session(emitted) as db:
            served_memories, memory_rows = _eligible_memories(db, namespace=namespace)
            grouped = _eligible_chunks(db)
            chunk_rows = migration.chunk_rows(db)
            memory_points = rebuild_memory(client, served_memories, embed=embed, batch=batch)
            chunk_points = rebuild_chunks(client, grouped, embed=embed, batch=batch)
            stamp_manifest(db, targets)
            db.execute(text(f"PRAGMA user_version = {PRE_P1B_USER_VERSION}"))
            db.commit()
    except BaseException:
        # Never leave a half-state behind: the artifacts are ours, not the
        # operator's — the source database is untouched either way.
        shutil.rmtree(chroma_path, ignore_errors=True)
        emitted.unlink(missing_ok=True)
        raise

    chunk_expected = {
        name: {str(chunk.id): str(record["tenant"]) for chunk, record in rows}
        for name, rows in ((chunk_collection(cid), rows) for cid, rows in grouped.items())
    }
    excluded = {
        entity_id: str(record["reason"])
        for entity_id, record in {**memory_rows, **chunk_rows}.items()
        if record["reason"] is not None
    }
    gates = gate_findings(
        client=client,
        memory_collection=MEMORY_COLLECTION,
        chunk_collections={name: client.get_collection(name) for name in chunk_expected},
        expected={str(memory.id): str(record["tenant"]) for memory, record in served_memories},
        excluded=excluded,
        chunk_expected=chunk_expected,
        token=mean_token(),
    )
    report = {
        "ready": not gates,
        "ok": not gates,
        "fingerprint": {"name": fingerprint, "canonical": mean_canonical(),
                        "token": mean_token(), "dim": int(LEGACY_MEAN_FINGERPRINT["dim"])},
        "contract_evidence": evidence,
        "db": {"source": str(db_path), "emitted": str(emitted),
               "user_version": PRE_P1B_USER_VERSION},
        "chroma": {"path": str(chroma_path),
                   "collections": {**{MEMORY_COLLECTION: {"points": memory_points}},
                                   **{name: {"points": points}
                                      for name, points in chunk_points.items()}}},
        "manifest": targets,
        "counts": {"memory": counts_for(memory_rows), "chunk": counts_for(chunk_rows)},
        "rollback_from": rollback_from(db_path),
        "gates": gates,
        "findings": gates,
        "elapsed_s": round(time.monotonic() - started, 3),
    }
    if gates:
        (chroma_path / NOT_READY_MARKER).write_text(json.dumps(
            {"ready": False, "built_at": datetime.now(UTC).isoformat(), "findings": gates,
             "message": "a fixture gate failed: do NOT point the pre-P1b binary at this "
                        "store; re-run the rollback with a fresh --chroma-path (and "
                        "--out-db, if the emitted copy is taken) once the finding is "
                        "understood — this store is never overwritten or merged into"},
            indent=2,
        ))
        log.error("rollback gates failed: %s", ", ".join(f"{k}={len(v)}" for k, v in gates.items()))
    return report


# ── argv ────────────────────────────────────────────────────────────────────


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rollback_to_chroma.py",
        description="Rebuild a pre-P1b Chroma store from the live SQLite database.",
    )
    parser.add_argument("--db", type=Path, required=True, help="the LIVE SQLite database")
    parser.add_argument("--chroma-path", type=Path, required=True,
                        help="directory to build the Chroma store in (must be free)")
    parser.add_argument("--fingerprint", default=MEAN, choices=sorted(FINGERPRINTS),
                        help="embedding contract to rebuild (only the legacy mean exists)")
    parser.add_argument("--out-db", type=Path, default=None,
                        help="where to emit the SQLite copy (default: beside the store)")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--namespace", default=PERSONAL,
                        help="memory namespace to export (P4a: 'personal', the "
                             "only one there is)")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    try:
        report = rollback(db_path=args.db, chroma_path=args.chroma_path,
                          fingerprint=args.fingerprint, out_db=args.out_db,
                          batch=args.batch, namespace=args.namespace)
    except RollbackRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, default=str))
    if not report["ok"]:
        print("rollback is NOT ready: " + "; ".join(
            f"{bucket}={len(ids)}" for bucket, ids in report["findings"].items()), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
