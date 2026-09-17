"""Task 4 (P4b): every resurrection path reads the suppression ledger.

A source identity the user forgot must not come back through ANY of the four
write paths that could put the projection in front of them again:

- the import service (``run_import``) — an item the user already forgot;
- the document upload (``document_service.upload_document``) — the SAME BYTES
  uploaded again. Re-uploading mints a NEW document id, so ``source_ref`` alone
  cannot catch it: R38's key is the content hash, computed at upload time
  (sha256 of the bytes) and recorded on the projection when the document is
  ingested, so a forget can pin the BYTES and not just the doc id;
- the reindex/backfill helper (``reindex_user_memories_sync``) — never
  re-embed a suppressed source (admin reindex was issue #45);
- the outbox applier (``_apply_memory_intent``) — an intent that would write a
  SERVING payload for a suppressed source.

The negative pin is R37: soft forget's OWN payload-refresh intent (an
``invalidated`` row whose source is suppressed by construction) MUST still
land. A guard that blocks it would break the state contract it protects: the
point would keep serving ``visibility_state=current``.

Every assertion reads real tables through the real code paths (private SQLite
file per test, fixtures in this package's conftest). The vector store is
monkeypatched at its WRITE SEAM — these tests are about what reaches it, and
about the DB state left behind.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import uuid
from pathlib import Path

import pytest
from sqlalchemy import select

from app import database
from app.ingestion import pipeline
from app.ingestion.document_memory import suppress_source_async
from app.models.conversation import Conversation
from app.models.document import Document
from app.models.index_outbox import IndexOutbox
from app.models.memory import Memory, MemorySuppression
from app.models.user import User
from app.retrieval.memory import outbox, vector_store
from app.retrieval.memory import reindex as reindex_module
from app.retrieval.memory.correction import state_of
from app.retrieval.memory.outbox import (
    bump_revision,
    drain_pending,
    enqueue_upsert,
)
from app.services import import_service
from app.services.document_service import upload_document
from app.services.erasure_service import soft_forget
from app.utils.chunker import ParentChunk

REPO_ROOT = Path(__file__).resolve().parents[2]


async def _owner(db) -> uuid.UUID:
    uid = uuid.uuid4()
    db.add(User(id=uid, email=f"{uid.hex}@test.invalid", hashed_password="x",
                display_name="Owner", is_verified=True, is_active=True))
    await db.commit()
    return uid


def _memory(user_id, content="x", **kwargs) -> Memory:
    return Memory(id=uuid.uuid4(), user_id=user_id, content=content, tags=[], **kwargs)


async def _suppressions() -> list[MemorySuppression]:
    async with database.AsyncSessionLocal() as session:
        return list((await session.execute(select(MemorySuppression))).scalars().all())


async def _intents() -> list[IndexOutbox]:
    async with database.AsyncSessionLocal() as session:
        return list((await session.execute(
            select(IndexOutbox).order_by(IndexOutbox.seq))).scalars().all())


@pytest.fixture
def drain_db(sessions, monkeypatch):
    """``drain_pending``'s own sessionmaker.

    ``outbox`` imported ``AsyncSessionLocal`` BY VALUE, so patching the
    ``database`` module (what ``sessions`` does) is not enough for the drain's
    private session (pattern: tests/retrieval/test_index_outbox.py).
    """
    monkeypatch.setattr(outbox, "AsyncSessionLocal", sessions)
    return sessions


# ── (a) import: a forgotten source is skipped, counted, never re-created ─────

GENERIC_EXPORT = [
    {"title": "Kept", "content": "fresh content", "ref": "c2"},
    {"title": "Forgotten", "content": "content the user forgot", "ref": "c1"},
]


async def test_import_skips_a_suppressed_source_and_counts_it(db, monkeypatch):
    owner = await _owner(db)
    await suppress_source_async(db, user_id=owner, source_ref="c1", reason="forgotten",
                                namespace="personal")
    await db.commit()

    indexed: list[str] = []

    async def _index(memory):
        indexed.append(str(memory.id))
        return False  # the row is the thing under test, not the fast-path embed

    monkeypatch.setattr(import_service, "index_new_memory", _index)

    summary = await import_service.run_import(
        db, owner, json.dumps(GENERIC_EXPORT).encode("utf-8"), "generic",
        requested_by="rest_api",
    )

    assert summary.parsed == 2
    assert summary.created == 1
    assert summary.suppressed_skipped == 1, (
        "a forgotten ref is counted, not silently dropped into `failed` or `skipped_duplicates`")
    assert summary.skipped_duplicates == 0, "the dedup counters keep their meaning"
    assert summary.failed == 0

    refs = (await db.execute(
        select(Memory.source_ref).where(Memory.user_id == owner))).scalars().all()
    assert list(refs) == ["c2"], (
        "DB state, not just a counter: no row exists for the forgotten source")
    intents = await _intents()
    assert {row.entity_id for row in intents} == {uuid.UUID(indexed[0]).hex}, (
        "no durable upsert intent was left for the forgotten identity either")


# ── (b) re-upload: same bytes, new document id → still a reimport ──────────

FILE_BYTES = b"# Notes\n\nThe file the user forgot and then uploaded again.\n"
FILE_HASH = hashlib.sha256(FILE_BYTES).hexdigest()


class _Upload:
    """The ``UploadFile`` surface ``upload_document`` reads, nothing more."""

    filename = "notes.md"
    content_type = "text/markdown"

    def __init__(self, data: bytes):
        self._data = data

    async def read(self) -> bytes:
        return self._data


def _stub_upload_side_effects(monkeypatch) -> None:
    """MinIO/Redis/pipeline are out of process (or the next stage) — stubbed."""

    async def _none(*_a, **_k):
        return None

    monkeypatch.setattr("app.storage.put_object", _none)
    monkeypatch.setattr("app.retrieval.retrieval_cache.invalidate_query_cache", _none)
    monkeypatch.setattr("app.ingestion.pipeline.process_document_sync", lambda *_a, **_k: None)


async def test_reupload_of_a_forgotten_file_pins_the_new_id_and_reprojects_nothing(
    db, sync_db, monkeypatch
):
    owner = await _owner(db)
    conversation = Conversation(id=uuid.uuid4(), user_id=owner, document_count=0)
    db.add(conversation)
    await db.commit()

    # 1. The first upload of these bytes, ingested: the pipeline reads the
    #    stored object and records sha256(bytes) on the projection (R38).
    first_id = str(uuid.uuid4())
    db.add(Document(id=uuid.UUID(first_id), conversation_id=conversation.id,
                    filename="notes.md", file_path=f"{conversation.id}/{first_id}_notes.md",
                    file_size=len(FILE_BYTES), mime_type="text/markdown", status="ready"))
    await db.commit()
    monkeypatch.setattr("app.storage.get_object_sync", lambda *_a, **_k: FILE_BYTES)
    monkeypatch.setattr(vector_store, "upsert_memories_sync", lambda rows: 0)

    pipeline._project_document_to_memories(sync_db, first_id,
                                           [ParentChunk(id="p1", content="body", index=0)])
    projected = (sync_db.execute(
        select(Memory).where(Memory.source_ref == first_id))).scalars().all()
    assert {row.extra_metadata.get("content_hash") for row in projected} == {FILE_HASH}, (
        "the ingest seam records the hash of the uploaded bytes on every projection row")
    built = next(row for row in projected
                 if row.extra_metadata.get("kind") == "document")

    # 2. The user forgets it: the ledger must now know these BYTES, or the
    #    re-upload guard has nothing to match on.
    await soft_forget(db, owner, [built.id], requested_by="agent:test")
    ledger = await _suppressions()
    assert [row.content_hash for row in ledger] == [FILE_HASH], (
        "R38: the upload-time hash of the forgotten source reaches the ledger")

    # 3. The same file is uploaded again — a new document id, same bytes.
    _stub_upload_side_effects(monkeypatch)
    again = await upload_document(db, conversation, _Upload(FILE_BYTES))  # type: ignore[arg-type]

    assert again.id != uuid.UUID(first_id)
    pinned = [row for row in await _suppressions() if row.source_ref == str(again.id)]
    assert pinned and pinned[0].content_hash == FILE_HASH, (
        "the re-uploaded identity is pinned to the forgotten content (R38)")
    assert pinned[0].namespace == "personal"

    # 4. The projection for the new document creates nothing — the raw upload
    #    itself is kept (spec §12.3: a memory-targeted forget never deletes it).
    pipeline._project_document_to_memories(sync_db, str(again.id),
                                           [ParentChunk(id="p2", content="body", index=0)])
    refs = (sync_db.execute(
        select(Memory.source_ref).where(Memory.user_id == owner))).scalars().all()
    assert set(refs) == {first_id}, "no memory row exists for the re-uploaded document"


async def test_a_fresh_file_is_not_blocked_by_the_reupload_guard(db, monkeypatch):
    """Negative control: a file this user never forgot uploads normally."""
    owner = await _owner(db)
    conversation = Conversation(id=uuid.uuid4(), user_id=owner, document_count=0)
    db.add(conversation)
    await db.commit()

    _stub_upload_side_effects(monkeypatch)
    doc = await upload_document(db, conversation, _Upload(b"never seen before"))  # type: ignore[arg-type]

    assert [row for row in await _suppressions()] == [], (
        "an ordinary upload writes no ledger row")
    assert doc.status == "pending"


# ── (c) reindex: a suppressed source is never re-embedded ───────────────────


async def test_reindex_skips_a_suppressed_source_and_counts_it(db, sync_db, monkeypatch):
    owner = await _owner(db)
    forgotten = _memory(owner, "forgotten", source_ref="doc-gone")
    fresh = _memory(owner, "fresh", source_ref="doc-fresh")
    db.add_all([forgotten, fresh])
    await db.commit()
    await suppress_source_async(db, user_id=owner, source_ref="doc-gone", reason="forgotten")
    await db.commit()

    indexed: list[str] = []

    def _capture(rows):
        indexed.extend(str(m.id) for m in rows)
        return len(rows)

    monkeypatch.setattr(vector_store, "upsert_memories_sync", _capture)

    summary = reindex_module.reindex_user_memories_sync(str(owner), only_missing=False)

    assert indexed == [str(fresh.id)], (
        "a suppressed source is not re-embedded into the index (issue #45)")
    assert summary["reindexed"] == 1
    assert summary["suppressed_skipped"] == 1
    assert summary["scanned"] == 1, "suppressed rows are filtered before paging, not mid-page"


# ── (d) outbox: never write a SERVING payload for a suppressed source ───────


async def test_outbox_refuses_a_serving_upsert_for_a_suppressed_source(
    db, drain_db, monkeypatch
):
    owner = await _owner(db)
    row = _memory(owner, "forgotten fact", source_ref="doc-1")
    bump_revision(row)
    db.add(row)
    await db.commit()
    await suppress_source_async(db, user_id=owner, source_ref="doc-1", reason="forgotten")
    await enqueue_upsert(db, row)  # same revision: not stale, so only the guard can stop it
    await db.commit()

    written: list[str] = []
    deleted: list[str] = []

    async def _write(memory):
        written.append(str(memory.id))

    async def _delete(entity_id):
        deleted.append(str(entity_id))
        return True

    monkeypatch.setattr(outbox, "upsert_memory", _write)
    monkeypatch.setattr(outbox, "delete_memory", _delete)

    report = await drain_pending(batch_size=10)

    assert written == [], (
        "a serving payload for a suppressed source must not (re)enter the index")
    assert deleted == []
    assert report == {"claimed": 1, "applied": 0, "skipped": 1, "blocked": 0, "failed": 0}
    intents = await _intents()
    assert intents[0].status == "done", "the intent is settled, not retried forever"


# ── (e) NEGATIVE (R37): soft forget's own refresh intent still applies ──────


async def test_the_soft_forget_payload_refresh_intent_still_lands(db, drain_db, monkeypatch):
    owner = await _owner(db)
    row = _memory(owner, "the fact", source_ref="doc-9")
    db.add(row)
    await db.commit()

    await soft_forget(db, owner, [row.id], requested_by="agent:test")
    assert [r.source_ref for r in await _suppressions()] == ["doc-9"], (
        "soft forget suppressed the source (R38) — which is exactly what the "
        "outbox guard must NOT mistake for a resurrection")

    written: list[Memory] = []

    async def _write(memory):
        written.append(memory)

    monkeypatch.setattr(outbox, "upsert_memory", _write)

    report = await drain_pending(batch_size=10)

    assert [str(m.id) for m in written] == [str(row.id)], (
        "R37: the invalidated payload refresh MUST land; blocking it leaves the "
        "point serving visibility_state=current")
    assert state_of(written[0]) == "invalidated", "the payload carries the invalidated state"
    assert report["applied"] == 1


# ── the migration backfill keeps excluding suppressed sources (R28, pinned) ──


def _load_migrate_cli():
    """The migration CLI is a script, not a package member (tests/migration)."""
    path = REPO_ROOT / "scripts" / "migrate_qdrant.py"
    spec = importlib.util.spec_from_file_location("migrate_qdrant_t4_pin", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


async def test_backfill_eligibility_still_excludes_a_suppressed_source(db, sync_db):
    owner = await _owner(db)
    forgotten = _memory(owner, "forgotten", source_ref="doc-gone")
    kept = _memory(owner, "kept", source_ref="doc-ok")
    db.add_all([forgotten, kept])
    await db.commit()
    await suppress_source_async(db, user_id=owner, source_ref="doc-gone", reason="forgotten")
    await db.commit()

    rows = _load_migrate_cli().memory_rows(sync_db)

    assert rows[str(forgotten.id)]["reason"] == "suppressed", (
        "R28: a forgotten source must not keep (or regain) a servable vector")
    assert rows[str(kept.id)]["reason"] is None
