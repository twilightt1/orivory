"""P1.1 unify: documents project into cross-conversation memories.

Real-SQLite proof (no fake sessions): every assertion runs against a real
temp database file created by the ``sync_db`` / ``db`` fixtures, so the
projection queries, the durable intents and the suppression ledger are
exercised on the same dialect the sync ingestion path uses.

Covers Task 3 of the durable-canonical plan:
  - projection queries are tenant-scoped (a foreign memory sharing a
    ``source_ref`` survives),
  - re-ingest enqueues one intent per removed/new row in the caller's txn,
  - a suppressed identity is never reprojected.
"""
from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from app.database import sync_session
from app.ingestion.document_memory import (
    DOC_MEMORY_SOURCE_TYPE,
    build_document_memories_sync,
    delete_document_memories_async,
    delete_document_memories_sync,
    is_suppressed,
    is_suppressed_async,
    suppress_source,
    suppress_source_async,
)
from app.models.conversation import Conversation
from app.models.document import Document
from app.models.index_outbox import IndexOutbox
from app.models.memory import Memory, MemorySuppression
from app.models.user import User
from app.utils.chunker import ParentChunk

pytestmark = pytest.mark.rag


# ── fixtures helpers ─────────────────────────────────────────────────────────


def _new_user(label: str = "Owner") -> User:
    uid = uuid.uuid4()
    return User(id=uid, email=f"{uid.hex}@test.invalid", hashed_password="x",
                display_name=label, is_verified=True, is_active=True)


def _user(session) -> uuid.UUID:
    user = _new_user()
    session.add(user)
    session.flush()
    return user.id


def _document(session, user_id, filename: str = "report.pdf") -> Document:
    conversation = Conversation(id=uuid.uuid4(), user_id=user_id, title="conv")
    doc = Document(id=uuid.uuid4(), conversation_id=conversation.id, filename=filename,
                   file_path=f"uploads/{filename}")
    session.add_all([conversation, doc])
    session.flush()
    return doc


def _parents(n: int) -> list[ParentChunk]:
    return [ParentChunk(id=str(uuid.uuid4()), content=f"Parent {i} content body.", index=i)
            for i in range(n)]


def _rows(session, document_id: str) -> list[Memory]:
    return list(
        session.execute(select(Memory).where(Memory.source_ref == document_id)).scalars().all()
    )


def _outbox(session) -> list[IndexOutbox]:
    return list(session.execute(select(IndexOutbox).order_by(IndexOutbox.seq)).scalars().all())


def _memory(user_id, source_ref: str | None, content: str = "x") -> Memory:
    return Memory(id=uuid.uuid4(), user_id=user_id, content=content, tags=[],
                  source_ref=source_ref)


# ── projection shape (real SQLite) ───────────────────────────────────────────


def test_builds_one_doc_plus_n_passages(sync_db):
    user_id = _user(sync_db)
    doc = _document(sync_db, user_id)

    result = build_document_memories_sync(sync_db, str(doc.id), _parents(3), user_id=user_id)
    sync_db.commit()

    assert result.doc_memory_id is not None
    assert len(result.passage_memory_ids) == 3
    assert len(result.all_ids) == 4

    rows = _rows(sync_db, str(doc.id))
    assert len(rows) == 4
    doc_rows = [m for m in rows if m.extra_metadata.get("kind") == "document"]
    passage_rows = [m for m in rows if m.extra_metadata.get("kind") == "passage"]
    assert len(doc_rows) == 1
    assert len(passage_rows) == 3


def test_passages_link_to_doc_memory_via_parent_id(sync_db):
    user_id = _user(sync_db)
    doc = _document(sync_db, user_id)

    result = build_document_memories_sync(sync_db, str(doc.id), _parents(2), user_id=user_id)
    sync_db.commit()

    rows = _rows(sync_db, str(doc.id))
    doc_mem = next(m for m in rows if m.extra_metadata.get("kind") == "document")
    passages = [m for m in rows if m.extra_metadata.get("kind") == "passage"]

    assert str(doc_mem.id) == result.doc_memory_id
    for p in passages:
        assert str(p.parent_id) == str(doc_mem.id)  # passage hangs off the doc memory


def test_all_memories_carry_document_linkage(sync_db):
    user_id = _user(sync_db)
    doc = _document(sync_db, user_id)

    build_document_memories_sync(sync_db, str(doc.id), _parents(2), user_id=user_id)
    sync_db.commit()

    for m in _rows(sync_db, str(doc.id)):
        assert m.source_ref == str(doc.id)           # linkage key for cleanup
        assert m.source_type == DOC_MEMORY_SOURCE_TYPE
        assert m.user_id == user_id                  # owner resolved from the conversation
        assert m.extra_metadata["document_id"] == str(doc.id)


def test_idempotent_reingest_deletes_prior_memories(sync_db):
    user_id = _user(sync_db)
    doc = _document(sync_db, user_id)

    first = build_document_memories_sync(sync_db, str(doc.id), _parents(2), user_id=user_id)
    sync_db.commit()
    prior_ids = set(first.all_ids)

    second = build_document_memories_sync(sync_db, str(doc.id), _parents(1), user_id=user_id)
    sync_db.commit()

    rows = _rows(sync_db, str(doc.id))
    assert len(rows) == 2  # 1 doc + 1 passage; the prior projection is gone
    assert {str(m.id) for m in rows} == set(second.all_ids)
    assert set(second.removed_memory_ids) == prior_ids
    assert set(second.stale_vector_ids) == prior_ids
    assert not (set(second.all_ids) & prior_ids)


def test_skips_when_document_missing(sync_db):
    user_id = _user(sync_db)
    sync_db.commit()

    result = build_document_memories_sync(sync_db, str(uuid.uuid4()), _parents(2), user_id=user_id)
    sync_db.commit()

    assert result.doc_memory_id is None
    assert result.passage_memory_ids == []
    assert sync_db.execute(select(Memory)).scalars().all() == []
    assert sync_db.execute(select(IndexOutbox)).scalars().all() == []


# ── tenant scoping: a foreign memory sharing a source_ref is never touched ───


def test_projection_delete_is_tenant_scoped(sync_db):
    owner = _user(sync_db)
    stranger = _user(sync_db)
    doc = _document(sync_db, owner)
    shared_ref = str(doc.id)

    mine = _memory(owner, shared_ref, content="mine")
    theirs = _memory(stranger, shared_ref, content="theirs")
    sync_db.add_all([mine, theirs])
    sync_db.commit()

    removed = delete_document_memories_sync(sync_db, shared_ref, user_id=owner)
    sync_db.commit()

    assert removed == [str(mine.id)]
    survivors = _rows(sync_db, shared_ref)
    assert [str(m.id) for m in survivors] == [str(theirs.id)]  # foreign row untouched


async def test_async_projection_delete_is_tenant_scoped(db):
    owner, stranger = _new_user("Owner"), _new_user("Stranger")
    db.add_all([owner, stranger])
    conversation = Conversation(id=uuid.uuid4(), user_id=owner.id, title="conv")
    doc = Document(id=uuid.uuid4(), conversation_id=conversation.id, filename="f.pdf",
                   file_path="uploads/f.pdf")
    db.add_all([conversation, doc])
    shared_ref = str(doc.id)
    mine = _memory(owner.id, shared_ref, content="mine")
    theirs = _memory(stranger.id, shared_ref, content="theirs")
    db.add_all([mine, theirs])
    await db.commit()

    removed = await delete_document_memories_async(db, shared_ref, user_id=owner.id)
    await db.commit()

    assert removed == [str(mine.id)]
    survivors = (
        await db.execute(select(Memory).where(Memory.source_ref == shared_ref))
    ).scalars().all()
    assert [str(m.id) for m in survivors] == [str(theirs.id)]


def test_reingest_does_not_touch_a_foreign_projection(sync_db):
    owner = _user(sync_db)
    stranger = _user(sync_db)
    doc = _document(sync_db, owner)
    foreign = _memory(stranger, str(doc.id), content="foreign")
    sync_db.add(foreign)
    sync_db.commit()

    result = build_document_memories_sync(sync_db, str(doc.id), _parents(1), user_id=owner)
    sync_db.commit()

    assert str(foreign.id) not in set(result.all_ids)
    survivors = _rows(sync_db, str(doc.id))
    assert str(foreign.id) in {str(m.id) for m in survivors}
    owner_rows = [m for m in survivors if m.user_id == owner]
    assert len(owner_rows) == 2  # this projection only replaced the owner's rows


# ── durable intents: one per removed / created row, same transaction ─────────


def test_reingest_enqueues_intent_for_every_changed_row(sync_db):
    owner = _user(sync_db)
    doc = _document(sync_db, owner)

    first = build_document_memories_sync(sync_db, str(doc.id), _parents(2), user_id=owner)
    sync_db.commit()
    first_intents = _outbox(sync_db)
    assert sorted((r.entity_id, r.revision, r.operation) for r in first_intents) == sorted(
        (uuid.UUID(m).hex, 1, "upsert") for m in first.all_ids
    )

    second = build_document_memories_sync(sync_db, str(doc.id), _parents(1), user_id=owner)
    sync_db.flush()
    # Not committed yet: a second connection cannot see the new intents.
    with sync_session() as peek:
        assert len(_outbox(peek)) == len(first_intents)
    sync_db.commit()

    all_intents = _outbox(sync_db)
    assert sorted((r.entity_id, r.revision, r.operation) for r in all_intents) == sorted(
        [(uuid.UUID(m).hex, 1, "upsert") for m in first.all_ids]
        + [(uuid.UUID(m).hex, 1, "delete") for m in second.removed_memory_ids]
        + [(uuid.UUID(m).hex, 1, "upsert") for m in second.all_ids]
    )
    assert {r.tenant_id for r in all_intents} == {owner.hex}


def test_build_rolls_back_with_the_caller_transaction(sync_db):
    owner = _user(sync_db)
    doc = _document(sync_db, owner)
    sync_db.commit()

    build_document_memories_sync(sync_db, str(doc.id), _parents(1), user_id=owner)
    sync_db.rollback()

    assert sync_db.execute(select(Memory)).scalars().all() == []
    assert sync_db.execute(select(IndexOutbox)).scalars().all() == []


# ── suppression ledger: a forgotten identity is never reprojected ────────────


def test_suppressed_source_is_not_reprojected(sync_db):
    owner = _user(sync_db)
    doc = _document(sync_db, owner)
    sync_db.commit()

    suppress_source(sync_db, user_id=owner, source_ref=str(doc.id), reason="forgotten")
    sync_db.commit()

    result = build_document_memories_sync(sync_db, str(doc.id), _parents(2), user_id=owner)
    sync_db.commit()

    assert result.doc_memory_id is None
    assert result.passage_memory_ids == []
    assert _rows(sync_db, str(doc.id)) == []                 # nothing recreated
    assert sync_db.execute(select(IndexOutbox)).scalars().all() == []  # no intent either


def test_suppress_source_is_idempotent_and_tenant_scoped(sync_db):
    owner = _user(sync_db)
    stranger = _user(sync_db)
    ref = "drive-file-123"
    sync_db.commit()

    suppress_source(sync_db, user_id=owner, source_ref=ref)
    suppress_source(sync_db, user_id=owner, source_ref=ref)  # replay: no second row
    sync_db.commit()

    rows = sync_db.execute(select(MemorySuppression)).scalars().all()
    assert len(rows) == 1
    assert rows[0].reason == "forgotten"
    assert is_suppressed(sync_db, user_id=owner, source_ref=ref) is True
    assert is_suppressed(sync_db, user_id=stranger, source_ref=ref) is False
    assert is_suppressed(sync_db, user_id=owner, source_ref="other") is False


async def test_suppress_and_check_work_on_an_async_session(db):
    owner = _new_user("Owner")
    db.add(owner)
    await db.commit()

    await suppress_source_async(db, user_id=owner.id, source_ref="ref-a")
    await db.commit()

    assert await is_suppressed_async(db, user_id=owner.id, source_ref="ref-a") is True
    assert await is_suppressed_async(db, user_id=owner.id, source_ref="ref-b") is False


# ── pipeline projection query is tenant-scoped too ───────────────────────────


def test_pipeline_projection_is_tenant_scoped(sync_db, monkeypatch):
    from app.ingestion.pipeline import _project_document_to_memories
    from app.retrieval.memory import vector_store

    upserted: list[Memory] = []
    deleted: list[str] = []
    monkeypatch.setattr(vector_store, "upsert_memories_sync", lambda rows: upserted.extend(rows))
    monkeypatch.setattr(vector_store, "delete_memories_sync", lambda ids: deleted.extend(ids))

    owner = _user(sync_db)
    stranger = _user(sync_db)
    doc = _document(sync_db, owner)
    foreign = _memory(stranger, str(doc.id), content="foreign")
    sync_db.add(foreign)
    sync_db.commit()

    _project_document_to_memories(sync_db, str(doc.id), _parents(2))

    assert deleted == []
    assert upserted, "the projection must be embedded"
    assert {m.user_id for m in upserted} == {owner}  # never the foreign row
    assert str(foreign.id) not in {str(m.id) for m in upserted}
    survivors = _rows(sync_db, str(doc.id))
    assert str(foreign.id) in {str(m.id) for m in survivors}
    # The projection's intents rode the projection's own commit.
    intents = _outbox(sync_db)
    assert {r.tenant_id for r in intents} == {owner.hex}
    assert {r.entity_id for r in intents} == {uuid.UUID(str(m.id)).hex for m in upserted}
