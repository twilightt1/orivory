"""Durable deletion (P1a Task 4): one closure transaction, delete intents,
honest receipts — real SQLite files (fixtures in this package's conftest).

Chroma is the only monkeypatched seam: P1a keeps Chroma as the vector backend
and no live Chroma runs here, but every DB assertion below runs against a real
temp SQLite file with the real service, sessions and outbox.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select, text

from app import database
from app.ingestion import document_memory
from app.models.conversation import Conversation
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.entity import Entity, MemoryEntity, Relation
from app.models.erasure_receipt import ErasureReceipt
from app.models.index_outbox import IndexOutbox
from app.models.memory import Memory, MemorySuppression
from app.models.memory_access_log import MemoryAccessLog
from app.models.user import User
from app.retrieval.memory import outbox, vector_store
from app.retrieval.memory.correction import DerivedClosureError
from app.services import erasure_service
from app.services.erasure_service import erase_memories
from app.utils.chunker import ParentChunk


def _memory(user_id, content="x", **kwargs) -> Memory:
    return Memory(id=uuid.uuid4(), user_id=user_id, content=content, tags=[], **kwargs)


def _chunk(document: Document, *, index: int = 0) -> DocumentChunk:
    """A child chunk as the pipeline writes it (id, revision, metadata)."""
    return DocumentChunk(
        id=uuid.uuid4(), document_id=document.id, content=f"chunk {index}",
        chunk_index=index, revision=1,
        chunk_metadata={
            "document_id": str(document.id),
            "conversation_id": str(document.conversation_id),
            "chunk_type": "child",
            "child_index": index,
        },
    )


async def _owner(db) -> uuid.UUID:
    """A user row — ``memories.user_id`` is a FK."""
    uid = uuid.uuid4()
    db.add(User(id=uid, email=f"{uid.hex}@test.invalid", hashed_password="x",
                display_name="Owner", is_verified=True, is_active=True))
    await db.commit()
    return uid


async def _outbox_rows() -> list[IndexOutbox]:
    async with database.AsyncSessionLocal() as session:
        return list((await session.execute(
            select(IndexOutbox).order_by(IndexOutbox.seq))).scalars().all())


async def _receipts() -> list[ErasureReceipt]:
    async with database.AsyncSessionLocal() as session:
        return list((await session.execute(select(ErasureReceipt))).scalars().all())


async def _memory_ids() -> set[uuid.UUID]:
    async with database.AsyncSessionLocal() as session:
        return set((await session.execute(select(Memory.id))).scalars().all())


@pytest.fixture()
def no_chroma(monkeypatch):
    """Chroma seams: the purge lands and absence is confirmed."""
    async def _purge(_memory_id):
        return True

    async def _absent(_memory_ids):
        return set()

    monkeypatch.setattr(erasure_service, "safe_delete_from_index", _purge)
    monkeypatch.setattr(erasure_service, "_vector_present_ids", _absent)


@pytest.fixture()
def chroma_down(monkeypatch):
    """Chroma seams: the purge fails and the presence check is unreachable."""
    async def _purge_fails(_memory_id):
        return False

    async def _unreachable(_memory_ids):
        raise ConnectionError("chroma down")

    monkeypatch.setattr(erasure_service, "safe_delete_from_index", _purge_fails)
    monkeypatch.setattr(erasure_service, "_vector_present_ids", _unreachable)


# ── one closure transaction, delete intent per affected id ───────────────────


async def test_erase_is_one_closure_transaction_with_delete_intents(db, no_chroma):
    uid = await _owner(db)
    root = _memory(uid, "root")
    child = _memory(uid, "child", parent_id=root.id)
    db.add_all([root, child])
    await db.commit()

    snapshots: list[tuple[int, int]] = []
    real_commit = db.commit

    async def _commit_spy():
        # Same transaction, just before the commit: rows gone AND intents in.
        await db.flush()
        rows = (await db.execute(select(func.count()).select_from(Memory))).scalar_one()
        intents = (await db.execute(select(func.count()).select_from(IndexOutbox))).scalar_one()
        snapshots.append((int(rows), int(intents)))
        await real_commit()

    db.commit = _commit_spy
    receipt = await erase_memories(db, uid, [root.id], requested_by="rest_api")

    # The row deletes and their intents rode ONE commit (then the receipt commit).
    assert snapshots[0] == (0, 2), "rows + intents must commit together"
    assert snapshots[1] == (0, 2)

    target = receipt.detail["targets"][0]
    assert target["status"] == "deleted"
    assert target["affected_memory_ids"] == [str(child.id)]
    assert target["vector_state"] == "verified"

    intents = await _outbox_rows()
    assert {row.entity_id for row in intents} == {root.id.hex, child.id.hex}
    assert {row.operation for row in intents} == {"delete"}
    assert {row.tenant_id for row in intents} == {uid.hex}
    assert {row.status for row in intents} == {"pending"}
    assert receipt.detail["index_pending"] == 2  # the drain still owes both deletes
    assert receipt.detail["verification"] == "verified"


async def test_erase_closure_has_no_silent_depth_cap(db, no_chroma):
    uid = await _owner(db)
    root = _memory(uid, "root")
    chain = [_memory(uid, f"c{i}") for i in range(7)]
    parent = root
    for node in chain:
        node.parent_id = parent.id
        parent = node
    db.add(root)
    db.add_all(chain)
    await db.commit()

    receipt = await erase_memories(db, uid, [root.id], requested_by="rest_api")

    target = receipt.detail["targets"][0]
    assert target["status"] == "deleted"
    assert target["traversal_depth"] == 7
    assert target["affected_memory_ids"] == [str(node.id) for node in chain]
    assert target["truncated"] is False
    assert await _memory_ids() == set()
    assert {row.entity_id for row in await _outbox_rows()} == {
        root.id.hex, *(node.id.hex for node in chain)}


async def test_closure_cycle_terminates(db, no_chroma):
    uid = await _owner(db)
    a = _memory(uid, "a")
    b = _memory(uid, "b", parent_id=a.id)
    db.add_all([a, b])
    await db.flush()
    a.parent_id = b.id  # a ↔ b: the traversal must not spin
    await db.commit()

    receipt = await erase_memories(db, uid, [a.id], requested_by="rest_api")

    target = receipt.detail["targets"][0]
    assert target["status"] == "deleted"
    assert str(b.id) in target["affected_memory_ids"]
    assert await _memory_ids() == set()


async def test_closure_over_safety_bound_refuses_a_partial_erase(db, no_chroma, monkeypatch):
    monkeypatch.setattr(erasure_service, "_MAX_CLOSURE_IDS", 3)
    uid = await _owner(db)
    root = _memory(uid, "root")
    chain = [_memory(uid, f"c{i}") for i in range(4)]
    parent = root
    for node in chain:
        node.parent_id = parent.id
        parent = node
    db.add(root)
    db.add_all(chain)
    await db.commit()

    receipt = await erase_memories(db, uid, [root.id], requested_by="rest_api")

    target = receipt.detail["targets"][0]
    assert target["status"] == "error"
    assert target["truncated"] is True
    assert receipt.status == "completed_with_errors"  # never a clean completion
    # Nothing was erased and nothing was enqueued: a partial closure is refused.
    assert len(await _memory_ids()) == 5
    assert await _outbox_rows() == []


# ── vector verification states ───────────────────────────────────────────────


async def test_chroma_outage_reports_pending_vectors(db, chroma_down):
    uid = await _owner(db)
    mem = _memory(uid)
    db.add(mem)
    await db.commit()

    receipt = await erase_memories(db, uid, [mem.id], requested_by="rest_api")

    target = receipt.detail["targets"][0]
    assert target["status"] == "deleted"
    assert target["vector_state"] == "pending"  # the durable delete intent owns it now
    assert target["vector_residual_checked"] is False
    assert receipt.status == "completed_unverified"  # spec §5.4 / P1 gate: no positive readback
    assert receipt.detail["verification"] == "pending"
    assert receipt.detail["index_pending"] == 1


async def test_verify_unknown_is_not_reported_as_verified(db, monkeypatch):
    async def _purge_ok(_memory_id):
        return True

    async def _unreachable(_memory_ids):
        raise ConnectionError("present-check down")

    monkeypatch.setattr(erasure_service, "safe_delete_from_index", _purge_ok)
    monkeypatch.setattr(erasure_service, "_vector_present_ids", _unreachable)

    uid = await _owner(db)
    mem = _memory(uid)
    db.add(mem)
    await db.commit()

    receipt = await erase_memories(db, uid, [mem.id], requested_by="rest_api")

    target = receipt.detail["targets"][0]
    assert target["vector_state"] == "unknown"
    assert target["vector_state"] != "verified"
    assert receipt.status == "completed_unverified"  # spec §5.4 / P1 gate
    assert receipt.detail["verification"] == "unknown"


async def test_derived_closure_failure_records_unknown(db, no_chroma, monkeypatch):
    uid = await _owner(db)
    mem = _memory(uid)
    db.add(mem)
    await db.commit()

    async def _boom(*_args, **_kwargs):
        raise DerivedClosureError("derived-memory closure query failed")

    monkeypatch.setattr(erasure_service, "collect_derived_ids", _boom)
    receipt = await erase_memories(db, uid, [mem.id], requested_by="rest_api")

    target = receipt.detail["targets"][0]
    assert target["status"] == "deleted"
    assert target["derived_closure"] == "unknown"
    assert target["vector_state"] == "verified"
    assert receipt.status == "completed_with_residual"  # never claims a complete closure


async def test_collect_derived_ids_raises_typed_error(db):
    from app.retrieval.memory.correction import collect_derived_ids

    # A real DB failure (table gone), not a stub: the closure must not read as [].
    await db.execute(text("ALTER TABLE memories RENAME TO memories_hidden"))
    await db.commit()

    with pytest.raises(DerivedClosureError):
        await collect_derived_ids(db, uuid.uuid4(), [uuid.uuid4()])


async def test_vector_store_delete_reports_failure(monkeypatch):
    class _Client:
        async def delete(self, **_kwargs):
            return None

    async def _up(_dim):
        return _Client(), "generation", None

    monkeypatch.setattr(vector_store, "_open_collection", _up)
    assert await vector_store.delete_memory("mem-1") is True
    assert await vector_store.delete_memories(["mem-1", "mem-2"]) is True

    async def _down(_dim):
        raise ConnectionError("qdrant down")

    monkeypatch.setattr(vector_store, "_open_collection", _down)
    assert await vector_store.delete_memory("mem-1") is False
    assert await vector_store.delete_memories(["mem-1"]) is False


async def test_failed_vector_delete_is_transient_not_done(db, sessions, monkeypatch, no_chroma):
    uid = await _owner(db)
    mem = _memory(uid)
    db.add(mem)
    await db.commit()
    await outbox.enqueue_delete(db, entity_id=str(mem.id), tenant_id=str(uid), revision=1)
    await db.delete(mem)
    await db.commit()

    monkeypatch.setattr(outbox, "AsyncSessionLocal", sessions)

    async def _delete_fails(_memory_id):
        return False

    monkeypatch.setattr(outbox, "delete_memory", _delete_fails)
    report = await outbox.drain_pending()

    assert report == {"claimed": 1, "applied": 0, "skipped": 0, "blocked": 0, "failed": 1}
    row = (await _outbox_rows())[0]
    assert row.status == "pending" and row.attempts == 1
    assert row.next_attempt_at is not None

    async def _delete_ok(_memory_id):
        return True

    monkeypatch.setattr(outbox, "delete_memory", _delete_ok)
    async with database.AsyncSessionLocal() as session:
        due = await session.get(IndexOutbox, row.seq)
        due.next_attempt_at = datetime.now(UTC) - timedelta(seconds=1)
        await session.commit()

    assert (await outbox.drain_pending())["applied"] == 1
    assert (await _outbox_rows())[0].status == "done"


# ── suppression on forget ────────────────────────────────────────────────────


async def test_forget_projection_suppresses_reprojection(db, sync_db, no_chroma):
    uid = await _owner(db)
    conversation = Conversation(id=uuid.uuid4(), user_id=uid, title="conv")
    doc = Document(id=uuid.uuid4(), conversation_id=conversation.id, filename="notes.md",
                   file_path="uploads/notes.md")
    db.add_all([conversation, doc])
    await db.flush()
    projected = _memory(uid, "projected", source_type="file_upload", source_ref=str(doc.id))
    passage = _memory(uid, "passage", source_type="file_upload", source_ref=str(doc.id),
                      parent_id=projected.id)
    db.add_all([projected, passage])
    await db.commit()

    receipt = await erase_memories(db, uid, [projected.id], requested_by="rest_api")

    target = receipt.detail["targets"][0]
    assert target["status"] == "deleted"
    assert target["suppressed_source"] == str(doc.id)
    async with database.AsyncSessionLocal() as session:
        rows = (await session.execute(select(MemorySuppression))).scalars().all()
    assert [(row.user_id, row.source_ref, row.reason) for row in rows] == [
        (uid, str(doc.id), "forgotten")]
    # The raw upload is untouched (no raw-upload deletion).
    async with database.AsyncSessionLocal() as session:
        assert await session.get(Document, doc.id) is not None

    # A re-ingest of the same source does not resurrect the forgotten projection.
    result = document_memory.build_document_memories_sync(
        sync_db, str(doc.id),
        [ParentChunk(id=str(uuid.uuid4()), content="body", index=0)], user_id=uid)
    sync_db.commit()
    assert result.doc_memory_id is None and result.passage_memory_ids == []
    assert await _memory_ids() == set()


# ── REST + MCP delete collapse into the erasure path ─────────────────────────


async def test_rest_and_mcp_delete_go_through_erasure(db, no_chroma, monkeypatch):
    from app.api.v1 import memories as memories_api
    from app.mcp_hub import tools as hub_tools
    from app.mcp_hub.identity import AgentPrincipal

    uid = await _owner(db)
    rest_mem = _memory(uid, "via rest")
    mcp_mem = _memory(uid, "via mcp")
    db.add_all([rest_mem, mcp_mem])
    await db.commit()

    await memories_api.delete_memory(rest_mem.id, SimpleNamespace(id=uid), db)

    principal = AgentPrincipal(user_id=uid, agent_client_id=uuid.uuid4(), name="TestAgent",
                               scopes=frozenset({"memory:read", "memory:write"}))
    monkeypatch.setattr(hub_tools, "_current_principal", lambda: principal)
    monkeypatch.setattr(hub_tools, "_session", lambda: database.AsyncSessionLocal())
    out = await hub_tools.delete_memory(memory_id=str(mcp_mem.id))

    assert out["deleted"] is True and out["id"] == str(mcp_mem.id)
    assert uuid.UUID(out["receipt_id"])

    receipts = await _receipts()
    assert {r.detail["requested_by"] for r in receipts} == {"rest_api", "agent:TestAgent"}
    for receipt in receipts:
        assert receipt.detail["targets"][0]["status"] == "deleted"
    assert await _memory_ids() == set()

    async with database.AsyncSessionLocal() as session:
        ledger = (await session.execute(select(MemoryAccessLog))).scalars().all()
    assert [(row.action, row.memory_id) for row in ledger] == [("mcp_delete", None)]
    assert ledger[0].detail["memory_id"] == str(mcp_mem.id)
    assert ledger[0].detail["receipt_id"] == out["receipt_id"]


async def test_rest_delete_of_foreign_memory_is_404_with_no_receipt(db, no_chroma):
    from app.api.v1 import memories as memories_api

    uid = await _owner(db)
    owner_id = uuid.uuid4()
    db.add(User(id=owner_id, email=f"{owner_id.hex}@test.invalid", hashed_password="x",
                is_verified=True, is_active=True))
    foreign = _memory(owner_id, "not yours")
    db.add(foreign)
    await db.commit()

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        await memories_api.delete_memory(foreign.id, SimpleNamespace(id=uid), db)
    assert excinfo.value.status_code == 404
    assert await _receipts() == []
    assert foreign.id in await _memory_ids()


# ── document / conversation deletes: intents in the commit, purge after it ───


async def test_delete_document_purges_chunks_after_the_commit(db, no_chroma, monkeypatch):
    from app.retrieval import retrieval_cache, vector_retriever
    from app.services import document_service

    uid = await _owner(db)
    conversation = Conversation(id=uuid.uuid4(), user_id=uid, title="conv", document_count=1)
    doc = Document(id=uuid.uuid4(), conversation_id=conversation.id, filename="f.pdf",
                   file_path="uploads/f.pdf")
    chunks = [_chunk(doc, index=i) for i in range(2)]
    db.add_all([conversation, doc, *chunks])
    await db.flush()
    projected = _memory(uid, "projected", source_type="file_upload", source_ref=str(doc.id))
    db.add(projected)
    await db.commit()

    order: list[str] = []
    purges: list[tuple] = []
    real_commit = db.commit

    async def _commit_spy():
        order.append("commit")
        await real_commit()
        order.append("after-commit")

    async def _chunks(*args, **kwargs):
        order.append("chunk-purge")
        purges.append((args, kwargs))

    async def _vectors(*_args, **_kwargs):
        order.append("memory-purge")

    async def _noop(*_args, **_kwargs):
        return None

    db.commit = _commit_spy
    monkeypatch.setattr(vector_retriever, "delete_document_chunks", _chunks)
    monkeypatch.setattr(vector_store, "delete_memories", _vectors)
    monkeypatch.setattr("app.storage.remove_object", _noop)
    monkeypatch.setattr(retrieval_cache, "invalidate_query_cache", _noop)
    monkeypatch.setattr("app.retrieval.bm25_retriever.bm25_retriever.publish_rebuild_async", _noop)

    await document_service.delete_document(db, doc, conversation)

    assert order.index("after-commit") < order.index("chunk-purge")
    assert order.index("commit") < order.index("memory-purge")
    # The purge is the immediate attempt for the document's points, scoped to
    # the owner — the delete intents above it are the durable proof.
    assert purges == [((str(conversation.id), str(doc.id)), {"user_id": str(uid)})]

    async with database.AsyncSessionLocal() as session:
        doc_gone = await session.get(Document, doc.id)
        mem_gone = await session.get(Memory, projected.id)
        chunk_gone = await session.get(DocumentChunk, chunks[0].id)
    assert doc_gone is None and mem_gone is None and chunk_gone is None
    intents = await _outbox_rows()
    assert sorted((row.entity_id, row.operation) for row in intents) == sorted([
        (projected.id.hex, "delete"),
        (chunks[0].id.hex, "delete"),
        (chunks[1].id.hex, "delete"),
    ])


async def test_delete_session_enqueues_intents_and_purges_best_effort(db, no_chroma, monkeypatch):
    from app.api.v1 import chat
    from app.retrieval import vector_retriever

    uid = await _owner(db)
    conversation = Conversation(id=uuid.uuid4(), user_id=uid, title="session", document_count=1)
    doc = Document(id=uuid.uuid4(), conversation_id=conversation.id, filename="f.pdf",
                   file_path="uploads/f.pdf")
    chunks = [_chunk(doc, index=i) for i in range(2)]
    db.add_all([conversation, doc, *chunks])
    await db.flush()
    projected = _memory(uid, "projected", source_type="file_upload", source_ref=str(doc.id))
    db.add(projected)
    await db.commit()

    purged: list[list[str]] = []
    swept: list[tuple] = []
    invalidated: list[str] = []

    async def _vectors(ids):
        purged.append([str(i) for i in ids])

    async def _chunks(*args, **kwargs):
        swept.append((args, kwargs))

    async def _invalidate(conv_id):
        invalidated.append(conv_id)

    monkeypatch.setattr(vector_store, "delete_memories", _vectors)
    monkeypatch.setattr(vector_retriever, "delete_conversation_chunks", _chunks)
    monkeypatch.setattr(
        "app.retrieval.bm25_retriever.bm25_retriever.publish_invalidate_async", _invalidate)

    await chat.delete_session(session_id=conversation.id, current_user=SimpleNamespace(id=uid), db=db)

    async with database.AsyncSessionLocal() as session:
        assert await session.get(Conversation, conversation.id) is None
        assert await session.get(Memory, projected.id) is None
        assert await session.get(DocumentChunk, chunks[0].id) is None
    assert sorted((row.entity_id, row.operation) for row in await _outbox_rows()) == sorted([
        (projected.id.hex, "delete"),
        (chunks[0].id.hex, "delete"),
        (chunks[1].id.hex, "delete"),
    ])
    assert purged == [[str(projected.id)]]
    assert swept == [((str(conversation.id),), {"user_id": str(uid)})]
    assert invalidated == [str(conversation.id)]


# ── entities left without memory links go with the closure ───────────────────


async def test_erase_removes_entities_left_without_memory_links(db, no_chroma):
    uid = await _owner(db)
    erased = _memory(uid, "erased")
    survivor = _memory(uid, "survivor")
    orphan = Entity(id=uuid.uuid4(), user_id=uid, name="Orphan topic", entity_type="topic",
                    aliases=[])
    kept = Entity(id=uuid.uuid4(), user_id=uid, name="Kept topic", entity_type="topic",
                  aliases=[])
    db.add_all([erased, survivor, orphan, kept])
    await db.flush()
    db.add_all([
        MemoryEntity(memory_id=erased.id, entity_id=orphan.id),
        MemoryEntity(memory_id=survivor.id, entity_id=kept.id),
        Relation(user_id=uid, source_entity_id=orphan.id, target_entity_id=kept.id,
                 relation="related_to"),
    ])
    await db.commit()

    receipt = await erase_memories(db, uid, [erased.id], requested_by="rest_api")

    assert receipt.detail["targets"][0]["orphan_entities"] == 1
    async with database.AsyncSessionLocal() as session:
        assert await session.get(Entity, orphan.id) is None
        assert await session.get(Entity, kept.id) is not None
        assert (await session.execute(select(Relation))).scalars().all() == []
        assert (await session.execute(select(MemoryEntity))).scalars().all() != []
