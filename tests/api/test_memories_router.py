"""Wiring tests for the memories router create endpoint — CI-safe, no live DB.

`create_memory` is called directly: its parent-ownership validation raises
404 before any persistence or vector indexing, so no app/auth/db fixtures
are needed. Mirrors the routers' not-found style (agents router).
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api.v1.memories import create_memory
from app.schemas.Orivory import MemoryCreate


class _FakeDB:
    """Only `get` is reachable on the validation-404 path."""

    def __init__(self, parent):
        self._parent = parent

    async def get(self, model, pk):
        return self._parent


def _body(parent_id: uuid.UUID) -> MemoryCreate:
    return MemoryCreate(title="t", content="c", parent_id=parent_id)


async def test_create_memory_rejects_missing_parent():
    db = _FakeDB(parent=None)  # parent_id points at nothing

    with pytest.raises(HTTPException) as exc_info:
        await create_memory(_body(uuid.uuid4()), SimpleNamespace(id=uuid.uuid4()), db)

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Parent memory not found"


async def test_create_memory_rejects_foreign_parent():
    parent = SimpleNamespace(id=uuid.uuid4(), user_id=uuid.uuid4())
    db = _FakeDB(parent=parent)  # parent exists but belongs to another user

    with pytest.raises(HTTPException) as exc_info:
        await create_memory(_body(parent.id), SimpleNamespace(id=uuid.uuid4()), db)

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Parent memory not found"


# ── OCR review fix: the metadata PATCH cannot rewrite lifecycle state ───────


async def test_a_metadata_patch_cannot_clear_or_forge_lifecycle_keys(monkeypatch):
    """OCR fix (P4b review): the PATCH replaced ``extra_metadata`` wholesale, so
    ``{"metadata": {}}`` cleared ``cm_invalidated`` — an un-forget through a
    client metadata write — and a forged ``cm_superseded_by`` landed as state.
    The patch now merges: client keys replace client keys, ``cm_*`` rides
    through untouched, forged markers never land."""
    from datetime import UTC, datetime

    from app.api.v1 import memories as memories_api
    from app.models.memory import Memory
    from app.retrieval.memory.namespaces import personal_namespace
    from app.schemas.Orivory import MemoryUpdate

    uid = uuid.uuid4()
    now = datetime.now(UTC)
    memory = Memory(id=uuid.uuid4(), user_id=uid, title="t", content="c", tags=[],
                    source_type="manual_note", salience=0.5, pinned=False,
                    recall_count=0, revision=1, namespace=personal_namespace(uid),
                    extra_metadata={"cm_invalidated": True, "client": "old"},
                    captured_at=now, indexed_at=now, updated_at=now)

    class _DB:
        async def get(self, model, pk):
            return memory

        async def commit(self):
            pass

        async def refresh(self, obj):
            pass

    monkeypatch.setattr(memories_api, "bump_revision", lambda m: None)

    async def _noop_enqueue(db, m):
        pass

    async def _no_upsert(m):
        return False

    monkeypatch.setattr(memories_api, "enqueue_upsert", _noop_enqueue)
    monkeypatch.setattr(memories_api, "safe_upsert_to_index", _no_upsert)

    forged = MemoryUpdate(metadata={"client": "new", "cm_superseded_by": "evil",
                                    "cm_invalidated": None})
    out = await memories_api.update_memory(memory.id, forged,
                                           SimpleNamespace(id=uid), _DB())

    assert memory.extra_metadata["cm_invalidated"] is True, (
        "an invalidated row cannot be un-forgotten by a metadata PATCH")
    assert "cm_superseded_by" not in memory.extra_metadata, "forged markers never land"
    assert memory.extra_metadata["client"] == "new", "client keys still replace client keys"
    assert out.state == "invalidated"

    empty = MemoryUpdate(metadata={})
    await memories_api.update_memory(memory.id, empty, SimpleNamespace(id=uid), _DB())

    assert memory.extra_metadata["cm_invalidated"] is True, (
        "PATCH metadata={} never clears lifecycle state")
    assert memory.extra_metadata == {"cm_invalidated": True}, (
        "client keys go, reserved keys stay")
