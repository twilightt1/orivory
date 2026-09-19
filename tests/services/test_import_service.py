"""Unit tests for the import service — fake DB, monkeypatched indexing.

CI-safe: the single dedup select is answered by a fake returning refs
(pattern: tests/services/test_erasure_service.py); index_new_memory is
patched with a recorder (no Chroma).

Signature/counters follow the controller's binding rules, not the plan's
draft: run_import(db, user_id, raw_data, source_format, *, requested_by)
with summary counts {parsed, created, skipped_duplicates, failed,
index_failures}; parse_import is called as parse_import(data, fmt).
"""
from __future__ import annotations

import json
import uuid

import pytest
from sqlalchemy.dialects import postgresql

from app.ingestion.import_formats import ImportFormatError
from app.models.memory import Memory
from app.services import import_service
from app.services.import_service import run_import

CHATGPT_PAYLOAD = [
    {
        "id": "c1",
        "title": "First",
        "create_time": 1738454400.0,
        "mapping": {
            "u": {"message": {"author": {"role": "user"}, "create_time": 1.0,
                              "content": {"parts": ["hello"]}}},
            "a": {"message": {"author": {"role": "assistant"}, "create_time": 2.0,
                              "content": {"parts": ["hi there"]}}},
        },
    },
    {
        "id": "c2",
        "title": "Second",
        "create_time": 1738454500.0,
        "mapping": {
            "u": {"message": {"author": {"role": "user"}, "create_time": 1.0,
                              "content": {"parts": ["second conv"]}}},
        },
    },
]


class _FakeRows:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _FakeDB:
    """Answers the TWO ref-reads run_import issues — kept separate because a
    ref can be in both: the dedup select (memories) answers ``existing_refs``,
    the suppression-ledger probe answers ``suppressed_refs`` (and the ledger
    check runs FIRST: suppression owns the count when both match)."""

    def __init__(self, existing_refs=(), suppressed_refs=()):
        self._existing_refs = list(existing_refs)
        self._suppressed_refs = list(suppressed_refs)
        self.info: dict = {}  # mirrors Session.info (outbox generation memo)
        self.added = []
        self.committed = 0
        self.refreshed = []
        self.statements = []

    async def execute(self, stmt):
        self.statements.append(stmt)
        if "memory_suppressions" in str(stmt).lower():
            return _FakeRows(self._suppressed_refs)
        return _FakeRows(self._existing_refs)

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.committed += 1

    async def refresh(self, obj):
        self.refreshed.append(obj)


@pytest.fixture()
def indexed(monkeypatch):
    calls: list = []

    async def _fake_index(memory):
        calls.append(memory)

    monkeypatch.setattr(import_service, "index_new_memory", _fake_index)
    return calls


def _payload_bytes(payload) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _sql(stmt) -> str:
    return str(stmt.compile(dialect=postgresql.dialect()))


async def test_run_import_creates_and_indexes(indexed):
    db = _FakeDB()
    summary = await run_import(db, uuid.uuid4(), _payload_bytes(CHATGPT_PAYLOAD), "chatgpt",
                               requested_by="rest_api")
    assert summary.parsed == 2
    assert summary.created == 2
    assert summary.skipped_duplicates == 0
    assert summary.failed == 0
    assert summary.index_failures == 0
    assert len(db.added) == 2
    memory = db.added[0]
    assert memory.source_type == "chatgpt_import"
    assert memory.source_ref == "c1"
    assert memory.title == "First"
    assert memory.tags == ["chatgpt"]
    assert memory.captured_at.tzinfo is not None
    assert db.committed == 1
    assert len(db.refreshed) == 2
    assert indexed == db.added  # every created row indexed exactly once, in order


async def test_run_import_dedup_select_is_one_batched_query_scoped_to_user(indexed):
    """The dedup pre-check must be ONE select with source_ref IN (...), scoped
    to user_id + source_type — never N per-item queries (plan's 'batched'
    requirement, locked as a compiled-SQL guard). The batch's durable outbox
    intents ride along as inserts in the same transaction (P1a task 2)."""
    user_id = uuid.uuid4()
    db = _FakeDB()
    summary = await run_import(db, user_id, _payload_bytes(CHATGPT_PAYLOAD), "chatgpt",
                               requested_by="rest_api")

    selects = [stmt for stmt in db.statements if stmt.is_select]
    dedup = [stmt for stmt in selects if "FROM MEMORIES" in _sql(stmt).upper()]
    assert len(dedup) == 1, "run_import must issue exactly one dedup select"
    # dedup + the suppression-ledger read (T4: ONE batched ledger probe, not
    # one per item) + the outbox's active-generation read, which is memoized
    # per session (P1a task 2) — never one read per created row.
    assert len(selects) == 3
    sql = _sql(dedup[0])
    assert "IN" in sql.upper()
    params = dedup[0].compile(dialect=postgresql.dialect()).params
    assert set(params["source_ref_1"]) == {"c1", "c2"}, "refs must go in as one list param"
    # User-bind guard (carried T3 review fix): the UUID(as_uuid=True) column
    # binds the uuid object itself, so compare against user_id, not str(user_id).
    assert params["user_id_1"] == user_id, "dedup select must bind the caller's user id"
    # Every created row got its index intent in that same transaction.
    inserts = [stmt for stmt in db.statements if not stmt.is_select]
    assert len(inserts) == summary.created == 2


async def test_run_import_dedups_against_db_and_within_file(indexed):
    payload = [*CHATGPT_PAYLOAD, dict(CHATGPT_PAYLOAD[0])]  # c1 twice in-file, once in db
    db = _FakeDB(existing_refs=["c1"])
    summary = await run_import(db, uuid.uuid4(), _payload_bytes(payload), "chatgpt",
                               requested_by="rest_api")
    assert summary.created == 1          # only c2
    assert summary.skipped_duplicates == 2  # c1 in db + c1 repeated in-file
    assert [m.source_ref for m in db.added] == ["c2"]
    assert indexed == db.added           # duplicate never reaches indexing


async def test_a_soft_forgotten_ref_is_suppressed_not_a_duplicate(indexed):
    """OCR fix (P4b review): the suppression branch ran AFTER the duplicate
    branch — but a soft-forgotten source stays in SQL as an invalidated row, so
    its ref matches BOTH reads and the summary said ``skipped_duplicates``
    about a source the user explicitly forgot. Suppression is checked first
    and owns the count."""
    db = _FakeDB(existing_refs=["c1"], suppressed_refs=["c1"])
    summary = await run_import(db, uuid.uuid4(), _payload_bytes(CHATGPT_PAYLOAD), "chatgpt",
                               requested_by="rest_api")
    assert summary.suppressed_skipped == 1
    assert summary.skipped_duplicates == 0, "the ledger's reason outranks 'duplicate'"
    assert summary.created == 1          # only c2
    assert [m.source_ref for m in db.added] == ["c2"], "the suppressed ref never lands"


async def test_run_import_auto_detects_and_scopes_dedup_by_detected_type(indexed):
    claude_payload = [{
        "uuid": "k1",
        "name": "Conv",
        "created_at": "2026-01-20T09:15:00Z",
        "chat_messages": [{"uuid": "m1", "sender": "human", "text": "hi"}],
    }]
    db = _FakeDB()
    summary = await run_import(db, uuid.uuid4(), _payload_bytes(claude_payload), None,
                               requested_by="rest_api")
    assert summary.created == 1
    assert db.added[0].source_type == "claude_import"

    # The dedup select is scoped to the DETECTED format's source_type.
    params = db.statements[0].compile(dialect=postgresql.dialect()).params
    assert params["source_type_1"] == "claude_import"


async def test_run_import_rejects_non_json():
    with pytest.raises(ImportFormatError, match=r"not valid UTF-8 JSON"):
        await run_import(_FakeDB(), uuid.uuid4(), b"{not json", "generic", requested_by="rest_api")


async def test_run_import_rejects_undetectable_format():
    with pytest.raises(ImportFormatError, match=r"could not detect"):
        await run_import(_FakeDB(), uuid.uuid4(), _payload_bytes({"stray": 1}), None,
                         requested_by="rest_api")


async def test_run_import_empty_parse_creates_nothing(indexed):
    """Empty parse result → created=0, no commit, no error (brief's empty rule)."""
    db = _FakeDB()
    summary = await run_import(db, uuid.uuid4(), _payload_bytes([]), "chatgpt",
                               requested_by="rest_api")
    assert summary.parsed == 0
    assert summary.created == 0
    assert summary.skipped_duplicates == 0
    assert summary.failed == 0
    assert summary.index_failures == 0
    assert db.added == []
    assert db.committed == 0
    assert indexed == []


async def test_run_import_isolates_failed_items(indexed, monkeypatch):
    """Per-item failures are isolated (house pattern: SourceSyncService) —
    simulated via a Memory SUBCLASS that rejects one ref at construction.
    A plain replacement class would break the (real) dedup select below,
    so the seam stays a Memory and only row construction fails."""
    class _BoomMemory(Memory):
        def __init__(self, **kwargs):
            if kwargs.get("source_ref") == "c2":
                raise ValueError("simulated row failure")
            super().__init__(**kwargs)

    monkeypatch.setattr(import_service, "Memory", _BoomMemory)
    db = _FakeDB()
    summary = await run_import(db, uuid.uuid4(), _payload_bytes(CHATGPT_PAYLOAD), "chatgpt",
                               requested_by="rest_api")
    assert summary.created == 1
    assert summary.failed == 1
    assert db.committed == 1  # the good row still commits
    assert indexed == db.added  # the good row is still indexed


async def test_run_import_counts_index_failures_not_raises(indexed, monkeypatch):
    """Best-effort indexing: a raising index_new_memory is caught per row,
    counted in index_failures, and Postgres (the committed rows) is truth."""
    calls: list = []

    async def _flaky(memory):
        calls.append(memory)
        if memory.source_ref == "c1":
            raise ConnectionError("chroma down")

    monkeypatch.setattr(import_service, "index_new_memory", _flaky)
    db = _FakeDB()
    summary = await run_import(db, uuid.uuid4(), _payload_bytes(CHATGPT_PAYLOAD), "chatgpt",
                               requested_by="rest_api")
    assert summary.created == 2
    assert summary.index_failures == 1
    assert summary.failed == 0
    assert db.committed == 1
    assert len(calls) == 2  # c2 still indexed after c1 blew up
