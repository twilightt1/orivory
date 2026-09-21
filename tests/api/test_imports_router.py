"""Wiring tests for the imports router — CI-safe, no live DB."""
from __future__ import annotations

import json
import typing
import uuid
from types import SimpleNamespace

import pytest
from fastapi.routing import APIRoute
from httpx import ASGITransport, AsyncClient

from app.api.v1 import imports as imports_module
from app.api.v1.imports import MAX_IMPORT_UPLOAD_BYTES, create_import, router
from app.database import get_db
from app.ingestion.import_formats import SOURCE_FORMATS
from app.main import app
from app.schemas.Orivory import ImportSummary
from app.services import import_service

# The two DB-FREE pins — the summary shape (T4's `suppressed_skipped`) and the
# byte-exact full-request seam — live in tests/api/test_import_summary_shape.py
# and nowhere else.
# Everything left here needs a live session.


def test_imports_routes_registered():
    paths = {route.path for route in router.routes if isinstance(route, APIRoute)}
    assert "/imports" in paths


def test_import_upload_cap_is_20mib():
    assert MAX_IMPORT_UPLOAD_BYTES == 20 * 1024 * 1024


async def test_router_prevalidates_unknown_source_format():
    """Handoff 3: unknown explicit source_format → 422 BEFORE the service
    runs (no misleading 'could not detect' error for explicitly-invalid
    input, and no service call at all)."""
    from fastapi import HTTPException

    class _RecordingDB:
        async def execute(self, stmt):  # pragma: no cover - must not be reached
            raise AssertionError("service must not run for an invalid source_format")

    file = SimpleNamespace(filename="x.json")
    with pytest.raises(HTTPException) as exc_info:
        # Call with the unknown format; body not yet read — validation
        # must fail before .read() or run_import is ever invoked.
        await create_import(
            file=file,
            source_format="not_a_format",
            current_user=SimpleNamespace(id=uuid.uuid4()),
            db=_RecordingDB(),
        )
    assert exc_info.value.status_code == 422
    assert "not_a_format" in str(exc_info.value.detail)


def test_router_prevalidation_target_locks_to_source_formats():
    """The pre-validation whitelist must be SOURCE_FORMATS itself (the
    single registry of accepted values), not a hand-copied Literal set
    that could drift from the ingestion layer."""
    from app.api.v1 import imports as imports_module

    assert imports_module.SOURCE_FORMATS is SOURCE_FORMATS


def test_memory_create_source_types_include_imports():
    """The three import source types must be legal in POST /api/v1/memories
    (brief Step 3; locked across tasks by test_source_type_mapping_locked)."""
    from app.schemas.Orivory import MemoryCreate

    annotation = MemoryCreate.model_fields["source_type"].annotation
    values = _literal_values(annotation)
    for source_type in ("chatgpt_import", "claude_import", "generic_import"):
        assert source_type in values


def test_memories_list_filter_includes_import_types():
    """GET /api/v1/memories?source_type= filter must accept the same three
    import values (brief Step 3)."""
    from app.api.v1.memories import list_memories

    hints = typing.get_type_hints(list_memories)
    values = _literal_values(hints["source_type"])
    for source_type in ("chatgpt_import", "claude_import", "generic_import"):
        assert source_type in values


async def test_router_rejects_empty_upload():
    from fastapi import HTTPException

    class _DB:  # pragma: no cover - never reached past the empty check
        pass

    class _File:
        filename = "empty.json"

        async def read(self) -> bytes:
            return b""

    with pytest.raises(HTTPException) as exc_info:
        await create_import(
            file=_File(),
            source_format="chatgpt",
            current_user=SimpleNamespace(id=uuid.uuid4()),
            db=_DB(),
        )
    assert exc_info.value.status_code == 422


async def test_router_rejects_oversized_upload():
    """413 over the 20 MiB cap — testable at wiring level with a fake file
    whose read() returns MAX+1 bytes (no network/DB needed)."""
    from fastapi import HTTPException

    class _File:
        filename = "big.json"

        async def read(self) -> bytes:
            return b"x" * (MAX_IMPORT_UPLOAD_BYTES + 1)

    class _DB:  # pragma: no cover - never reached past the size check
        pass

    with pytest.raises(HTTPException) as exc_info:
        await create_import(
            file=_File(),
            source_format="chatgpt",
            current_user=SimpleNamespace(id=uuid.uuid4()),
            db=_DB(),
        )
    assert exc_info.value.status_code == 413


async def test_router_rejects_undecodable_bytes():
    """Handoff 4 (now via the service seam): undecodable bytes reach
    run_import, whose utf-8 decode maps them to ImportFormatError →
    422, not a 500. The router no longer decodes at all (Fix 1) — the
    real service's decode path is exercised, with the DB seam stubbed."""
    from fastapi import HTTPException

    class _File:
        filename = "bad.json"

        async def read(self) -> bytes:
            return b"\xff\xfe{}"  # invalid utf-8

    class _DB:
        async def execute(self, stmt):  # pragma: no cover - decode fails first
            raise AssertionError("dedup select must not run for undecodable bytes")

    with pytest.raises(HTTPException) as exc_info:
        await create_import(
            file=_File(),
            source_format="chatgpt",
            current_user=SimpleNamespace(id=uuid.uuid4()),
            db=_DB(),
        )
    assert exc_info.value.status_code == 422


async def test_router_maps_import_format_error_to_422():
    """Handoff 4: ImportFormatError raised by run_import → HTTP 422 with
    detail=str(exc) (service seam reached with a stubbed run_import)."""
    from fastapi import HTTPException

    from app.api.v1 import imports as imports_module
    from app.ingestion.import_formats import ImportFormatError

    class _File:
        filename = "garbage.json"

        async def read(self) -> bytes:
            return b"not json at all"

    class _DB:
        pass

    async def _boom(db, user_id, raw_data, source_format, *, requested_by):
        raise ImportFormatError("import payload is not valid UTF-8 JSON: boom")

    original = imports_module.run_import
    imports_module.run_import = _boom
    try:
        with pytest.raises(HTTPException) as exc_info:
            await create_import(
                file=_File(),
                source_format="chatgpt",
                current_user=SimpleNamespace(id=uuid.uuid4()),
                db=_DB(),
            )
    finally:
        imports_module.run_import = original
    assert exc_info.value.status_code == 422
    assert "not valid UTF-8 JSON" in str(exc_info.value.detail)


# ---------------------------------------------------------------------------
# Fix round 1 — str/bytes seam, integration happy path, blank-format, CL gate
# ---------------------------------------------------------------------------


def _file_stub(payload: bytes):
    class _File:
        filename = "upload.json"

        async def read(self) -> bytes:
            return payload

    return _File()


async def test_run_import_receives_bytes_not_str(monkeypatch):
    """Fix 1 seam test — would have caught the str/bytes seam break: the
    router must hand run_import the raw BYTES (it previously decoded to a
    str, and the service's raw_data.decode("utf-8") then AttributeError'd
    → 500 on every valid upload). Stubs run_import at the router module's
    import point with an async fake asserting the received type."""
    received: list = []

    async def _fake_run_import(db, user_id, raw_data, source_format, *, requested_by):
        received.append((raw_data, source_format, requested_by))
        return ImportSummary(parsed=0, created=0, skipped_duplicates=0,
                             failed=0, index_failures=0)

    monkeypatch.setattr(imports_module, "run_import", _fake_run_import)

    payload = b'{"schema": "portable-ai-memory", "memories": []}'
    summary = await create_import(
        file=_file_stub(payload),
        source_format=None,
        current_user=SimpleNamespace(id=uuid.uuid4()),
        db=object(),
    )
    assert isinstance(received[0][0], bytes), "router must pass raw bytes to run_import"
    assert received[0][0] == payload
    assert received[0][1] is None
    assert received[0][2] == "rest_api"
    assert summary.parsed == 0


async def test_import_endpoint_happy_path_real_service(monkeypatch):
    """Fix 1 integration-ish happy path: run_import stays REAL, only the
    DB session and index_new_memory are faked (CI-safe — no Postgres, no
    Chroma). A tiny generic-format JSON array must come back 201 with the
    shipped counts; pre-fix this endpoint 500'd on every valid upload."""
    class _FakeResult:
        def scalars(self):
            return self

        def all(self):
            return []  # no pre-existing refs → nothing deduped away

    class _FakeDB:
        def __init__(self):
            self.added = []
            self.committed = 0
            self.refreshed = []

        async def execute(self, stmt):
            return _FakeResult()

        def add(self, obj):
            self.added.append(obj)

        async def commit(self):
            self.committed += 1

        async def refresh(self, obj):
            self.refreshed.append(obj)

    async def _fake_index(memory):
        return None

    monkeypatch.setattr(import_service, "index_new_memory", _fake_index)
    db = _FakeDB()

    async def _db_override():
        yield db

    async def _current_user_override():
        return SimpleNamespace(id=uuid.uuid4())

    # The endpoint resolves its caller through `_optional_user` (local owner /
    # agent token), not the `get_current_user` dependency — override the
    # former, which is the one the route actually depends on.
    app.dependency_overrides[imports_module._optional_user] = _current_user_override
    app.dependency_overrides[get_db] = _db_override
    try:
        payload = json.dumps([
            {"content": "first generic memory", "ref": "g1", "tags": ["t"]},
            {"content": "second generic memory", "ref": "g2"},
        ]).encode("utf-8")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/imports",
                files={"file": ("generic.json", payload, "application/json")},
                data={"source_format": "generic"},
            )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["parsed"] == 2
    assert body["created"] == 2
    assert body["skipped_duplicates"] == 0
    assert body["failed"] == 0
    assert body["index_failures"] == 0
    # service-side proof the rows actually materialized through the real path
    assert len(db.added) == 2
    assert db.committed == 1
    assert {m.source_ref for m in db.added} == {"g1", "g2"}
    assert all(m.source_type == "generic_import" for m in db.added)


async def test_blank_source_format_normalized_to_auto_detect(monkeypatch):
    """Fix 3: a blank/whitespace source_format form field means the same
    as an omitted one (auto-detect), not a 422."""
    received: list = []

    async def _fake_run_import(db, user_id, raw_data, source_format, *, requested_by):
        received.append(source_format)
        return ImportSummary(parsed=0, created=0, skipped_duplicates=0,
                             failed=0, index_failures=0)

    monkeypatch.setattr(imports_module, "run_import", _fake_run_import)

    for blank in ("", "   "):
        summary = await create_import(
            file=_file_stub(b"[]"),
            source_format=blank,
            current_user=SimpleNamespace(id=uuid.uuid4()),
            db=object(),
        )
        assert summary.created == 0
        assert received[-1] is None, f"blank {blank!r} must normalize to None (auto-detect)"


async def test_blank_source_format_over_http_behaves_as_omitted(monkeypatch):
    """Fix 3, through the ASGI stack: posting source_format="" must give
    the same 201/auto-detect outcome as omitting the field entirely."""
    received: list = []

    async def _fake_run_import(db, user_id, raw_data, source_format, *, requested_by):
        received.append(source_format)
        return ImportSummary(parsed=0, created=0, skipped_duplicates=0,
                             failed=0, index_failures=0)

    monkeypatch.setattr(imports_module, "run_import", _fake_run_import)

    async def _current_user_override():
        return SimpleNamespace(id=uuid.uuid4())

    async def _db_override():
        yield object()

    # The endpoint resolves its caller through `_optional_user` (local owner /
    # agent token), not the `get_current_user` dependency — override the
    # former, which is the one the route actually depends on.
    app.dependency_overrides[imports_module._optional_user] = _current_user_override
    app.dependency_overrides[get_db] = _db_override
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            blank = await client.post(
                "/api/v1/imports",
                files={"file": ("items.json", b"[]", "application/json")},
                data={"source_format": ""},
            )
            omitted = await client.post(
                "/api/v1/imports",
                files={"file": ("items.json", b"[]", "application/json")},
            )
    finally:
        app.dependency_overrides.clear()
    assert blank.status_code == 201, blank.text
    assert omitted.status_code == 201, omitted.text
    assert received[-2:] == [None, None]


async def test_content_length_header_over_cap_early_rejected(monkeypatch):
    """Fix 2 cheap hardening: a Content-Length header above the cap (even
    with a body under it) is rejected 413 BEFORE the read — best-effort,
    header-trusting; the read-then-check stays the authoritative gate."""
    async def _fake_run_import(db, user_id, raw_data, source_format, *, requested_by):
        raise AssertionError("run_import must not run for an over-cap Content-Length")

    monkeypatch.setattr(imports_module, "run_import", _fake_run_import)

    async def _current_user_override():
        return SimpleNamespace(id=uuid.uuid4())

    async def _db_override():
        yield object()

    # The endpoint resolves its caller through `_optional_user` (local owner /
    # agent token), not the `get_current_user` dependency — override the
    # former, which is the one the route actually depends on.
    app.dependency_overrides[imports_module._optional_user] = _current_user_override
    app.dependency_overrides[get_db] = _db_override
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/imports",
                files={"file": ("big.json", b"[]", "application/json")},
                data={"source_format": "generic"},
                headers={"Content-Length": str(MAX_IMPORT_UPLOAD_BYTES + 1)},
            )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 413, response.text
    assert "Content-Length" in response.json()["detail"]


def _literal_values(annotation) -> tuple:
    """Unwrap Optional[Literal[...]] / Literal[...] → the Literal's values."""
    if typing.get_origin(annotation) is typing.Literal:
        return typing.get_args(annotation)
    for arg in typing.get_args(annotation):
        if typing.get_origin(arg) is typing.Literal:
            return typing.get_args(arg)
    return ()


async def test_whitespace_wrapped_source_format_resolves_normally(monkeypatch):
    """Fix wave: strip BEFORE blankness check — '  chatgpt  ' must resolve
    to the chatgpt format (not 422, not auto-detect)."""
    received: list = []

    async def _fake_run_import(db, user_id, raw_data, source_format, *, requested_by):
        received.append(source_format)
        return ImportSummary(parsed=0, created=0, skipped_duplicates=0,
                             failed=0, index_failures=0)

    monkeypatch.setattr(imports_module, "run_import", _fake_run_import)

    summary = await create_import(
        file=_file_stub(b'[{"content": "x"}]'),
        source_format="  chatgpt  ",
        current_user=SimpleNamespace(id=uuid.uuid4()),
        db=object(),
    )
    assert summary.parsed == 0
    assert received == ["chatgpt"]
