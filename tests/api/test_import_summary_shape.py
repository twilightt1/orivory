"""DB-free imports pins: the summary shape (T4) and the byte-exact seam.

Split out of ``tests/api/test_imports_router.py`` on purpose. That module
carries an autouse Postgres probe, so when the test database is down the WHOLE
module skips — including these two pins, which never touch a database at all.
The T4 shape pin (`suppressed_skipped`, additive) was therefore invisible to
every run without Postgres, and the P4b CI step (temp SQLite, no services)
could not gate it. Everything here needs only the schema, the router function
and the ASGI app, so it runs everywhere the interpreter does.

The PG-gated seam tests that need a real session stay in
``test_imports_router.py``; these two now live HERE and nowhere else.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

from httpx import ASGITransport, AsyncClient

from app.api.v1 import imports as imports_module
from app.database import get_db
from app.main import app
from app.schemas.Orivory import ImportSummary


def test_import_summary_response_shape():
    """Shipped 5-count schema (T3 binding rules) + the T4 suppression counter —
    the plan's old field set (detected_format/filename/errors/...) does not exist.

    DB-free by construction (schema-only: no session, no client), which is why
    it lives outside the PG-gated module: a dropped `suppressed_skipped` field
    must fail in CI, not skip there.
    """
    fields = set(ImportSummary.model_fields)
    assert fields == {"parsed", "created", "skipped_duplicates", "suppressed_skipped",
                      "failed", "index_failures"}
    assert ImportSummary.model_fields["suppressed_skipped"].default == 0, (
        "additive: an older caller's 5-count construction keeps working")


async def test_import_endpoint_passes_bytes_full_request(monkeypatch):
    """Fix 1, full-request variant: multipart POST through the ASGI app
    (dependency overrides for auth+db) must hand the service the exact
    uploaded bytes — not a decoded str, not a re-wrapped one. The response body
    is also the six-key shape over the wire (the same pin, one layer up); the
    DB override yields a bare object, so no database is involved.
    """
    received: list = []

    async def _fake_run_import(db, user_id, raw_data, source_format, *, requested_by):
        received.append(raw_data)
        return ImportSummary(parsed=1, created=1, skipped_duplicates=0,
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
        payload = b'[{"content": "hello", "ref": "r1"}]'
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/imports",
                files={"file": ("items.json", payload, "application/json")},
                data={"source_format": "generic"},
            )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 201
    assert response.json() == {"parsed": 1, "created": 1, "skipped_duplicates": 0,
                              "suppressed_skipped": 0, "failed": 0, "index_failures": 0}
    assert received == [payload], "service must receive the exact uploaded bytes"
