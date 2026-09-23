"""Tests for the OpenClaw auto-capture daemon (stdlib-only script).

Covers: markdown→entries conversion (turn detection), file fingerprinting,
state persistence, the scan loop's dedup/dry-run behavior (post_import
monkeypatched), and the ONE thing about post_import itself that no mock can
see: the request shape the endpoint actually declares.
"""
from __future__ import annotations

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_spec = importlib.util.spec_from_file_location(
    "openclaw_capture",
    Path(__file__).resolve().parents[2] / "scripts" / "openclaw_capture.py",
)
capture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(capture)


def test_md_to_entries_turn_detection():
    text = """**User:** Remember we chose pgvector
Some extra detail here.

**Assistant:** Got it — pgvector it is.

Random prose without a marker should continue the last turn."""
    entries = capture.md_to_entries(text)
    assert len(entries) == 2
    assert entries[0]["role"] == "user"
    assert "pgvector" in entries[0]["content"]
    assert "Some extra detail" in entries[0]["content"]  # continues the turn
    assert entries[1]["role"] == "assistant"


def test_md_to_entries_heading_style():
    text = "## User\nhello\n## Assistant\nhi there"
    entries = capture.md_to_entries(text)
    assert [e["role"] for e in entries] == ["user", "assistant"]


def test_md_to_entries_no_markers_single_user_entry():
    entries = capture.md_to_entries("just some notes without turn markers")
    assert len(entries) == 1
    assert entries[0]["role"] == "user"


def test_md_to_entries_empty_file():
    assert capture.md_to_entries("   \n\n  ") == []


def test_to_session_payload_shape(tmp_path):
    md = tmp_path / "2026-09-05_session.md"
    md.write_text("**User:** remember this fact\n")
    payload = capture.to_session_payload(md, md.read_text())
    assert payload["entries"] == [{"role": "user", "content": "remember this fact"}]
    assert payload["session_id"]
    assert "session" in payload["title"].lower()


def test_scan_dedup_via_state(tmp_path, monkeypatch):
    md = tmp_path / "note.md"
    md.write_text("**User:** capture me once\n")
    state_file = tmp_path / ".state.json"
    posted: list[dict] = []

    monkeypatch.setattr(capture, "post_import",
                        lambda url, token, payload: posted.append(payload) or {"created": 1})

    state = {}
    captured, _ = capture.scan_and_capture(tmp_path, state, "http://x", "oa_t")
    assert captured == 1 and len(posted) == 1

    # second scan, file unchanged → skipped, not re-posted
    captured2, skipped2 = capture.scan_and_capture(tmp_path, state, "http://x", "oa_t")
    assert captured2 == 0 and skipped2 == 1 and len(posted) == 1

    capture.save_state(state_file, state)
    assert json.loads(state_file.read_text())[str(md)]


def test_scan_forgets_deleted_files(tmp_path, monkeypatch):
    # state references a file that no longer exists on disk → forget it,
    # and never attempt to read/POST anything (post_import would fail).
    ghost = str(tmp_path / "gone.md")
    state = {ghost: "stale-fingerprint"}

    def _fail_on_post(url, token, payload):
        raise AssertionError("must not POST for a deleted file")

    monkeypatch.setattr(capture, "post_import", _fail_on_post)
    captured, skipped = capture.scan_and_capture(tmp_path, state, "http://x", "oa_t")
    assert ghost not in state
    assert captured == 0 and skipped == 0


def test_state_survives_new_file_addition(tmp_path, monkeypatch):
    a = tmp_path / "a.md"
    a.write_text("**User:** first\n")
    state = {}
    monkeypatch.setattr(capture, "post_import",
                        lambda url, token, payload: {"created": 1})
    capture.scan_and_capture(tmp_path, state, "http://x", "oa_t")

    b = tmp_path / "b.md"
    b.write_text("**User:** second\n")
    captured, _ = capture.scan_and_capture(tmp_path, state, "http://x", "oa_t")
    assert captured == 1  # only the new file


def test_dry_run_does_not_mutate_state(tmp_path, monkeypatch):
    """Regression: --dry-run recorded fingerprints as if captured, so the
    next REAL run skipped those files and silently captured nothing."""
    md = tmp_path / "note.md"
    md.write_text("**User:** dry-run me\n")
    state: dict = {}

    def _fail_on_post(url, token, payload):
        raise AssertionError("dry run must never POST")

    monkeypatch.setattr(capture, "post_import", _fail_on_post)
    captured, _ = capture.scan_and_capture(
        tmp_path, state, "http://x", "oa_t", dry_run=True
    )
    assert captured == 1
    assert state == {}, "dry run must leave persistent state untouched"

    # A subsequent real run still captures the file.
    posted: list[dict] = []
    monkeypatch.setattr(capture, "post_import",
                        lambda url, token, payload: posted.append(payload) or {"created": 1})
    captured2, _ = capture.scan_and_capture(tmp_path, state, "http://x", "oa_t")
    assert captured2 == 1 and len(posted) == 1


def test_post_import_sends_the_multipart_body_the_endpoint_declares():
    """`/api/v1/imports` declares `file: UploadFile = File(...)`.

    Review finding (verified): the daemon posted a JSON body instead, and the
    endpoint answers 422 for that on every upload — auto-capture could never
    work. Asserted against the real request on the wire, because a monkeypatch
    (the rest of this file) cannot see a body shape.
    """
    seen: dict[str, Any] = {}

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            seen["body"] = self.rfile.read(length)
            seen["content_type"] = self.headers.get("Content-Type", "")
            seen["authorization"] = self.headers.get("Authorization", "")
            payload = b'{"created": 1}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        summary = capture.post_import(
            f"http://127.0.0.1:{server.server_port}",
            "oa_token",
            {"entries": [{"role": "user", "content": "remember this"}]},
        )
    finally:
        server.shutdown()
        server.server_close()

    assert summary == {"created": 1}
    assert seen["authorization"] == "Bearer oa_token"
    content_type: str = seen["content_type"]
    assert content_type.startswith("multipart/form-data; boundary="), content_type
    boundary = content_type.split("boundary=", 1)[1].encode()
    body: bytes = seen["body"]
    assert b'name="file"' in body and b'filename="' in body
    assert b"remember this" in body
    assert boundary in body
    assert b'name="source_format"' in body and b"openclaw" in body


def test_the_endpoint_still_declares_the_multipart_field_we_send():
    """Pin the PAIR: the defect was a mismatch, not either half.

    If this route ever changes to accepting a JSON body, the wire-shape test
    above must change with it — this assertion is what makes that visible.
    """
    imports = (
        Path(__file__).resolve().parents[2] / "app" / "api" / "v1" / "imports.py"
    ).read_text()
    assert 'file: UploadFile = File(..., description="Export file (JSON)' in imports
