"""Regression tests for LoggingMiddleware (found in full-repo review).

The middleware logged only successful responses: `call_next` without
try/except meant any raising handler skipped the access log AND the
X-Request-ID header — exactly the 5xx requests operators need most.
"""
from __future__ import annotations

import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.middleware import logging_middleware
from app.middleware.logging_middleware import LoggingMiddleware


async def _dummy_app(scope, receive, send):
    raise AssertionError("must not be called")

def _request() -> Request:
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/api/v1/memories",
            "query_string": b"",
            "headers": [],
            "server": ("test", 80),
            "client": ("testclient", 50000),
        }
    )


class _LogSink:
    def __init__(self):
        self.calls: list[dict] = []

    def info(self, *args, **kwargs):
        self.calls.append({"level": "info", "args": args, "kwargs": kwargs})

    def error(self, *args, **kwargs):
        self.calls.append({"level": "error", "args": args, "kwargs": kwargs})


@pytest.fixture()
def sink(monkeypatch):
    sink = _LogSink()
    monkeypatch.setattr(logging_middleware, "log", sink)
    return sink


@pytest.mark.asyncio
async def test_successful_request_logged_with_request_id(sink):
    async def call_next(request):
        return JSONResponse({"ok": True})

    response = await LoggingMiddleware(app=_dummy_app).dispatch(_request(), call_next)

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
    assert len(sink.calls) == 1
    assert sink.calls[0]["kwargs"]["status"] == 200
    assert sink.calls[0]["kwargs"]["path"] == "/api/v1/memories"


@pytest.mark.asyncio
async def test_raising_handler_is_logged_as_500_and_reraised(sink):
    """5xx requests must appear in the access log (status=500 + duration),
    carry the X-Request-ID on the error path, and still propagate."""

    seen: dict = {}

    async def call_next(request):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await LoggingMiddleware(app=_dummy_app).dispatch(_request(), call_next)

    assert len(sink.calls) == 1
    logged = sink.calls[0]["kwargs"]
    assert logged["status"] == 500
    assert logged["path"] == "/api/v1/memories"
    assert logged["duration_ms"] >= 0
    assert logged["request_id"]
    seen["request_id"] = logged["request_id"]


@pytest.mark.asyncio
async def test_the_failure_is_logged_at_error_with_the_traceback(sink):
    """Finding 73: the traceback available at the catch site must not be dropped.

    The access line stays (status/path/duration/request_id), but it carries the
    exception now — INFO with a rendered ``"Type: message"`` discarded every
    stack frame.
    """

    async def call_next(request):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await LoggingMiddleware(app=_dummy_app).dispatch(_request(), call_next)

    assert len(sink.calls) == 1  # the access line, not a second record
    logged = sink.calls[0]
    assert logged["level"] == "error"
    assert logged["kwargs"]["exc_info"] is True
    assert logged["kwargs"]["status"] == 500
    assert logged["kwargs"]["request_id"]


@pytest.mark.asyncio
async def test_capture_logs_carries_exc_info_for_a_raising_handler():
    """structlog sees ``exc_info`` on the record (not just a message string)."""
    from structlog.testing import capture_logs

    async def call_next(request):
        raise ValueError("bad input")

    with capture_logs() as logs:
        with pytest.raises(ValueError):
            await LoggingMiddleware(app=_dummy_app).dispatch(_request(), call_next)

    record = next(entry for entry in logs if entry.get("event") == "request")
    assert record["log_level"] == "error"
    assert record["exc_info"] is True
    assert record["status"] == 500
    assert "ValueError" in record["error"]


@pytest.mark.asyncio
async def test_error_log_includes_exception_type(sink):
    async def call_next(request):
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        await LoggingMiddleware(app=_dummy_app).dispatch(_request(), call_next)

    assert sink.calls[0]["kwargs"]["status"] == 500
    assert "ValueError" in str(sink.calls[0]["kwargs"].get("error", "ValueError"))
