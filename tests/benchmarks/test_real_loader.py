"""Regression: real LongMemEval-S ships numeric golds — loader coerces."""
from __future__ import annotations

from eval.benchmarks.longmemeval_s import _require_str


def test_require_str_coerces_numeric_answers():
    assert _require_str({"answer": 3}, "answer", "ctx", coerce_number=True) == "3"
    assert _require_str({"answer": 2.0}, "answer", "ctx", coerce_number=True) == "2.0"


def test_require_str_still_strict_without_coercion():
    import pytest

    with pytest.raises(ValueError, match="must be a string"):
        _require_str({"answer": 3}, "answer", "ctx")
    # bools are never coerced (True is not "True" in the dataset's semantics)
    with pytest.raises(ValueError, match="must be a string"):
        _require_str({"answer": True}, "answer", "ctx", coerce_number=True)


def test_require_str_accepts_strings():
    assert _require_str({"answer": "Runkeeper"}, "answer", "ctx") == "Runkeeper"


def test_vector_store_async_face_is_awaitable_in_local_mode(monkeypatch, tmp_path):
    """Regression from the real system run: the embedded store hands back a
    SYNCHRONOUS client while the store's read path awaits it — every search
    then failed silently (recalled=0). In lite mode the async face must expose
    awaitable store calls, driven from the ONE embedded client."""
    import asyncio
    import inspect

    # Patch settings, not just env: app.config.settings is a module
    # singleton built at first import (likely BEFORE this test set any
    # env), and the backend reads settings.* — env changes alone are
    # invisible in a combined test run.
    from app.config import settings
    from app.retrieval import vector_backend

    monkeypatch.setattr(settings, "QDRANT_MODE", "local")
    monkeypatch.setattr(settings, "QDRANT_LOCAL_PATH", str(tmp_path / "qdrant"))

    async def _check():
        try:
            client = vector_backend.get_async_client()
            return (
                inspect.iscoroutinefunction(client.count),
                inspect.iscoroutinefunction(client.query_points),
                client._inner is vector_backend.get_sync_client()._inner,
            )
        finally:
            await vector_backend.close_clients()

    count_async, query_async, one_owner = asyncio.run(_check())
    assert count_async and query_async, (
        "local-mode store calls must be awaitable — raw sync methods silently "
        "break recall"
    )
    assert one_owner, "both faces must drive the ONE embedded client"
