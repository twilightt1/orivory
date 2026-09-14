"""Lite-mode readiness for the vector store (T2 fix round 1).

Regression pin: the lite branch of ``health_service._check_chroma`` used to
probe Chroma's HTTP client — which lite never starts, so ``GET /ready``
answered 503 ``degraded`` forever. The lite probe now opens the embedded
Qdrant owner client (in-process, no socket, one client) and nothing else.
"""
from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from app.config import settings
from app.retrieval import vector_backend
from app.services import health_service


def _lite(monkeypatch, folder) -> None:
    """LITE_MODE's shape: embedded Qdrant, local Chroma, no external services."""
    monkeypatch.setattr(settings, "QDRANT_MODE", "local")
    monkeypatch.setattr(settings, "QDRANT_LOCAL_PATH", str(folder))
    monkeypatch.setattr(settings, "CHROMA_MODE", "local")


async def _probe() -> dict:
    """The vector-store probe, exactly as ``/ready`` measures it."""
    _, payload = await health_service._measure("chroma", health_service._check_chroma)
    return payload


async def test_lite_readiness_probe_opens_and_reuses_the_qdrant_owner_client(
    monkeypatch, tmp_path
):
    folder = tmp_path / "qdrant"
    folder.mkdir()
    _lite(monkeypatch, folder)
    opened: list[object] = []

    class _CountingClient(QdrantClient):  # a REAL client that leaves a count
        def __init__(self, *args, **kwargs):
            opened.append(kwargs.get("path"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(vector_backend, "QdrantClient", _CountingClient)
    await vector_backend.close_clients()  # nothing from an earlier test survives
    try:
        for _ in range(2):
            payload = await _probe()
            assert payload["status"] == "ok", payload
        assert opened == [str(folder)]  # opened once, then the owner is reused
    finally:
        await vector_backend.close_clients()


async def test_lite_readiness_probe_fails_when_the_folder_is_foreign_owned(
    monkeypatch, tmp_path
):
    """A folder another live client owns is not a servable store: the probe
    raises the store's typed error and ``/ready`` goes degraded."""
    folder = tmp_path / "qdrant"
    folder.mkdir()
    _lite(monkeypatch, folder)
    await vector_backend.close_clients()
    foreign = QdrantClient(path=str(folder))
    try:
        with pytest.raises(RuntimeError, match="already accessed"):
            await health_service._check_chroma()
        payload = await _probe()
        assert payload["status"] == "failed"
        assert "already accessed" in payload["error"]
    finally:
        foreign.close()
        await vector_backend.close_clients()
