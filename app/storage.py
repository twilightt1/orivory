"""Storage backend: the local filesystem.

Uploaded objects live under ``FS_STORAGE_PATH`` with an async surface, so
callers never branch on a backend. The MinIO client (and the full-stack
``STORAGE_BACKEND=minio`` path) went with the rest of the full-stack surface:
one container, no external services.
"""
from __future__ import annotations

from pathlib import Path

from app.config import settings

_fs_root: Path | None = None


def _fs_root_dir() -> Path:
    global _fs_root
    if _fs_root is None:
        _fs_root = Path(settings.FS_STORAGE_PATH)
        _fs_root.mkdir(parents=True, exist_ok=True)
    return _fs_root


def _fs_path(object_name: str) -> Path:
    """Resolve an object name inside the fs root, refusing traversal."""
    # Resolve BOTH sides: FS_STORAGE_PATH may itself sit under a symlink
    # (macOS /tmp -> /private/tmp, /var -> /private/var), which made the
    # string-prefix check reject every object name on those hosts.
    root = _fs_root_dir().resolve()
    candidate = (root / object_name).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"invalid object name: {object_name!r}")
    return candidate


async def ensure_bucket() -> None:
    _fs_root_dir()


async def bucket_exists(bucket_name: str | None = None) -> bool:
    # bucket_name is a legacy of the MinIO surface: the fs root is the only
    # "bucket" and callers that still pass a name get the same answer.
    return _fs_root_dir().exists()


async def put_object(object_name: str, data: bytes, content_type: str) -> None:
    path = _fs_path(object_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


async def get_object(object_name: str) -> bytes:
    return _fs_path(object_name).read_bytes()


async def remove_object(object_name: str) -> None:
    _fs_path(object_name).unlink(missing_ok=True)


async def list_objects(prefix: str) -> list[str]:
    root = _fs_root_dir()
    return sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and str(p.relative_to(root)).startswith(prefix)
    )


def get_object_sync(object_name: str) -> bytes:
    return _fs_path(object_name).read_bytes()


def put_object_sync(object_name: str, data: bytes, content_type: str) -> None:
    path = _fs_path(object_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
