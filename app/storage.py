"""Storage backend: MinIO (full stack) or local filesystem (lite mode).

``STORAGE_BACKEND=minio`` (default) uses MinIO; ``STORAGE_BACKEND=fs``
stores objects under ``FS_STORAGE_PATH`` with the same async surface, so
callers never branch on the backend.
"""
from __future__ import annotations

import asyncio
import io
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — imported lazily in _get_client()
    from minio import Minio
    from minio.error import S3Error

from app.config import settings

_client: Minio | None = None


def _get_client() -> Minio:
    global _client
    if _client is None:
        from minio import Minio as _Minio

        _client = _Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
    return _client


_fs_root: Path | None = None


def _fs_root_dir() -> Path:
    global _fs_root
    if _fs_root is None:
        _fs_root = Path(settings.FS_STORAGE_PATH)
        _fs_root.mkdir(parents=True, exist_ok=True)
    return _fs_root


def _fs_path(object_name: str) -> Path:
    """Resolve an object name inside the fs root, refusing traversal."""
    root = _fs_root_dir()
    candidate = (root / object_name).resolve()
    if not str(candidate).startswith(str(root)):
        raise ValueError(f"invalid object name: {object_name!r}")
    return candidate


def _use_fs() -> bool:
    return settings.STORAGE_BACKEND == "fs" or (settings.LITE_MODE and not settings.MINIO_ACCESS_KEY)


async def ensure_bucket() -> None:
    if _use_fs():
        _fs_root_dir()
        return
    loop = asyncio.get_event_loop()
    client = _get_client()
    exists = await loop.run_in_executor(
        None, partial(client.bucket_exists, settings.MINIO_BUCKET)
    )
    if not exists:
        await loop.run_in_executor(
            None, partial(client.make_bucket, settings.MINIO_BUCKET)
        )


async def bucket_exists(bucket_name: str | None = None) -> bool:
    if _use_fs():
        return _fs_root_dir().exists()
    loop = asyncio.get_event_loop()
    client = _get_client()
    return await loop.run_in_executor(
        None,
        partial(client.bucket_exists, bucket_name or settings.MINIO_BUCKET),
    )


async def put_object(object_name: str, data: bytes, content_type: str) -> None:
    if _use_fs():
        path = _fs_path(object_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return
    loop = asyncio.get_event_loop()
    client = _get_client()
    await loop.run_in_executor(
        None,
        partial(
            client.put_object,
            settings.MINIO_BUCKET,
            object_name,
            io.BytesIO(data),
            len(data),
            content_type=content_type,
        ),
    )


async def get_object(object_name: str) -> bytes:
    if _use_fs():
        return _fs_path(object_name).read_bytes()
    loop = asyncio.get_event_loop()
    client = _get_client()

    def _read():
        response = client.get_object(settings.MINIO_BUCKET, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    return await loop.run_in_executor(None, _read)


async def remove_object(object_name: str) -> None:
    if _use_fs():
        path = _fs_path(object_name)
        path.unlink(missing_ok=True)
        return
    loop = asyncio.get_event_loop()
    client = _get_client()
    try:
        await loop.run_in_executor(
            None,
            partial(client.remove_object, settings.MINIO_BUCKET, object_name),
        )
    except Exception as exc:
        from minio.error import S3Error

        if not isinstance(exc, S3Error):
            raise
        # Missing object on MinIO is fine — fs branch uses missing_ok=True.


async def list_objects(prefix: str) -> list[str]:
    if _use_fs():
        root = _fs_root_dir()
        return sorted(
            str(p.relative_to(root))
            for p in root.rglob("*")
            if p.is_file() and str(p.relative_to(root)).startswith(prefix)
        )
    loop = asyncio.get_event_loop()
    client = _get_client()

    def _list():
        return [
            obj.object_name
            for obj in client.list_objects(settings.MINIO_BUCKET, prefix=prefix, recursive=True)
        ]

    return await loop.run_in_executor(None, _list)


def get_object_sync(object_name: str) -> bytes:
    if _use_fs():
        return _fs_path(object_name).read_bytes()
    client = _get_client()
    response = client.get_object(settings.MINIO_BUCKET, object_name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def put_object_sync(object_name: str, data: bytes, content_type: str) -> None:
    if _use_fs():
        path = _fs_path(object_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return
    client = _get_client()
    client.put_object(
        settings.MINIO_BUCKET, object_name,
        io.BytesIO(data), len(data), content_type=content_type,
    )
