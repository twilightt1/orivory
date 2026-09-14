"""fs storage path guard: symlinked roots must work, traversal must not."""
from __future__ import annotations

import pytest

from app import storage
from app.config import settings


def test_fs_path_accepts_symlinked_root(tmp_path, monkeypatch):
    """FS_STORAGE_PATH under a symlink (macOS /tmp -> /private/tmp) must
    still resolve object names instead of failing the traversal guard."""
    real = tmp_path / "real-root"
    real.mkdir()
    link = tmp_path / "link-root"
    link.symlink_to(real, target_is_directory=True)

    monkeypatch.setattr(settings, "STORAGE_BACKEND", "fs")
    monkeypatch.setattr(settings, "FS_STORAGE_PATH", str(link))
    monkeypatch.setattr(storage, "_fs_root", None)

    path = storage._fs_path("a/b.txt")
    assert path.is_relative_to(real.resolve())


def test_fs_path_still_refuses_traversal(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()

    monkeypatch.setattr(settings, "STORAGE_BACKEND", "fs")
    monkeypatch.setattr(settings, "FS_STORAGE_PATH", str(root))
    monkeypatch.setattr(storage, "_fs_root", None)

    with pytest.raises(ValueError):
        storage._fs_path("../escape.txt")
