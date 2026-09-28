"""
Artifact storage for Orivory experiments.

Lightweight wrapper around local filesystem for storing prompt versions,
datasets, and config snapshots attached to experiment runs.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("eval/experiments/_artifacts")


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ArtifactStore:
    """Filesystem-backed artifact store with content hashing."""

    def __init__(self, root: str | Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_text(self, name: str, text: str, kind: str = "text") -> dict[str, Any]:
        digest = sha256_text(text)
        out = self.root / kind / f"{digest[:12]}_{name}"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        meta = {
            "name": name,
            "kind": kind,
            "sha256": digest,
            "path": str(out),
            "saved_at": _utcnow_iso(),
        }
        meta_path = out.with_suffix(out.suffix + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    def save_json(self, name: str, payload: dict[str, Any], kind: str = "json") -> dict[str, Any]:
        text = json.dumps(payload, indent=2, default=str)
        return self.save_text(name + ".json", text, kind=kind)

    def load_text(self, sha256: str) -> str:
        for path in self.root.rglob(f"{sha256[:12]}_*"):
            if path.name.endswith(".meta.json"):
                continue
            return path.read_text(encoding="utf-8")
        raise FileNotFoundError(f"No artifact with sha256={sha256}")

    def list_artifacts(self, kind: str | None = None) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for meta_path in self.root.rglob("*.meta.json"):
            if kind and kind not in meta_path.parts:
                continue
            try:
                results.append(json.loads(meta_path.read_text(encoding="utf-8")))
            except Exception:  # pragma: no cover
                continue
        return sorted(results, key=lambda r: r.get("saved_at", ""), reverse=True)

    def copy_file(self, src: str | Path, kind: str = "file") -> dict[str, Any]:
        src = Path(src)
        digest = sha256_text(src.read_text(encoding="utf-8"))
        dest_dir = self.root / kind
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{digest[:12]}_{src.name}"
        shutil.copy2(src, dest)
        meta = {
            "name": src.name,
            "kind": kind,
            "sha256": digest,
            "path": str(dest),
            "original_path": str(src),
            "saved_at": _utcnow_iso(),
        }
        meta_path = dest.with_suffix(dest.suffix + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta



