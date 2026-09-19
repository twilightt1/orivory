"""arctic-embed-m-v2.0 ONNX runner — eval-only, standalone from ``app/``.

Why standalone
--------------
``eval/`` is experiment tooling: importing ``app.config`` for a cache dir would
drag the whole settings surface (and ``.env`` handling) into every ablation run.
This module touches only ``onnxruntime``, ``tokenizers``, ``numpy`` and stdlib.

Contract (spec §3.3/§3.4)
-------------------------
- 768-dim, **CLS pooling** (``last_hidden_state[:, 0, :]`` — the trained form).
- **MRL**: keep the first ``dim`` coordinates, then L2-normalize AGAIN. Outputs
  are plain float vectors — never the 128-byte / 4-bit claim from the model card.
- **Prefixes**: queries get exactly ``"query: "``; passages get NOTHING.
- Tokenization is capped at 512 tokens (``MAX_TOKENS``) regardless of the 8192
  the checkpoint declares; 8192 is deliberately NOT enabled until measured.
- Batch: encode together, pad to the longest truncated sequence, attention mask
  respected. Padding fills with the tokenizer's real ``<pad>`` id (1 for this
  export family — filling with 0 would seed ``<s>`` at every padded position).
  FP32 is batch-invariant within parity tolerance; **INT8 vectors depend on
  their batch mates** (per-tensor dynamic quantization scales; parity measured
  cos 0.95-0.98 between a solo and a batched embedding) — use one text per call
  (``Mv2SoloText``) when exact INT8 vectors must stay stable.

Cache (outside the repo, same pattern as the XS cache under ~/.cache/orivory/e5)
-------------------------------------------------------------------------------
``~/.cache/orivory/mv2`` (override: ``ORIVORY_MV2_DIR``) holds the INT8 + FP32
ONNX exports and the tokenizer, each pinned to the sha256 HuggingFace publishes
for revision ``HF_REVISION``. ``ensure_mv2_files(*names)`` downloads+verifies
what you ask for; ``mv2_files_cached(*names)`` answers without touching the net
(CI skip guard). Pinned revision, never a branch: the artifact must be the
bytes the report claims.

Fixture-scale parity harness: ``python eval/mv2/parity.py`` (writes
``eval/mv2/parity_report.json``). Contract tests (CI-safe, skip when uncached):
``tests/benchmarks/test_mv2_runner.py``.
"""
from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

HF_REPO = "Snowflake/snowflake-arctic-embed-m-v2.0"
HF_REVISION = "95c2741480856aa9666782eb4afe11959938017f"
_BASE_URL = f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}"

INT8_FILE = "mv2_int8.onnx"
FP32_FILE = "mv2_fp32.onnx"
TOKENIZER_FILE = "mv2_tokenizer.json"
INT8_URL = f"{_BASE_URL}/onnx/model_int8.onnx"
FP32_URL = f"{_BASE_URL}/onnx/model.onnx"
TOKENIZER_URL = f"{_BASE_URL}/tokenizer.json"
# sha256 of the artifacts at HF_REVISION (HF LFS oid, re-verified on download
# and on every ensure call over a cached file).
INT8_SHA256 = "03d923bb1850ebdccb068e2f3abd8aa43fe81c50d07d037ef103fe3d0fb78e3b"
FP32_SHA256 = "c0c53d7f49a2db60761b92b7bbf5be87a7b3cf5d92dbbd7f1b5028bd5a40aa39"
TOKENIZER_SHA256 = "f1cc44ad7faaeec47241864835473fd5403f2da94673f3f764a77ebcb0a803ec"

_FILES: dict[str, tuple[str, str, str]] = {
    "int8": (INT8_FILE, INT8_URL, INT8_SHA256),
    "fp32": (FP32_FILE, FP32_URL, FP32_SHA256),
    "tokenizer": (TOKENIZER_FILE, TOKENIZER_URL, TOKENIZER_SHA256),
}

QUERY_PREFIX = "query: "
MAX_TOKENS = 512
_BATCH = 16
_HEARTBEAT_BYTES = 256 * 1024 * 1024


def model_dir() -> Path:
    configured = (os.environ.get("ORIVORY_MV2_DIR") or "").strip()
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "orivory" / "mv2"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, dest: Path, expected_sha256: str) -> None:
    """Stream to ``.part``, hashing on the way; rename only on digest match."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.warning("Downloading m-v2 artifact %s\n  from %s", dest.name, url)
    digest = hashlib.sha256()
    got = 0
    next_mark = _HEARTBEAT_BYTES
    with urllib.request.urlopen(url) as source, tmp.open("wb") as sink:
        while block := source.read(8 * 1024 * 1024):
            sink.write(block)
            digest.update(block)
            got += len(block)
            if got >= next_mark:
                log.warning("  %s: %.0f MB", dest.name, got / 1e6)
                next_mark += _HEARTBEAT_BYTES
    actual = digest.hexdigest()
    if actual != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"{dest.name}: sha256 {actual} != pinned {expected_sha256}")
    tmp.rename(dest)


def _resolve(name: str) -> Path:
    try:
        return model_dir() / _FILES[name][0]
    except KeyError:
        raise ValueError(f"unknown m-v2 artifact {name!r} — expected {sorted(_FILES)}") from None


def mv2_files_cached(*names: str) -> bool:
    """True when the named artifacts exist (no digest check, no network).

    Defaults to what a real session needs to load ('int8' + 'tokenizer') — this
    is the CI skip guard; use :func:`ensure_mv2_files` to make them exist.
    """
    return all(_resolve(n).exists() for n in (names or ("int8", "tokenizer")))


def ensure_mv2_files(*names: str) -> dict[str, Path]:
    """Verify-or-download the named artifacts; defaults to ALL THREE (~1.5 GB).

    Pass ``ensure_mv2_files("int8", "tokenizer")`` (~330 MB) for a load-only
    check; the parity harness wants both exports.
    """
    out: dict[str, Path] = {}
    for name in names or tuple(_FILES):
        path = _resolve(name)
        url = _FILES[name][1]
        expected = _FILES[name][2]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or _digest(path) != expected:
            path.unlink(missing_ok=True)
            _download(url, path, expected)
        out[name] = path
    return out


def _session_options(intra_op_threads: int = 0):
    """ORT options mirroring ``e5_local._session_options`` (0 = ORT's choice)."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    return options


def _feed(sess, ids: np.ndarray, mask: np.ndarray) -> dict:
    """Build the input feed from the session's own input names.

    Single texts => all-zero segment ids, and token_type_ids only when the
    export declares it (the Xenova-e5 lesson, kept for a new export).
    """
    names = {i.name for i in sess.get_inputs()}
    feed = {}
    if "input_ids" in names:
        feed["input_ids"] = ids
    if "attention_mask" in names:
        feed["attention_mask"] = mask
    if "token_type_ids" in names:
        feed["token_type_ids"] = np.zeros_like(ids)
    return feed


class Mv2Onnx:
    """arctic-embed-m-v2.0 ONNX embedder: CLS pool, optional MRL slice, L2.

    Same call shape as ``e5_local.arctic_embed_queries/passages``
    (``list[list[float]]``) so the T2 ablation can swap arms without a shim.
    """

    def __init__(
        self,
        model_path: str | Path,
        pool: str = "cls",
        dim: int = 768,
        prefix: str = QUERY_PREFIX,
        tokenizer_path: str | Path | None = None,
        intra_op_threads: int = 0,
    ) -> None:
        if pool != "cls":
            raise ValueError(f"unknown pooling {pool!r} — m-v2 ships CLS only")
        if dim not in (256, 768):
            raise ValueError(f"unsupported dim {dim!r} — MRL slice is 256 or 768")
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.dim = dim
        self.prefix = prefix
        if tokenizer_path is None:
            tokenizer_path = _resolve("tokenizer")
            if not tokenizer_path.exists():
                ensure_mv2_files("tokenizer")
        self._sess = ort.InferenceSession(
            str(model_path),
            sess_options=_session_options(intra_op_threads),
            providers=["CPUExecutionProvider"],
        )
        self._tok = Tokenizer.from_file(str(tokenizer_path))
        pad_id = self._tok.token_to_id("<pad>")
        if pad_id is None:
            raise ValueError("tokenizer has no <pad> token — padded batches would be wrong")
        self._pad_id = int(pad_id)
        outputs = {o.name for o in self._sess.get_outputs()}
        # Prefer the token-level output by name; a pooled-first export would
        # otherwise mis-slice silently (e5_local raises instead of guessing —
        # same stance, one step earlier).
        self.output_name = next(
            (n for n in ("last_hidden_state", "token_embeddings") if n in outputs), None
        )

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed([self.prefix + t for t in texts])

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Passages take NO prefix — that is the m-v2 contract, not an oversight."""
        return self._embed(texts)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), _BATCH):
            enc = self._tok.encode_batch(texts[i : i + _BATCH])
            seq = max(min(len(e.ids), MAX_TOKENS) for e in enc)
            ids = np.full((len(enc), seq), self._pad_id, dtype=np.int64)
            mask = np.zeros((len(enc), seq), dtype=np.int64)
            for r, e in enumerate(enc):
                take = min(len(e.ids), MAX_TOKENS)
                ids[r, :take] = e.ids[:take]
                mask[r, :take] = e.attention_mask[:take]
            run_names = [self.output_name] if self.output_name else None
            last = self._sess.run(run_names, _feed(self._sess, ids, mask))[0]
            if last.ndim != 3:
                raise ValueError(
                    f"CLS pooling requires token embeddings (ndim=3); got shape {last.shape}"
                )
            emb = last[:, 0, : self.dim]  # [CLS] is token 0; padding cannot shift it
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-12)
            out.extend(emb.astype(float).tolist())
        return out
