"""Local multilingual-e5-small embeddings on onnxruntime (no torch, no API).

 differentially from chroma's bundled MiniLM: e5 requires input prefixes —
``query: `` for queries, ``passage: `` for indexed texts (model card,
verbatim). Forgetting the passage side degrades retrieval, so both entry
points below apply their prefix and there is no unprefixed path.

Files (Xenova quantized, portable ARM+x86 — NOT the official avx512-only
int8): model_quantized.onnx (118MB) + tokenizer.json (17MB), lazy-downloaded
once into LOCAL_E5_DIR (default ~/.cache/orivory/e5, /data/models/e5 in the
lite image). 384-dim: drop-in for MiniLM collections dimension-wise, but a
*different* backend name (``local-e5``) so the dim guard fails loud instead
of silently mixing MiniLM and e5 vectors.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MODEL_URL = (
    "https://huggingface.co/Xenova/multilingual-e5-small"
    "/resolve/761b726dd34fb83930e26aab4e9ac3899aa1fa78/onnx/model_quantized.onnx"
)
TOKENIZER_URL = (
    "https://huggingface.co/Xenova/multilingual-e5-small"
    "/resolve/761b726dd34fb83930e26aab4e9ac3899aa1fa78/tokenizer.json"
)
MODEL_FILE = "model_quantized.onnx"
TOKENIZER_FILE = "tokenizer.json"
MODEL_SHA256 = "f80102d3f2a1229f387d3c81909990d8945513e347b0eab049f7de3c6f98c193"
TOKENIZER_SHA256 = "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"

ARCTIC_MODEL_URL = (
    "https://huggingface.co/Snowflake/snowflake-arctic-embed-xs"
    "/resolve/d8c86521100d3556476a063fc2342036d45c106f/onnx/model.onnx"
)
ARCTIC_TOKENIZER_URL = (
    "https://huggingface.co/Snowflake/snowflake-arctic-embed-xs"
    "/resolve/d8c86521100d3556476a063fc2342036d45c106f/tokenizer.json"
)
ARCTIC_MODEL_FILE = "arctic_model.onnx"
ARCTIC_TOKENIZER_FILE = "arctic_tokenizer.json"
ARCTIC_MODEL_SHA256 = "cf2698d30ff05da02c70a088313bad56e5c2f401d734cb24a8390d446111936c"
ARCTIC_TOKENIZER_SHA256 = "91f1def9b9391fdabe028cd3f3fcc4efd34e5d1f08c3bf2de513ebb5911a1854"
ARCTIC_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

_BATCH = 128
_MAX_TOKENS = 512

_sess = None
_tok = None
_asess = None
_atok = None


def model_dir() -> Path:
    from app.config import settings

    configured = (settings.LOCAL_E5_DIR or "").strip()
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "orivory" / "e5"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, dest: Path, expected_sha256: str) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.warning("Downloading embedding model %s (~%s)", dest.name, url)
    urllib.request.urlretrieve(url, tmp)
    if _digest(tmp) != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"Downloaded {dest.name} failed SHA256 verification")
    tmp.rename(dest)


def ensure_files() -> tuple[Path, Path]:
    d = model_dir()
    d.mkdir(parents=True, exist_ok=True)
    model, tok = d / MODEL_FILE, d / TOKENIZER_FILE
    if not model.exists() or _digest(model) != MODEL_SHA256:
        model.unlink(missing_ok=True)
        _download(MODEL_URL, model, MODEL_SHA256)
    if not tok.exists() or _digest(tok) != TOKENIZER_SHA256:
        tok.unlink(missing_ok=True)
        _download(TOKENIZER_URL, tok, TOKENIZER_SHA256)
    return model, tok


def _session():
    global _sess
    if _sess is None:
        import onnxruntime as ort

        model, _ = ensure_files()
        _sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    return _sess


def _tokenizer():
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer

        _, tok_path = ensure_files()
        _tok = Tokenizer.from_file(str(tok_path))
    return _tok


def _feed(sess, ids, mask) -> dict:
    """Build the input feed from the session's own input names.

    Xenova e5 wants token_type_ids in addition to input_ids/attention_mask;
    other exports don't. All-zero segment ids are correct for single texts.
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


def _encode(texts: list[str]) -> list[list[float]]:
    return _encode_with(texts, _session, _tokenizer)


def arctic_files_cached() -> bool:
    d = model_dir()
    return (d / ARCTIC_MODEL_FILE).exists() and (d / ARCTIC_TOKENIZER_FILE).exists()


def ensure_arctic_files() -> tuple[Path, Path]:
    d = model_dir()
    d.mkdir(parents=True, exist_ok=True)
    model, tok = d / ARCTIC_MODEL_FILE, d / ARCTIC_TOKENIZER_FILE
    if not model.exists() or _digest(model) != ARCTIC_MODEL_SHA256:
        model.unlink(missing_ok=True)
        _download(ARCTIC_MODEL_URL, model, ARCTIC_MODEL_SHA256)
    if not tok.exists() or _digest(tok) != ARCTIC_TOKENIZER_SHA256:
        tok.unlink(missing_ok=True)
        _download(ARCTIC_TOKENIZER_URL, tok, ARCTIC_TOKENIZER_SHA256)
    return model, tok


def _asession():
    global _asess
    if _asess is None:
        import onnxruntime as ort

        model, _ = ensure_arctic_files()
        _asess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    return _asess


def _atokenizer():
    global _atok
    if _atok is None:
        from tokenizers import Tokenizer

        _, tok_path = ensure_arctic_files()
        _atok = Tokenizer.from_file(str(tok_path))
    return _atok


def _encode_with(
    texts: list[str], sess_fn, tok_fn, pooling: str = "mean"
) -> list[list[float]]:
    """Tokenize → batch-pad to the longest truncated sequence → pool → L2.

    ``pooling`` is the only contract difference between the two local models
    and it is part of the embedding fingerprint: e5 has no CLS convention
    (masked mean over non-pad tokens — the sentence-transformers default),
    while arctic-embed-xs is trained for CLS (``last_hidden_state[:, 0, :]``,
    the reference form pinned by tests/retrieval/test_xs_parity.py). A typo
    must not silently fall back to the other contract, so it raises.
    """
    if pooling not in {"mean", "cls"}:
        raise ValueError(f"unknown pooling {pooling!r} — expected 'mean' or 'cls'")
    sess, tok = sess_fn(), tok_fn()
    out: list[list[float]] = []
    for i in range(0, len(texts), _BATCH):
        enc = tok.encode_batch(texts[i : i + _BATCH])
        seq = max(min(len(e.ids), _MAX_TOKENS) for e in enc)
        ids = np.zeros((len(enc), seq), dtype=np.int64)
        mask = np.zeros((len(enc), seq), dtype=np.int64)
        for r, e in enumerate(enc):
            take = min(len(e.ids), _MAX_TOKENS)
            ids[r, :take] = e.ids[:take]
            mask[r, :take] = e.attention_mask[:take]
        last = sess.run(None, _feed(sess, ids, mask))[0]
        if pooling == "cls":
            if last.ndim != 3:
                # A 2-D export (already pooled) would mis-slice silently.
                raise ValueError(
                    f"CLS pooling requires token embeddings (ndim=3); got shape {last.shape}"
                )
            # [CLS] is token 0 for both models; padding cannot shift it.
            emb = last[:, 0, :]
        else:
            summed = (last * mask[..., None]).sum(axis=1)
            counts = mask.sum(axis=1, keepdims=True).clip(min=1)
            emb = summed / counts
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-12)
        out.extend(emb.astype(float).tolist())
    return out


def embed_queries(texts: list[str]) -> list[list[float]]:
    return _encode([QUERY_PREFIX + t for t in texts])


def embed_passages(texts: list[str]) -> list[list[float]]:
    return _encode([PASSAGE_PREFIX + t for t in texts])


def arctic_embed_queries(texts: list[str]) -> list[list[float]]:
    return _encode_with(
        [ARCTIC_QUERY_PREFIX + t for t in texts], _asession, _atokenizer, pooling="cls"
    )


def arctic_embed_passages(texts: list[str]) -> list[list[float]]:
    return _encode_with(texts, _asession, _atokenizer, pooling="cls")
