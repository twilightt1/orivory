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

import logging
import urllib.request
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MODEL_URL = (
    "https://huggingface.co/Xenova/multilingual-e5-small"
    "/resolve/main/onnx/model_quantized.onnx"
)
TOKENIZER_URL = (
    "https://huggingface.co/Xenova/multilingual-e5-small"
    "/resolve/main/tokenizer.json"
)
MODEL_FILE = "model_quantized.onnx"
TOKENIZER_FILE = "tokenizer.json"

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

_BATCH = 128
_MAX_TOKENS = 512

_sess = None
_tok = None


def model_dir() -> Path:
    from app.config import settings

    configured = (settings.LOCAL_E5_DIR or "").strip()
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "orivory" / "e5"


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.warning("Downloading embedding model %s (~%s)", dest.name, url)
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def ensure_files() -> tuple[Path, Path]:
    d = model_dir()
    d.mkdir(parents=True, exist_ok=True)
    model, tok = d / MODEL_FILE, d / TOKENIZER_FILE
    if not model.exists():
        _download(MODEL_URL, model)
    if not tok.exists():
        _download(TOKENIZER_URL, tok)
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
    sess, tok = _session(), _tokenizer()
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
        # ponytail: mean-pool, not CLS — e5 has no CLS pooling convention;
        # mean over non-pad tokens is the sentence-transformers default.
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
