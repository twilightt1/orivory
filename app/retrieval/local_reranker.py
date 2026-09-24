"""Local GTE multilingual reranker on onnxruntime (no torch, no paid API).

``Alibaba-NLP/gte-multilingual-reranker-base`` (Apache-2.0) as the int8 export
published by ``onnx-community`` — the licence-clean counterpart of the Jina
lane (the hosted model, ``jina-reranker-v2-base-multilingual``, is CC-BY-NC).
Measured on the dev Mac, int8, batch=1: 135 ms/pair at 512 tokens and 324 ms
at 1024, against 137/351 ms for the Jina model's own int8 export — same cost
class, 341 MB on disk (~1.1 GB resident once ORT has the session).

Scoring is sequence classification: one forward pass per (query, document
window) pair, higher logit = more relevant. Long memories are split into
overlapping *token* windows and max-pooled, because the ONNX graph truncates
silently where the hosted API runs a sliding window server-side — a 4k-char
session chunk would otherwise lose its tail.

Files live next to the embedding models (``LOCAL_E5_DIR``, shared with
``e5_local`` — one directory, one lazy-download story) and are verified by
SHA256 before use. Nothing warms this at boot: the rerank is opt-in, so the
first call pays the session build.

ponytail: window count is capped (see ``_MAX_WINDOWS``) — a very long memory
has its tail under-scored instead of paying 4x the latency; raise the cap if a
gate shows long-memory recall actually moving.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.retrieval import e5_local  # model dir + download/session helpers

log = logging.getLogger(__name__)

_HF = "https://huggingface.co/onnx-community/gte-multilingual-reranker-base/resolve"
# Pinned revision: the digests below are of the bytes served at this commit.
REVISION = "ee64367e35a2db0da46bb6497e13a18f8bd585cb"
MODEL_URL = f"{_HF}/{REVISION}/onnx/model_int8.onnx"
TOKENIZER_URL = f"{_HF}/{REVISION}/tokenizer.json"
MODEL_FILE = "gte_reranker_int8.onnx"
TOKENIZER_FILE = "gte_reranker_tokenizer.json"
MODEL_SHA256 = "ccf51dba7f8aa9205753761cfaa68c55f741792501463a3bf25d7e5bcdac7c35"
TOKENIZER_SHA256 = "3ffb37461c391f096759f4a9bbbc329da0f36952f88bab061fcf84940c022e98"

# Model card ceiling is 8192 tokens; past 1024 the measured cost per pair
# (324 ms) is already the dominant term of a recall call, so a pair is scored
# in windows of this size and the windows overlap by _WINDOW_STRIDE's slack.
_MAX_TOKENS = 1024
_WINDOW_STRIDE = 768
_MAX_WINDOWS = 4

_sess = None
_tok = None


def model_dir() -> Path:
    return e5_local.model_dir()


def files_cached() -> bool:
    d = model_dir()
    return (d / MODEL_FILE).exists() and (d / TOKENIZER_FILE).exists()


def ensure_files() -> tuple[Path, Path]:
    """Fetch both files once, digest-verified (shared with the embedder).

    Same contract as ``e5_local.ensure_files``: a stale or partial file never
    reaches the session — a bad digest deletes and re-downloads it.
    """
    d = model_dir()
    d.mkdir(parents=True, exist_ok=True)
    model, tok = d / MODEL_FILE, d / TOKENIZER_FILE
    if not model.exists() or e5_local._digest(model) != MODEL_SHA256:
        model.unlink(missing_ok=True)
        e5_local._download(MODEL_URL, model, MODEL_SHA256)
    if not tok.exists() or e5_local._digest(tok) != TOKENIZER_SHA256:
        tok.unlink(missing_ok=True)
        e5_local._download(TOKENIZER_URL, tok, TOKENIZER_SHA256)
    return model, tok


def _session():
    global _sess
    if _sess is None:
        with e5_local._init_lock:
            if _sess is None:
                import onnxruntime as ort

                model, _ = ensure_files()
                _sess = ort.InferenceSession(
                    str(model), sess_options=e5_local._session_options(),
                    providers=["CPUExecutionProvider"],
                )
    return _sess


def _tokenizer():
    global _tok
    if _tok is None:
        with e5_local._init_lock:
            if _tok is None:
                from tokenizers import Tokenizer

                _, tok_path = ensure_files()
                # No truncation on the tokenizer itself: the document's FULL
                # ids are what _windows slices, and every pair is cut to
                # _MAX_TOKENS by hand (same discipline as e5_local._encode_with).
                _tok = Tokenizer.from_file(str(tok_path))
    return _tok


def _windows(tok, text: str, room: int) -> list[str]:
    """Split ``text`` into windows of at most ``room`` tokens.

    A text that fits is returned untouched (no decode round-trip). A longer one
    is sliced on token boundaries and decoded back to text so the pair template
    (special tokens, segment ids) stays the tokenizer's business, not ours.
    """
    ids = tok.encode(text).ids
    if len(ids) <= room:
        return [text]
    step = min(_WINDOW_STRIDE, room)
    return [tok.decode(ids[i : i + room]) for i in range(0, len(ids), step)][:_MAX_WINDOWS]


def _score_pair(sess, tok, query: str, doc: str) -> float:
    q_ids = tok.encode(query).ids
    best = None
    # The query rides in the same sequence as the window, special tokens
    # included; the slack keeps the window from pushing the query out.
    room = max(64, _MAX_TOKENS - len(q_ids) - 8)
    for window in _windows(tok, doc, room):
        enc = tok.encode(query, window)
        ids = np.asarray([enc.ids[:_MAX_TOKENS]], dtype=np.int64)
        mask = np.asarray([enc.attention_mask[:_MAX_TOKENS]], dtype=np.int64)
        logits = np.asarray(sess.run(None, e5_local._feed(sess, ids, mask))[0])
        # [1, 1] for this export; a 2-column head would make column 0 the
        # "irrelevant" logit, so read the LAST column rather than [0].
        score = float(logits.reshape(1, -1)[0, -1])
        best = score if best is None else max(best, score)
    return best if best is not None else 0.0


def _score_sync(query: str, docs: list[str]) -> list[float]:
    sess, tok = _session(), _tokenizer()
    return [_score_pair(sess, tok, query, doc) for doc in docs]


async def score_pairs(query: str, docs: list[str]) -> list[float]:
    """Score every doc against ``query``, best-relevance first is the caller's job.

    Runs on the embed executor (``e5_local``/``embedder._run_off_loop``): a
    synchronous forward pass per pair would stall the event loop for the whole
    pool's duration, and this sits on the recall path.
    """
    if not docs:
        return []
    from app.retrieval.embedder import _run_off_loop

    scores = await _run_off_loop(_score_sync, query, docs)
    if len(scores) != len(docs):
        raise ValueError(f"local reranker scored {len(scores)} of {len(docs)} documents")
    return scores
