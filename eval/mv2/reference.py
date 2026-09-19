"""FP reference arm — the OFFICIAL arctic-embed-m-v2 modeling code, pinned, reviewed, imported explicitly.

What this module runs
---------------------
The D3 ablation's reference arm must be the code Snowflake/Alibaba ship, not a
reimplementation. This module downloads the two Python files published at the
pinned revision into a cache OUTSIDE the repo, verifies their sha256, and
imports them EXPLICITLY — the classes are registered with transformers by this
module itself. Nothing here ever passes ``trust_remote_code=True``: transformers
is never allowed to fetch or execute code on its own (plan global constraint:
"Không bật remote custom code trong runtime"; this script is experiment-only,
pin + review before running).

Review of ``configuration_hf_alibaba_nlp_gte.py`` (144 lines, sha256 pinned below)
----------------------------------------------------------------------------------
- Defines exactly one class: ``GteConfig(PretrainedConfig)``, ``model_type =
  "gte"``; it declares the GTE-v2 switches (RoPE theta, ``pack_qkv``,
  ``unpad_inputs``, ``use_memory_efficient_attention``, ``logn_attention_*``,
  ``layer_norm_type``) and calls ``super().__init__(**kwargs)`` — unknown keys
  (``matryoshka_dimensions``) ride through PretrainedConfig.
- Imports only ``transformers.configuration_utils``/``utils.logging``: no
  network, no filesystem access, no torch, no side effects at import time.
- Nothing unexpected beyond cosmetic leftovers: the docstring is unedited
  template text for ``NewConfig``/``izhx/new-base-en`` and ``pad_token_id`` is
  commented out of the signature (``config.json`` supplies 1).
- The sibling ``modeling_hf_alibaba_nlp_gte.py`` (also pinned here, sha256
  below) is what defines the network; importing it requires the two files to
  sit in one package directory (it does ``from .configuration_hf_alibaba_nlp_gte
  import GteConfig``), so :func:`ensure_reference_code` writes the package
  ``__init__.py`` itself.

Two recorded deviations on this box (both inference-efficiency switches in the
published config, empirically required to run at all — see
:attr:`ReferenceEmbedder.deviations`, populated at load time, and the
"FP reference deviations" note in ``eval/ablation_retrieval_mv2.py``)
-------------------------------------------------------------------------------
- ``config.json`` ships ``"unpad_inputs": "true"`` and
  ``"use_memory_efficient_attention": "true"`` as STRINGS (truthy). The
  memory-efficient path asserts xformers, which has no macOS build, and the
  unpad path without xformers feeds an unpadded sequence into a padded
  attention mask. Both are forced False here: the padded SDPA path, which is
  what the ONNX export and every other arm compute.
- Everything else follows the model card's official snippet: CLS = position 0
  of ``last_hidden_state``, ``query: `` prefix on queries only, L2-normalize,
  no pooling layer.

Experiment-only dependencies
----------------------------
``torch`` (CPU) + ``transformers`` are installed in the worktree venv ONLY
(``uv pip install -p .venv torch transformers``); CI never installs them — the
repo's own test suite does not import this module, and the ablation imports it
lazily inside the reference arm, so a missing torch/transformers is a recorded
arm failure, never a collection error. Nothing in ``app/`` imports this module.
Every import is lazy so importing :mod:`eval.ablation_retrieval_mv2` stays
torch-free; the ablation catches a failed reference loading and records it as a
documented arm failure instead of fabricating numbers.

``transformers`` is pinned BELOW v5 for this arm: the official modeling file
calls ``PreTrainedModel.get_extended_attention_mask``, which transformers 5
removed (measured — v5.17 raised ``AttributeError`` on the first forward pass),
and the config was authored against the 4.x API (``transformers_version:
4.39.3`` in both config.json and config_sentence_transformers.json). The
installed versions are recorded in the artifact's fingerprint.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

from eval.mv2 import runner as mv2

ROOT = Path(__file__).resolve().parents[2]

CONFIG_FILE = "configuration_hf_alibaba_nlp_gte.py"
MODELING_FILE = "modeling_hf_alibaba_nlp_gte.py"
CONFIG_SHA256 = "d31622918ecf738d1bb082bb27fe8141c3189238cef51d464c7f68e32bbe3d54"
MODELING_SHA256 = "f8dff057e7c91790771f24ac8b8c172ab5c863835194872539c74fa5f538ed4c"
WEIGHTS_FILE = "model.safetensors"
WEIGHTS_SHA256 = "3d80d4727ac8759fb8624b690697c053a3d1992120111dc4a71178e608c26604"
# Package name for the downloaded files: the official modeling file does a
# RELATIVE import of its config sibling, so both must live in one importable
# package. The directory name is the package name.
PACKAGE = "reference"
MODELING_STEM = MODELING_FILE[: -len(".py")]

DEFAULT_BATCH = 16
MAX_TOKENS = mv2.MAX_TOKENS  # same 512-token cap as the ONNX arms (8192 not enabled)


def reference_dir() -> Path:
    """``~/.cache/orivory/mv2/reference`` — outside the repo, like every model cache."""
    return mv2.model_dir() / PACKAGE


def hf_cache_dir() -> Path:
    """HF hub cache for the reference weights (revision-pinned, outside the repo)."""
    return reference_dir() / "hf" / "hub"


def ensure_reference_code() -> dict[str, Path]:
    """Download-verify (or verify cached) the two pinned official files."""
    out: dict[str, Path] = {}
    directory = reference_dir()
    directory.mkdir(parents=True, exist_ok=True)
    # The initializer runs on import — keep it EMPTY (a foreign file here would
    # execute unreviewed code under the pinned-code guarantee).
    (directory / "__init__.py").write_text("", encoding="utf-8")
    for name, sha256 in ((CONFIG_FILE, CONFIG_SHA256), (MODELING_FILE, MODELING_SHA256)):
        path = directory / name
        if not path.exists() or mv2._digest(path) != sha256:
            path.unlink(missing_ok=True)
            mv2._download(f"{mv2._BASE_URL}/{name}", path, sha256)
        out[name] = path
    return out


def pinned_classes():
    """Import the pinned modules explicitly and hand back ``(GteConfig, GteModel)``.

    The parent of the cache package is put on ``sys.path`` so the package's own
    relative import works; the modules are then importable by name without any
    transformers remote-code machinery.
    """
    ensure_reference_code()
    parent = str(reference_dir().parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    module = importlib.import_module(f"{PACKAGE}.{MODELING_STEM}")
    expected = (reference_dir() / MODELING_FILE).resolve()
    module_file = module.__file__
    if module_file is None or Path(module_file).resolve() != expected:
        raise RuntimeError(
            f"imported {PACKAGE}.{MODELING_STEM} resolved to {module_file!r}, not the "
            f"verified cache file {expected} — refusing to run unpinned code"
        )
    return module.GteConfig, module.GteModel


class ReferenceEmbedder:
    """Float32 CPU reference: padded batches, CLS pooling, optional MRL slice, L2.

    Call shape matches ``e5_local.arctic_embed_queries/passages`` and
    :class:`eval.mv2.runner.Mv2Onnx` (``list[list[float]]``) so the ablation can
    swap arms without a shim.
    """

    def __init__(
        self,
        dim: int = 768,
        prefix: str = mv2.QUERY_PREFIX,
        batch_size: int = DEFAULT_BATCH,
        tokenizer_path: Path | None = None,
    ) -> None:
        if dim not in (256, 768):
            raise ValueError(f"unsupported dim {dim!r} — MRL slice is 256 or 768")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        import torch
        import transformers
        from tokenizers import Tokenizer
        from transformers import AutoConfig, AutoModel

        if int(transformers.__version__.split(".", 1)[0]) >= 5:
            raise RuntimeError(
                "ReferenceEmbedder requires transformers<5 — the pinned modeling file calls "
                f"get_extended_attention_mask (removed in v5); found {transformers.__version__}"
            )

        GteConfig, GteModel = pinned_classes()
        AutoConfig.register(GteConfig.model_type, GteConfig, exist_ok=True)
        AutoModel.register(GteConfig, GteModel, exist_ok=True)

        config = AutoConfig.from_pretrained(
            mv2.HF_REPO, revision=mv2.HF_REVISION, cache_dir=str(hf_cache_dir())
        )
        # The published flags are the string "true"; both drive xformers-only
        # paths (see the module docstring) — force the padded SDPA path.
        config.unpad_inputs = False
        config.use_memory_efficient_attention = False
        self.config = config
        self.deviations = {
            "unpad_inputs": 'published as string "true"; forced False (xformers-only path)',
            "use_memory_efficient_attention": 'published as string "true"; forced False (xformers-only path)',
        }
        self.model = AutoModel.from_pretrained(
            mv2.HF_REPO,
            revision=mv2.HF_REVISION,
            config=config,
            add_pooling_layer=False,
            torch_dtype=torch.float32,
            cache_dir=str(hf_cache_dir()),
        )
        self.model.eval()
        self._torch = torch
        self.dim = dim
        self.prefix = prefix
        self.batch_size = batch_size
        if tokenizer_path is None:
            tokenizer_path = mv2.ensure_mv2_files("tokenizer")["tokenizer"]
        self._tok = Tokenizer.from_file(str(tokenizer_path))
        pad_id = self._tok.token_to_id("<pad>")
        if pad_id is None:
            raise ValueError("tokenizer has no <pad> token — padded batches would be wrong")
        self._pad_id = int(pad_id)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed([self.prefix + t for t in texts])

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Passages take NO prefix — the m-v2 contract, not an oversight."""
        return self._embed(texts)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        torch = self._torch
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            enc = self._tok.encode_batch(chunk)
            seq = max(min(len(e.ids), MAX_TOKENS) for e in enc)
            ids = np.full((len(enc), seq), self._pad_id, dtype=np.int64)
            mask = np.zeros((len(enc), seq), dtype=np.int64)
            for r, e in enumerate(enc):
                take = min(len(e.ids), MAX_TOKENS)
                ids[r, :take] = e.ids[:take]
                mask[r, :take] = e.attention_mask[:take]
            with torch.no_grad():
                last = self.model(
                    input_ids=torch.from_numpy(ids), attention_mask=torch.from_numpy(mask)
                ).last_hidden_state
            emb = last[:, 0, : self.dim].to(torch.float32).numpy()  # CLS is token 0
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-12)
            out.extend(emb.astype(float).tolist())
        return out


def verify_loaded_weights() -> None:
    """Fail the arm when the cached checkpoint bytes are not the pinned ones.

    ``artifact_facts()`` records the digest; it never rejects a mismatch. The
    signed verdict must only ever rest on the reviewed weights.
    """
    facts = artifact_facts()
    sha = facts["weights"]["sha256"]
    if sha != WEIGHTS_SHA256:
        raise RuntimeError(
            f"reference weights digest {sha!r} != pinned {WEIGHTS_SHA256!r} — "
            "the FP reference arm would not be the reviewed artifact"
        )


def artifact_facts() -> dict:
    """Fingerprint facts for the artifact: files, digests, versions. Never raises."""
    facts: dict = {
        "repo": mv2.HF_REPO,
        "revision": mv2.HF_REVISION,
        "weights": {"file": WEIGHTS_FILE, "sha256": None, "path": None},
        "code": {CONFIG_FILE: CONFIG_SHA256, MODELING_FILE: MODELING_SHA256},
        "torch": None,
        "transformers": None,
    }
    try:
        facts["torch"] = importlib.import_module("torch").__version__
        facts["transformers"] = importlib.import_module("transformers").__version__
    except ImportError:
        pass
    weights = sorted(hf_cache_dir().glob(f"models--*/snapshots/{mv2.HF_REVISION}/{WEIGHTS_FILE}"))
    if weights:
        path = weights[0]
        facts["weights"] = {
            "file": WEIGHTS_FILE,
            "path": str(path),
            "sha256": mv2._digest(path),
            "size_bytes": path.stat().st_size,
        }
    return facts
