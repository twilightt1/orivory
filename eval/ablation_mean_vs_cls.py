"""Mean-vs-CLS ablation for the arctic XS generation (ruling R43).

The P1b cutover moved the production contract from legacy masked-MEAN pooling
to the trained CLS token. The BLOCKING parity gate was P0 (the reference cosine
of the production encoder against the ONNX [CLS] slice); this file is the
EVIDENCE artifact beside it: the same corpus, the same backend, both contracts
side by side — per-item cosines and recall@k on a fixed, committed sample.

Both sides go through the PUBLIC wrappers (``arctic_embed_passages`` /
``arctic_embed_queries`` for CLS, ``arctic_embed_passages_mean`` /
``arctic_embed_queries_mean`` for the pre-P1b contract — the pair the Chroma
rollback tool also rebuilds with, ruling R31/R32), so the two contracts cannot
drift apart in tokenization, truncation or normalization. The mean side is keyed
by ``generation_name("memory", LEGACY_MEAN_FINGERPRINT)`` — the retired
generation the old binary served.

Both contracts are indexed into the SAME backend, a real embedded Qdrant
folder, one collection each (a generation is a physical collection: the cutover
is a pointer flip, not a copy). Nothing here is a gate: it exits 0 even when the
CLS side wins on every metric. ``eval/ablation_mean_vs_cls.json`` is the
artifact, committed so the numbers are reviewable without the model.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The stores this measurement uses: a throwaway embedded Qdrant folder and the
# local arctic contract. Set BEFORE any app import (settings read the env once).
# A TemporaryDirectory, not mkdtemp: the SKIP path returns before main()'s own
# cleanup, so the interpreter-exit finalizer is what takes that one with it.
_TMPDIR = tempfile.TemporaryDirectory(prefix="ablation-mean-vs-cls-",
                                     ignore_cleanup_errors=True)
_QDRANT_DIR = Path(_TMPDIR.name)
os.environ["QDRANT_MODE"] = "local"
os.environ["QDRANT_LOCAL_PATH"] = str(_QDRANT_DIR)
os.environ["USE_LOCAL_EMBEDDINGS"] = "true"
os.environ["LOCAL_EMBED_MODEL"] = "arctic"

ARTIFACT = Path(__file__).with_suffix(".json")
K_VALUES = (1, 3, 5)

# Point ids must be stable across runs and identical for both contracts (each
# contract gets its own collection, so the same id never collides).
_NAMESPACE = uuid.UUID("6f1d0f1e-2c58-4e2f-9d0e-6b0f4b7a1c11")

# ── the fixed corpus (committed with the artifact: same sample every run) ────

PASSAGES: dict[str, str] = {
    "crash-safety": "A durable intent is committed in the same transaction as "
                    "the row, so a crash before the vector write is replayed by "
                    "the drain instead of losing the index entry.",
    "generations": "A generation is a physical collection named after the "
                   "embedding contract; a cutover flips a pointer in SQL and "
                   "never copies vectors between collections.",
    "tenant-scope": "The tenant clause is always the first must condition of a "
                    "query filter, so a caller's own filter can only narrow the "
                    "search, never widen it to another owner's points.",
    "erasure": "Forgetting deletes the rows and enqueues a durable delete intent "
               "per affected id; the receipt claims completed only after a "
               "positive absence readback against the vector store.",
    "correction": "A correction supersedes the old fact and marks derived views "
                  "dirty; the superseded row stays readable in history but is "
                  "never served as context or rerank evidence again.",
    "backup": "The backup is a consistent snapshot taken with VACUUM INTO, so "
              "committed write-ahead-log frames are part of the single file the "
              "restore drill reads back.",
    "rollback": "Rolling back rebuilds the retired store from live SQL at the "
                "legacy embedding contract, stamps the old schema version and "
                "leaves the live database untouched.",
    "offline-migration": "The migration runs while the application is stopped: "
                         "an inventory, a snapshot, a resumable backfill keyed by "
                         "primary key and an audit before the pointer moves.",
    "chunk-index": "Documents are indexed as parent and child chunks in one "
                   "collection per generation, scoped by tenant and conversation "
                   "rather than by a collection per conversation.",
    "embedding-contract": "The embedding contract covers the model artifact, "
                          "pooling, the query and passage prefixes, truncation "
                          "and normalization, and it is hashed into the "
                          "generation name.",
    "vietnamese-memory": "Bộ nhớ cá nhân lưu sự kiện đã cam kết trong cơ sở dữ "
                         "liệu, chỉ mục vector là thứ phái sinh có thể kiểm "
                         "chứng lại bất cứ lúc nào.",
    "vietnamese-erasure": "Xoá dữ liệu phải xoá cả hàng trong cơ sở dữ liệu lẫn "
                          "điểm vector, và biên nhận chỉ được ghi là hoàn tất "
                          "sau khi đọc lại xác nhận không còn điểm nào.",
}

# (query, relevant passage id) — one relevant passage per query keeps the
# recall@k figure unambiguous.
QUERIES: tuple[tuple[str, str], ...] = (
    ("What happens if the process dies before the vector write lands?", "crash-safety"),
    ("How is the pointer moved when the embedding model changes?", "generations"),
    ("Can a filter reach another user's memories?", "tenant-scope"),
    ("When is an erasure receipt allowed to say completed?", "erasure"),
    ("What happens to a derived summary after a fact is corrected?", "correction"),
    ("Does the snapshot include write-ahead-log frames?", "backup"),
    ("Làm sao để chỉ mục vector được kiểm chứng lại?", "vietnamese-memory"),
    ("Khi nào biên nhận xoá dữ liệu được coi là xong?", "vietnamese-erasure"),
)


def _contracts() -> dict[str, dict]:
    from app.retrieval.embedding_fingerprint import (
        ARCTIC_CLS_FINGERPRINT,
        LEGACY_MEAN_FINGERPRINT,
        generation_name,
    )

    return {
        "cls": {
            "fingerprint": ARCTIC_CLS_FINGERPRINT,
            "generation": generation_name("memory"),
        },
        "mean": {
            "fingerprint": LEGACY_MEAN_FINGERPRINT,
            "generation": generation_name("memory", LEGACY_MEAN_FINGERPRINT),
        },
    }


def _embed(contract: str) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Passage + query vectors for one contract, through the public wrappers."""
    from app.retrieval import e5_local

    labels = list(PASSAGES)
    texts = [PASSAGES[label] for label in labels]
    if contract == "cls":
        passages = e5_local.arctic_embed_passages(texts)
        queries = e5_local.arctic_embed_queries([query for query, _ in QUERIES])
    else:
        passages = e5_local.arctic_embed_passages_mean(texts)
        queries = e5_local.arctic_embed_queries_mean([query for query, _ in QUERIES])
    return dict(zip(labels, passages, strict=True)), dict(
        zip([query for query, _ in QUERIES], queries, strict=True)
    )


def _index(contract: str, generation: str, vectors: dict[str, list[float]]) -> None:
    """Write one contract's vectors into its own real collection."""
    from qdrant_client import models as qm

    from app.retrieval import vector_backend

    dim = len(next(iter(vectors.values())))
    vector_backend.ensure_collection("memory", generation, dim)
    client = vector_backend.get_sync_client()
    client.upsert(
        collection_name=generation,
        points=[
            qm.PointStruct(
                id=str(uuid.uuid5(_NAMESPACE, label)),
                vector=[float(value) for value in vector],
                payload={"label": label},
            )
            for label, vector in vectors.items()
        ],
    )


def _rank(generation: str, vectors: dict[str, list[float]]) -> list[list[str]]:
    """Best-first labels per query, read back through the real store."""
    from app.retrieval import vector_backend

    client = vector_backend.get_sync_client()
    ranked: list[list[str]] = []
    for query, _ in QUERIES:
        hits = client.query_points(
            collection_name=generation,
            query=[float(value) for value in vectors[query]],
            limit=len(PASSAGES),
        ).points
        ranked.append([str(point.payload["label"]) for point in hits])
    return ranked


def _cosines(mean: tuple, cls: tuple) -> dict:
    """Per-item cosine(mean vector, CLS vector) for passages AND queries."""
    mean_passages, mean_queries = mean
    cls_passages, cls_queries = cls

    def _per_item(pairs: dict[str, tuple[list[float], list[float]]]) -> list[dict]:
        return [
            {"item": key,
             "cosine": float(np.dot(np.asarray(mean_vec), np.asarray(cls_vec)))}
            for key, (mean_vec, cls_vec) in pairs.items()
        ]

    def _summary(items: list[dict]) -> dict:
        values = np.asarray([item["cosine"] for item in items], dtype=float)
        return {
            "mean": float(values.mean()) if values.size else None,
            "min": float(values.min()) if values.size else None,
            "max": float(values.max()) if values.size else None,
        }

    passages = _per_item({label: (mean_passages[label], cls_passages[label])
                          for label in PASSAGES})
    queries = _per_item({query: (mean_queries[query], cls_queries[query])
                         for query, _ in QUERIES})
    return {
        "passages": passages,
        "queries": queries,
        "summary": {"passages": _summary(passages), "queries": _summary(queries),
                    "all_items": _summary(passages + queries)},
    }


def _recall(ranked: list[list[str]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in K_VALUES:
        hits = sum(
            1 for ranking, (_, relevant) in zip(ranked, QUERIES, strict=True)
            if relevant in ranking[:k]
        )
        metrics[f"recall@{k}"] = round(hits / len(QUERIES), 4)
    return metrics


def main() -> int:
    """Measure both contracts; write the artifact. Never a gate."""
    from app.retrieval import e5_local, vector_backend

    if not e5_local.arctic_files_cached():
        print("SKIP: arctic ONNX cache missing — no model download attempted")
        return 0

    contracts = _contracts()
    folder = Path(os.environ["QDRANT_LOCAL_PATH"])
    try:
        embeddings = {name: _embed(name) for name in contracts}
        for name, contract in contracts.items():
            _index(name, contract["generation"], embeddings[name][0])
        ranked = {
            name: _rank(contract["generation"], embeddings[name][1])
            for name, contract in contracts.items()
        }
        recall = {name: _recall(ranking) for name, ranking in ranked.items()}
        cosines = _cosines(embeddings["mean"], embeddings["cls"])
    finally:
        # Release the embedded folder lock and take the temp store with us.
        import asyncio

        asyncio.run(vector_backend.close_clients())
        shutil.rmtree(folder, ignore_errors=True)

    artifact = {
        "generated_at": datetime.now(UTC).isoformat(),
        "note": ("evidence, not a blocking gate — the blocking mean-vs-CLS parity "
                 "gate is the P0 corpus baseline"),
        "backend": "qdrant local (embedded, in-process)",
        "corpus": {
            "passages": len(PASSAGES),
            "queries": len(QUERIES),
            "labels": sorted(PASSAGES),
            "queries_text": [query for query, _ in QUERIES],
        },
        "contracts": {
            name: {
                "generation": contract["generation"],
                "fingerprint": contract["fingerprint"],
                "pooling": contract["fingerprint"].get("pooling"),
                "dim": contract["fingerprint"].get("dim"),
            }
            for name, contract in contracts.items()
        },
        "k": list(K_VALUES),
        "recall": recall,
        "delta_cls_minus_mean": {
            key: round(recall["cls"][key] - recall["mean"][key], 4) for key in recall["cls"]
        },
        "cosines": cosines,
    }
    ARTIFACT.write_text(json.dumps(artifact, indent=2) + "\n")

    print(f"Mean-vs-CLS ablation — {len(PASSAGES)} passages, {len(QUERIES)} queries")
    print(f"  cls  generation: {contracts['cls']['generation']}")
    print(f"  mean generation: {contracts['mean']['generation']}")
    print("  cosine(cls, mean)  passages: "
          f"{cosines['summary']['passages']}  queries: {cosines['summary']['queries']}")
    for key in recall["cls"]:
        print(f"  {key:<10} cls={recall['cls'][key]:<8} mean={recall['mean'][key]:<8} "
              f"delta={artifact['delta_cls_minus_mean'][key]:+.4f}")
    print(f"artifact: {ARTIFACT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
