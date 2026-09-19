#!/usr/bin/env python
"""Filtered recall vs the exact subset: what a payload filter does to ANN recall.

Two arms, both measured against a store built by ``run_10k.py --rows 100000``:

1. **LOCAL (always).** The embedded Qdrant is an EXACT store (brute force): its
   filtered search cannot be graded on ANN quality — there is no ANN. So the
   local arm is a METHODOLOGY check measured anyway, because the number is
   load-bearing: the app's own filtered search (``search_memories`` through
   ``build_filter``) must equal a client-side cosine scan over the same filtered
   subset — same ids, same order up to ties. Recall@{1,5,10} = 1.0 here means
   the filter translation, the scoring and THIS harness's recall math are sound;
   anything less is a bug this harness just found, never a tolerance. It also
   records the real local scan latency per selectivity bucket (the 100K
   milestone's own scan cost, useful evidence in its own right). What it does
   NOT prove: any ANN quality — local mode has no ANN to grade.

2. **SERVER (when ``--server-url`` is given).** The same points are copied into
   a Qdrant SERVER collection (HNSW, default params, payload indexes for the
   filtered fields) and the same queries are re-run against it: recall@{1,5,10}
   vs the exact subset, per selectivity bucket. This is the REAL filtered-ANN
   number. If no server is reachable the report records ``not-run: <reason>`` —
   never a fabricated number.

Selectivity buckets
-------------------
``captured_at`` windows ending at the LAST captured_at of the run, sized to
100% / 50% / 10% / 1% / 0.1% of the run's time span. This is the only varied
payload field this workload has: P4a has ONE namespace per user and a
single-user local run, so tenant/namespace selectivity cannot be varied here —
recorded as such, not silently substituted. The subset sizes are computed two
ways (the store's own count-with-filter vs a client-side payload predicate) and
both are recorded; a disagreement is a filter-translation bug, not a tolerance.

Run::

    .venv/bin/python eval/scale/filtered_ann.py --workdir /tmp/orivory-scale-100k
    .venv/bin/python eval/scale/filtered_ann.py --workdir <dir> --server-url http://127.0.0.1:6333
    .venv/bin/python eval/scale/filtered_ann.py --self-check   # tiny local fixture
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.scale.gen_corpus import SEED, load_corpus  # noqa: E402
from eval.scale.run_10k import (  # noqa: E402
    SCALE_USER,
    build_queries,
    configure_environment,
    git_state,
    percentile,
    rss_block,
    sha256_file,
)

ARTIFACTS_DIR = ROOT / "eval" / "scale" / "artifacts"
SELECTIVITY_FRACTIONS = (1.0, 0.5, 0.1, 0.01, 0.001)
K_VALUES = (1, 5, 10)
POINT_BATCH = 1000


# ── pure helpers (the harness's math — unit-tested) ─────────────────────────


def recall_at_k(ann_ids: list[str], exact_ids: list[str], k: int) -> float:
    """|ann[:k] ∩ exact[:k]| / k over ORDERED id lists. ``k <= 0`` is 0.0.

    Set intersection, so equal-score boundary ties cannot fake a mismatch; the
    lists are truncated to k first, so a longer ANN list adds nothing.
    """
    if k <= 0:
        return 0.0
    return len(set(ann_ids[:k]) & set(exact_ids[:k])) / k


def exact_top_k(vectors: np.ndarray, query: np.ndarray, k: int, ids: list[str]) -> list[str]:
    """Client-side cosine top-k over ``vectors`` — the ground truth.

    Rows/queries are L2-normalized HERE (defensively): cosine ranking must not
    depend on an upstream normalization the store might not have applied, and a
    zero vector is skipped rather than scored as NaN. Ties break on id, the
    same order the app's own search sorts by.
    """
    if k <= 0 or len(ids) == 0:
        return []
    rows = np.asarray(vectors, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    rows = rows / np.where(norms == 0.0, 1.0, norms)
    query_norm = float(np.linalg.norm(query))
    if query_norm == 0.0 or not math.isfinite(query_norm):
        return []
    scores = rows @ (query / query_norm)
    order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), str(ids[i])))
    return [str(ids[i]) for i in order[:k]]


def window_start(last_captured_at: datetime, span_seconds: float, fraction: float) -> datetime:
    """Start of the ``captured_at`` window covering ``fraction`` of the span.

    The window always ends at the run's LAST captured_at, so every bucket is
    anchored to measured data rather than to a clock. A fraction outside
    (0, 1] is refused: a 0-width or inverted window would silently select
    everything (or nothing) and the bucket's label would be a lie.

    The result is tz-aware UTC: payload stamps read back through SQLite are
    NAIVE (the column drops the offset — measured on this store), Qdrant parses
    offset-less datetime payloads as UTC, and the app's filter builder REFUSES
    a naive bound on purpose. Normalizing here keeps both sides of the
    comparison in the same zone instead of silently shifting the window.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"selectivity fraction must be within (0, 1], got {fraction}")
    if last_captured_at.tzinfo is None:
        last_captured_at = last_captured_at.replace(tzinfo=UTC)
    return last_captured_at - timedelta(seconds=float(span_seconds) * fraction)


def bucket_where(start: datetime | None) -> dict | None:
    """The caller-facing filter for a bucket: ``None`` when it is the full set."""
    if start is None:
        return None
    return {"captured_at": {"$gte": start.isoformat()}}


# ── the store side ──────────────────────────────────────────────────────────


def _parse_utc(raw: str | None) -> datetime | None:
    """An ISO stamp read as UTC; naive values (SQLite drops the offset) get UTC.

    Returns ``None`` for a missing/unparseable value — "unknown", never a guess.
    """
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


def _captured_at_predicate(start_iso: str | None):
    """The client-side twin of ``captured_at >= start`` (ISO strings, UTC).

    Payload stamps are NAIVE (SQLite drops the offset on read) while the bound
    is tz-aware: naive values are read as UTC on BOTH sides, or the comparison
    would be absent-vs-aware and crash instead of answering.
    """
    def predicate(payload: dict) -> bool:
        if start_iso is None:
            return True
        stamp = _parse_utc(payload.get("captured_at"))
        return stamp is not None and stamp >= _parse_utc(start_iso)
    return predicate


def _scroll_all(local_client, generation: str) -> dict:
    """ONE unfiltered pass: every id, vector and captured_at in the generation.

    One pass, not one per selectivity bucket: the embedded store's filtered
    scroll re-evaluates the filter over the store for EVERY page, so five
    buckets meant five full scans of the same 100K rows (measured: the first
    pass through all five took >10 CPU-minutes before the second bucket
    finished). The per-bucket filter is applied HERE instead, and the store's
    own count-with-filter still verifies the translation on the store side.
    """
    ids: list[str] = []
    blocks: list[np.ndarray] = []
    stamps: list[datetime | None] = []
    offset = None
    while True:
        records, offset = local_client.scroll(generation, limit=POINT_BATCH, offset=offset,
                                              with_payload=True, with_vectors=True)
        page = []
        for record in records:
            payload = dict(record.payload or {})
            ids.append(str(record.id))
            stamps.append(_parse_utc(payload.get("captured_at")))
            page.append(record.vector)
        if page:
            blocks.append(np.asarray(page, dtype=np.float32))
        if offset is None:
            break
    vectors = np.concatenate(blocks) if blocks else np.zeros((0, 0), dtype=np.float32)
    return {"ids": ids, "vectors": vectors, "stamps": stamps}


def _run_inputs(workdir: Path, query_count: int) -> tuple[list[str], str]:
    """The run's own queries + corpus digest: prefer ``run1.json``, fall back to corpus.

    The ops run records the exact query strings it measured (60 of them); the
    filtered arm reuses those first ``query_count`` so the two numbers describe
    the same query set. A workdir that carries only the corpus still works.
    """
    run1_path = workdir / "run1.json"
    if run1_path.is_file():
        run1 = json.loads(run1_path.read_text(encoding="utf-8"))
        return list(run1["queries"])[:query_count], run1["fingerprint"]["corpus"]["sha256"]
    rows, manifest = load_corpus(workdir / "corpus.jsonl")
    return build_queries(rows, count=query_count), manifest["corpus_sha256"]


def _model_fingerprint() -> dict:
    from app.retrieval import e5_local
    from app.retrieval.embedding_fingerprint import current_fingerprint

    model_path = e5_local.model_dir() / e5_local.ARCTIC_MODEL_FILE
    return {
        "id": "arctic-xs",
        "file": str(model_path),
        "model_sha256": sha256_file(model_path) if model_path.is_file() else None,
        "declared_model_sha256": e5_local.ARCTIC_MODEL_SHA256,
        "dim": 384,
        "pooling": "cls",
        "query_prefix": e5_local.ARCTIC_QUERY_PREFIX,
        "embedding_contract": current_fingerprint(),
    }


# ── strategy 1: LOCAL (exact store, methodology check) ─────────────────────


async def run_local_arm(*, workdir: Path, query_count: int, k_max: int) -> dict:
    """The LOCAL arm. Leaves the store OPEN for the server copy; caller closes."""
    configure_environment(workdir)
    from app.retrieval.embedder import embed_query, warmup_embedder
    from app.retrieval.memory.namespaces import personal_namespace
    from app.retrieval.memory.outbox import active_generation
    from app.retrieval.memory.vector_store import search_memories
    from app.retrieval.qdrant_filter import build_filter
    from app.retrieval.vector_backend import get_sync_client

    queries, corpus_sha = _run_inputs(workdir, query_count)
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    await warmup_embedder()
    timings["embedder_warmup_seconds"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    embeddings = [await embed_query(query) for query in queries]
    timings["query_embed_seconds"] = time.perf_counter() - t0

    generation, manifest_fingerprint = await active_generation()
    user_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{SCALE_USER}/{SEED}")
    namespace = personal_namespace(user_id)
    local_client = get_sync_client()

    t0 = time.perf_counter()
    full = _scroll_all(local_client, generation)
    timings["scroll_all_seconds"] = time.perf_counter() - t0
    stamps = full.pop("stamps")
    known = [stamp for stamp in stamps if stamp is not None]
    last, first = max(known), min(known)
    span_seconds = (last - first).total_seconds()
    stamp_values = np.array(
        [np.datetime64(stamp.replace(tzinfo=None)) if stamp else np.datetime64("NaT")
         for stamp in stamps],
        dtype="datetime64[us]",
    )

    buckets: list[dict] = []
    server_buckets: list[dict] = []
    for fraction in SELECTIVITY_FRACTIONS:
        start = None if fraction >= 1.0 else window_start(last, span_seconds, fraction)
        where = bucket_where(start)
        qdrant_filter = build_filter(str(user_id), where, namespace=namespace)

        # The store's own count with the translated filter (the translation check).
        t0 = time.perf_counter()
        store_count = int(local_client.count(generation, count_filter=qdrant_filter).count)
        count_seconds = time.perf_counter() - t0

        # The client-side subset: the SAME bound, applied to the one scrolled pass.
        if start is None:
            mask = np.ones(len(full["ids"]), dtype=bool)
        else:
            mask = stamp_values >= np.datetime64(start.replace(tzinfo=None))
        subset_ids = [full["ids"][i] for i in np.nonzero(mask)[0]]
        predicate_count = int(mask.sum())

        t0 = time.perf_counter()
        exact_matrix = np.asarray(full["vectors"][mask], dtype=np.float64)
        exact = [exact_top_k(exact_matrix, embedding, k_max, subset_ids)
                 for embedding in embeddings]
        exact_seconds = time.perf_counter() - t0

        recalls = {k: [] for k in K_VALUES}
        latencies: list[float] = []
        order_mismatches = 0
        t0 = time.perf_counter()
        for index, embedding in enumerate(embeddings):
            t_start = time.perf_counter()
            items = await search_memories(embedding, user_id=str(user_id), top_k=k_max,
                                          where=where, namespace=namespace)
            latencies.append((time.perf_counter() - t_start) * 1000.0)
            ann_ids = [item["memory_id"] for item in items]
            for k in K_VALUES:
                recalls[k].append(recall_at_k(ann_ids, exact[index], k))
            if ann_ids[:k_max] != exact[index][:k_max]:
                order_mismatches += 1
        search_seconds = time.perf_counter() - t0

        buckets.append({
            "fraction": fraction,
            "where": where,
            "subset_size": len(subset_ids),
            "store_count_with_filter": store_count,
            "predicate_count": predicate_count,
            "counts_agree": len(subset_ids) == store_count == predicate_count,
            "exact": "client-side cosine top-k over the subset of the ONE scrolled pass "
                     "(numpy, float64)",
            "ann": f"app search_memories (EMBEDDED Qdrant = EXACT brute force, "
                   f"{len(subset_ids)} points)",
            "recall": {f"recall@{k}": (sum(v) / len(v) if v else None)
                       for k, v in recalls.items()},
            "exact_order_mismatches": order_mismatches,
            "search_latency_ms": {
                "calls": len(latencies),
                "p50_ms": percentile(latencies, 0.50),
                "p95_ms": percentile(latencies, 0.95),
                "max_ms": max(latencies) if latencies else None,
            },
            "stage_seconds": {"count": count_seconds, "exact": exact_seconds,
                              "search": search_seconds},
        })
        server_buckets.append({"fraction": fraction, "where": where,
                               "subset_size": len(subset_ids), "exact_ids": exact})
        print(f"  local bucket {fraction:<6} subset={len(subset_ids):>6} "
              f"recall@10={buckets[-1]['recall']['recall@10']} "
              f"mismatches={order_mismatches} "
              f"p50={buckets[-1]['search_latency_ms']['p50_ms']:.1f}ms "
              f"(count={count_seconds:.1f}s exact={exact_seconds:.1f}s "
              f"search={search_seconds:.1f}s)", flush=True)

    return {
        "artifact": {
            "generation": generation,
            "generation_manifest_fingerprint": manifest_fingerprint,
            "user_id": str(user_id),
            "namespace": namespace,
            "queries": queries,
            "captured_at_span_seconds": span_seconds,
            "first_captured_at": first.isoformat(),
            "last_captured_at": last.isoformat(),
            "timings": timings,
            "buckets": buckets,
        },
        "server_inputs": {
            "generation": generation,
            "user_id": str(user_id),
            "namespace": namespace,
            "embeddings": embeddings,
            "buckets": server_buckets,
            "corpus_sha": corpus_sha,
            "model": _model_fingerprint(),
            "rss": rss_block(os.getpid()),
        },
    }


# ── strategy 2: SERVER (filtered HNSW, the real ANN number) ─────────────────


def _server_wait_until_ready(client, collection: str, expected: int, timeout: float = 600.0) -> dict:
    """Wait for the copy to be complete and the HNSW build to settle."""
    deadline = time.perf_counter() + timeout
    count, ok, status_text = 0, None, None
    indexed, collection_status = 0, "unknown"
    while time.perf_counter() < deadline:
        info = client.get_collection(collection)
        count = int(info.points_count or 0)
        indexed = int(info.indexed_vectors_count or 0)
        status = getattr(info, "optimizer_status", None)
        ok = status if isinstance(status, bool) else getattr(status, "ok", None)
        status_text = str(getattr(status, "value", status))
        collection_status = str(getattr(info.status, "value", info.status)).lower()
        # Ready means copied AND indexed: points visible before the HNSW build
        # finishes would measure a half-built index.
        if count == expected and indexed >= expected and collection_status == "green":
            return {"points_count": count, "indexed_vectors_count": indexed,
                    "collection_status": collection_status, "optimizer_ok": ok,
                    "optimizer_status": status_text,
                    "waited_seconds": timeout - (deadline - time.perf_counter())}
        time.sleep(1.0)
    raise TimeoutError(f"server collection {collection} not ready in {timeout}s "
                       f"(count={count}, indexed={indexed}, status={collection_status}, "
                       f"optimizer_status={status_text})")


def run_server_arm(*, local_client, generation: str, user_id: str, namespace: str,
                   embeddings: list[list[float]], local_buckets: list[dict],
                   server_url: str, collection: str) -> dict:
    """The SERVER arm: the same points in a server collection, filtered HNSW."""
    from qdrant_client import QdrantClient
    from qdrant_client import models as qm

    from app.retrieval.qdrant_filter import build_filter

    client = QdrantClient(url=server_url, timeout=60)
    try:
        version = None
        try:
            version = str(client.info().version)
        except Exception:  # a leaner server may not answer info()
            pass
        if client.collection_exists(collection):
            raise RuntimeError(
                f"refusing to replace existing collection {collection!r} — this harness only "
                f"creates its own; drop stale benchmark collections yourself"
            )
        client.create_collection(
            collection_name=collection,
            vectors_config=qm.VectorParams(size=len(embeddings[0]), distance=qm.Distance.COSINE),
            # Force the HNSW path even for tiny filtered subsets: with the
            # client default full_scan_threshold, selective buckets fall back to
            # exact scans and their "ANN" recall would be a full scan in
            # disguise (recorded in the artifact's collection_info).
            hnsw_config=qm.HnswConfigDiff(full_scan_threshold=0),
        )
        # The app's SERVER-mode payload index set for the fields this filter
        # reads (vector_backend._PAYLOAD_INDEXES): without them, filtered search
        # is a full scan and the arm would measure the wrong thing.
        client.create_payload_index(collection, "user_id", field_schema=qm.PayloadSchemaType.KEYWORD)
        client.create_payload_index(collection, "namespace", field_schema=qm.PayloadSchemaType.KEYWORD)
        client.create_payload_index(collection, "captured_at",
                                    field_schema=qm.PayloadSchemaType.DATETIME)

        copied = 0
        t0 = time.perf_counter()
        offset = None
        while True:
            records, offset = local_client.scroll(generation, limit=POINT_BATCH, offset=offset,
                                                  with_payload=True, with_vectors=True)
            client.upsert(
                collection_name=collection,
                points=[qm.PointStruct(id=str(record.id), vector=list(record.vector),
                                       payload=dict(record.payload or {}))
                        for record in records],
                wait=False,
            )
            copied += len(records)
            if offset is None:
                break
        copy_seconds = time.perf_counter() - t0
        ready = _server_wait_until_ready(client, collection, copied)

        buckets: list[dict] = []
        for local_bucket in local_buckets:
            where = local_bucket["where"]
            qdrant_filter = build_filter(user_id, where, namespace=namespace)
            server_count = int(client.count(collection_name=collection,
                                            count_filter=qdrant_filter).count)
            recalls = {k: [] for k in K_VALUES}
            latencies: list[float] = []
            for index, embedding in enumerate(embeddings):
                t0 = time.perf_counter()
                response = client.query_points(collection_name=collection, query=embedding,
                                               query_filter=qdrant_filter, limit=max(K_VALUES),
                                               with_payload=False)
                latencies.append((time.perf_counter() - t0) * 1000.0)
                ann_ids = [str(point.id) for point in response.points]
                for k in K_VALUES:
                    recalls[k].append(recall_at_k(ann_ids, local_bucket["exact_ids"][index], k))
            buckets.append({
                **{key: local_bucket[key] for key in ("fraction", "where", "subset_size")},
                "server_count_with_filter": server_count,
                "counts_agree": server_count == local_bucket["subset_size"],
                "ann": f"Qdrant SERVER filtered HNSW (default m/ef_construct), "
                       f"{copied} points, collection={collection}",
                "recall": {f"recall@{k}": (sum(v) / len(v) if v else None)
                           for k, v in recalls.items()},
                "search_latency_ms": {
                    "calls": len(latencies),
                    "p50_ms": percentile(latencies, 0.50),
                    "p95_ms": percentile(latencies, 0.95),
                    "max_ms": max(latencies) if latencies else None,
                },
            })
            print(f"  server bucket {local_bucket['fraction']:<6} "
                  f"subset={server_count:>6} recall@10={buckets[-1]['recall']['recall@10']} "
                  f"p50={buckets[-1]['search_latency_ms']['p50_ms']:.1f}ms")
        info = client.get_collection(collection)
        config = info.config.params.vectors
        return {
            "url": server_url,
            "version": version,
            "collection": collection,
            "points_copied": copied,
            "copy_seconds": copy_seconds,
            "ready": ready,
            "collection_info": {
                "points_count": int(info.points_count or 0),
                "dim": int(config.size),
                "distance": str(getattr(config.distance, "value", config.distance)),
                "optimizer_status": str(getattr(info.optimizer_status, "value",
                                                info.optimizer_status)),
            },
            "buckets": buckets,
            "method": "the same points+payloads copied from the embedded store into a server "
                      "collection with payload indexes for the filtered fields; queries are "
                      "the SAME embedded vectors, compared against the SAME exact lists",
        }
    finally:
        client.close()


# ── self-check (tiny local fixture, no model needed) ────────────────────────


def self_check(workdir: Path | None) -> int:
    """The harness's own check: exact store in, exact store out, recall == 1.0.

    ``--workdir`` is treated as a PARENT for a fresh scratch child: the
    harness never deletes a caller-supplied path — a mistyped scale workdir
    would take the corpus, store and artifacts down with it. The scratch child
    is always cleaned up.
    """
    parent = Path(workdir) if workdir is not None else None
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="filtered-ann-selfcheck-", dir=parent) as tmp:
        return _self_check_at(Path(tmp))


def _self_check_at(path: Path) -> int:
    from qdrant_client import QdrantClient
    from qdrant_client import models as qm

    rng = np.random.default_rng(7)
    dim, n = 16, 64
    base = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)
    stamps = [base + timedelta(minutes=index) for index in range(n)]
    client = QdrantClient(path=str(path))
    try:
        client.create_collection("self", vectors_config=qm.VectorParams(size=dim,
                                                                       distance=qm.Distance.COSINE))
        vectors = rng.normal(size=(n, dim)).astype(np.float32)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"self/{index}")) for index in range(n)]
        client.upsert("self", points=[
            qm.PointStruct(id=ids[index], vector=vectors[index].tolist(),
                           payload={"user_id": "u", "namespace": "personal",
                                    "captured_at": stamps[index].isoformat()})
            for index in range(n)
        ])
        query = rng.normal(size=dim)
        must = [qm.FieldCondition(key="user_id", match=qm.MatchValue(value="u")),
                qm.FieldCondition(key="namespace", match=qm.MatchValue(value="personal"))]

        # (1) the store's own query == client-side exact, on the whole fixture.
        # Explicit checks, not asserts: `python -O` strips asserts and would
        # print success without validating anything.
        response = client.query_points("self", query=query.tolist(),
                                       query_filter=qm.Filter(must=must), limit=10)
        ann_ids = [str(point.id) for point in response.points]
        exact = exact_top_k(vectors, query, 10, ids)
        if recall_at_k(ann_ids, exact, 10) != 1.0:
            raise RuntimeError(f"self-check: whole-fixture recall != 1.0: ann={ann_ids}")

        # (2) a selective window: counts must agree and the subset must shrink.
        start = window_start(stamps[-1], (stamps[-1] - stamps[0]).total_seconds(), 0.25)
        filtered = qm.Filter(must=[*must, qm.FieldCondition(
            key="captured_at", range=qm.DatetimeRange(gte=start))])
        selected = [index for index, stamp in enumerate(stamps) if stamp >= start]
        count = client.count("self", count_filter=filtered).count
        if not (count == len(selected) == 16):
            raise RuntimeError(f"self-check: window count mismatch: {count} vs {len(selected)} vs 16")
        response = client.query_points("self", query=query.tolist(),
                                       query_filter=filtered, limit=10)
        ann_ids = [str(point.id) for point in response.points]
        predicate = _captured_at_predicate(start.isoformat())
        subset_ids = [ids[index] for index in range(n)
                      if predicate({"captured_at": stamps[index].isoformat()})]
        exact = exact_top_k(vectors[selected], query, 10, subset_ids)
        if recall_at_k(ann_ids, exact, 10) != 1.0:
            raise RuntimeError(f"self-check: selective recall != 1.0: ann={ann_ids}")
        if recall_at_k(ann_ids[1:], exact, 10) != 0.9:
            raise RuntimeError("self-check: the metric must be able to move (0.9 expected)")
    finally:
        client.close()

    print("self-check: OK (exact store == client-side exact, selective window consistent)")
    return 0


# ── artifact + main ─────────────────────────────────────────────────────────


def write_artifact(payload: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


async def run_experiment(opts) -> dict:
    workdir = Path(opts.workdir).resolve()
    local = await run_local_arm(workdir=workdir, query_count=opts.queries, k_max=max(K_VALUES))
    inputs = local["server_inputs"]
    local_buckets = local["artifact"]["buckets"]
    counts_ok = all(bucket["counts_agree"] for bucket in local_buckets)
    recall_all_one = all(
        all(value == 1.0 for value in (bucket["recall"] or {}).values())
        for bucket in local_buckets
    )
    artifact = {
        # Counts disagreeing on the exact store is a filter-translation bug —
        # the artifact is not evidence of anything. Recall below 1.0 alone is
        # NOT invalid: with duplicated texts the top-k cut falls inside an
        # exact tie group and the store picks different (equal-score) members
        # than id-order; the per-bucket mismatch counts record it.
        "status": "complete" if counts_ok else "invalid",
        "local_validation": {
            "counts_agree_every_bucket": counts_ok,
            "recall_1.0_every_bucket": recall_all_one,
            "note": "the local arm runs against an EXACT store — counts must agree everywhere; "
                    "recall < 1.0 is expected only at exact-tie boundaries (duplicate texts), "
                    "see exact_order_mismatches per bucket",
        },
        "role": "filtered ANN vs exact subset — 100K milestone store",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "workdir": str(workdir),
        "git": git_state(),
        "corpus": {"sha256": inputs["corpus_sha"], "queries": len(local["artifact"]["queries"])},
        "model": inputs["model"],
        "local_arm": local["artifact"],
        "selectivity_note": "captured_at windows are the only varied payload field on this "
                            "workload: P4a has ONE namespace per user and this run writes ONE "
                            "user, so tenant/namespace selectivity cannot be varied here "
                            "(recorded, not silently substituted)",
        "methodology": {
            "exact": "client-side cosine top-k (numpy, float64, L2-normalized defensively) over "
                     "the SAME filtered subset the store returns; ties break on id",
            "local_ann": "app's own search_memories through build_filter — an EXACT store, so "
                         "recall 1.0 is the expected number and any miss is a bug found",
            "server_ann": "same points+payloads in a Qdrant server collection (default HNSW), "
                          "same filters, same embedded queries",
        },
        "seams": {
            "tenant_selectivity": "not measurable here (single user, one namespace) — see "
                                  "selectivity_note",
            "naive_payload_timestamps": "payload captured_at stamps read back through SQLite "
                                        "are offset-less; the arm normalizes the filter bound "
                                        "AND the client-side predicate to UTC (the app's "
                                        "filter builder refuses naive bounds on purpose)",
            "warm_cache": "every query runs in-process after a warmup; no page-cache drops",
            "hnsw_params": "server defaults (m=16, ef_construct=100, default ef search) — no "
                           "tuning was swept and no tuned number is claimed",
        },
        "rss": inputs["rss"],
    }
    if opts.server_url:
        from app.retrieval.vector_backend import close_clients, get_sync_client

        local_client = get_sync_client()
        try:
            artifact["server_arm"] = run_server_arm(
                local_client=local_client, generation=inputs["generation"],
                user_id=inputs["user_id"], namespace=inputs["namespace"],
                embeddings=inputs["embeddings"], local_buckets=inputs["buckets"],
                server_url=opts.server_url, collection=opts.server_collection)
        except Exception as exc:  # unreachable/ill server: record, keep the local arm
            artifact["server_arm"] = {
                "status": "not-run",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        finally:
            await close_clients()
        if artifact["server_arm"].get("status") != "not-run":
            counts_match = all(b["counts_agree"]
                               for b in artifact["server_arm"]["buckets"])
            artifact["server_vs_local"] = {
                "counts_match_every_bucket": counts_match,
                "note": "server counts are re-counted on the server with the same filter — a "
                        "disagreement means the copy lost points, not a tolerance",
            }
            if not counts_match:
                artifact["status"] = "invalid"
                artifact["server_arm"]["status"] = "invalid-count-mismatch"
    else:
        from app.retrieval.vector_backend import close_clients

        await close_clients()
        artifact["server_arm"] = {
            "status": "not-run",
            "reason": "no --server-url given (no Qdrant server reachable for this invocation); "
                      "the local arm alone cannot grade ANN quality",
        }
    return artifact


def _peek_corpus_sha(workdir: Path) -> str:
    """The corpus digest, from the run's own artifact (or the corpus manifest)."""
    queries, sha = _run_inputs(workdir, 0)
    del queries
    return sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="filtered recall vs the exact subset, on a run_10k store")
    parser.add_argument("--workdir", help="the finished run's workdir (has corpus.jsonl)")

    def positive_int(raw: str) -> int:
        value = int(raw)
        if value <= 0:
            raise argparse.ArgumentTypeError("must be a positive integer")
        return value

    parser.add_argument("--queries", type=positive_int, default=40)
    parser.add_argument("--server-url", default=None, help="e.g. http://127.0.0.1:6333")
    parser.add_argument("--server-collection", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)

    if args.self_check:
        return self_check(Path(args.workdir) if args.workdir else None)

    if not args.workdir:
        parser.error("--workdir is required (or use --self-check)")
    if not args.server_collection:
        args.server_collection = f"orivory_scale_filtered_{_peek_corpus_sha(Path(args.workdir))[:12]}"
    artifact = asyncio.run(run_experiment(args))
    out = Path(args.out) if args.out else (
        ARTIFACTS_DIR / f"filtered_ann_{artifact['corpus']['sha256'][:12]}.json")
    write_artifact(artifact, out)
    print(f"\nartifact: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
