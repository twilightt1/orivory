"""CI-safe contracts for the 10K scale workload: generator determinism + ops artifact shape.

Nothing here touches a paid API. The generator contracts run everywhere (the
committed LongMemEval-S fixture stands in for the real dataset, which is
gitignored); the mini ops run is ``skipif``-guarded on the local arctic model
being cached, exactly like the other local-model suites in this repo.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DATASET = REPO_ROOT / "eval" / "benchmarks" / "fixtures" / "longmemeval_s_fixture.json"
RUNNER = REPO_ROOT / "eval" / "scale" / "run_10k.py"

pytestmark = pytest.mark.eval


def _rows(path: Path) -> list[dict]:
    """JSONL rows: split on "\\n" only (see gen_corpus.load_corpus)."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def _generate(dirpath: Path, *, rows: int, seed: int = 7) -> tuple[Path, dict]:
    from eval.scale.gen_corpus import generate

    out = dirpath / "corpus.jsonl"
    manifest = generate(rows=rows, seed=seed, dataset=FIXTURE_DATASET, out=out)
    return out, manifest


def test_same_seed_is_byte_identical(tmp_path):
    """The determinism gate: same seed -> same corpus bytes and the same manifest.

    The digest is the self-check the plan asks for — if this fails, every
    number measured against the corpus is un-reproducible.
    """
    a_jsonl, a_manifest = _generate(tmp_path / "a", rows=200)
    b_jsonl, b_manifest = _generate(tmp_path / "b", rows=200)

    assert a_jsonl.read_bytes() == b_jsonl.read_bytes()
    assert a_manifest == b_manifest
    assert a_manifest["corpus_sha256"] == hashlib.sha256(a_jsonl.read_bytes()).hexdigest()


def test_different_seed_moves_the_corpus(tmp_path):
    a_jsonl, _ = _generate(tmp_path / "a", rows=200, seed=7)
    b_jsonl, _ = _generate(tmp_path / "b", rows=200, seed=8)

    assert a_jsonl.read_bytes() != b_jsonl.read_bytes()


def test_manifest_records_both_parts_separately(tmp_path):
    jsonl, manifest = _generate(tmp_path, rows=200)
    rows = _rows(jsonl)

    assert len(rows) == 200
    real = [row for row in rows if row["part"] == "real"]
    synthetic = [row for row in rows if row["part"] == "synthetic"]
    assert manifest["parts"]["real"]["count"] == len(real) > 0
    assert manifest["parts"]["synthetic"]["count"] == len(synthetic) > 0
    assert len(real) + len(synthetic) == 200

    # Every row is plain data: an id, a part, a language, a non-empty text.
    assert all(row["text"].strip() for row in rows)
    assert {row["lang"] for row in synthetic} == {"vi", "en"}
    for row in real:
        assert row["source"]["question_id"] and row["source"]["session_index"] >= 0

    # The dataset digest travels with the corpus, and the split is recorded
    # (the "requested" vs "selected" distinction is the honest one: a small
    # pool cannot fill a large share).
    assert manifest["dataset"]["sha256"] == hashlib.sha256(FIXTURE_DATASET.read_bytes()).hexdigest()
    assert manifest["dataset"]["requested_real"] >= manifest["parts"]["real"]["count"]
    assert manifest["seed"] == 7 and manifest["rows"] == 200


def test_missing_dataset_fails_loudly_never_fabricates(tmp_path):
    from eval.scale.gen_corpus import CorpusError, generate

    with pytest.raises(CorpusError, match="longmemeval_s_cleaned"):
        generate(rows=10, seed=1, dataset=tmp_path / "absent.json", out=tmp_path / "c.jsonl")
    assert not (tmp_path / "c.jsonl").exists(), "a failed generation must write nothing"


def test_corpus_ids_are_unique(tmp_path):
    jsonl, _ = _generate(tmp_path, rows=200)
    ids = [row["memory_id"] for row in _rows(jsonl)]
    assert len(set(ids)) == len(ids) == 200


def test_line_separator_in_real_text_does_not_shred_a_row(tmp_path):
    """Regression: real LongMemEval-S turns carry U+2028, which ``json.dumps``
    does not escape and ``str.splitlines`` splits on — one row became two and
    the corpus could not be read back (the 10K run died on it)."""
    dataset = tmp_path / "mini_dataset.json"
    dataset.write_text(json.dumps([{
        "question_id": "sep_0001",
        "haystack_sessions": [[
            {"role": "user", "content": "before\u2028after " + "x" * 30},
        ]],
    }]), encoding="utf-8")
    from eval.scale.gen_corpus import generate, load_corpus

    out = tmp_path / "corpus.jsonl"
    manifest = generate(rows=4, seed=3, dataset=dataset, out=out)
    rows, read_manifest = load_corpus(out)
    assert read_manifest["corpus_sha256"] == manifest["corpus_sha256"]
    assert len(rows) == 4
    assert any("\u2028" in row["text"] for row in rows), "the real text must survive intact"


ARCHETIC_CACHED = pytest.mark.skipif(
    not __import__("app.retrieval.e5_local", fromlist=["x"]).arctic_files_cached(),
    reason="local arctic XS model not cached (CI) — the mini ops run needs the real embedder",
)


@ARCHETIC_CACHED
def test_mini_ops_run_produces_a_real_artifact(tmp_path):
    """The harness at 300 memories: real stores, real embedder, real drill.

    Skipped without the cached model; when it runs, every number in the
    artifact must be self-consistent (counts agree, percentiles ordered,
    backup restore verified, outbox drained). Run 1 ingests, run 2 re-opens
    the same store in a fresh process — the cold-process measurement.
    """
    out = tmp_path / "artifact.json"
    workdir = tmp_path / "run"
    run1 = [
        sys.executable, str(RUNNER),
        "--rows", "300", "--dataset", str(FIXTURE_DATASET),
        "--workdir", str(workdir), "--out", str(out),
        "--concurrent-seconds", "4",
    ]
    proc = subprocess.run(run1, capture_output=True, text=True, timeout=900, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]

    run2 = [sys.executable, str(RUNNER), "--reuse", str(workdir), "--out", str(out)]
    proc = subprocess.run(run2, capture_output=True, text=True, timeout=900, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]

    artifact = json.loads(out.read_text(encoding="utf-8"))
    assert artifact["status"] == "complete"
    assert artifact["rss_ceiling"] == "pending-user"

    counts = artifact["counts"]
    assert counts["memories_sql"] >= 1
    assert counts["memory_points"] == counts["memories_sql"], "SQL rows and vector points disagree"
    assert counts["chunks"] == 0, "the workload is memory-only; chunks are recorded separately"
    assert counts["corpus_rows"] == 300

    ingest = artifact["ingest"]
    assert ingest["memories"] >= 1 and ingest["seconds"] > 0
    assert ingest["memories_per_min"] == pytest.approx(
        ingest["memories"] / (ingest["seconds"] / 60.0), rel=1e-6)

    concurrent = artifact["concurrent"]
    assert concurrent["recall_calls"] >= 1
    latency = concurrent["latency_ms"]
    assert latency["p50_ms"] <= latency["p95_ms"] <= latency["p99_ms"]
    assert latency["calls"] == concurrent["recall_calls"]
    assert concurrent["gate"]["p95_le_150ms"] in (True, False)  # recorded, never asserted on

    backup = artifact["backup"]
    assert backup["backup_seconds"] > 0 and backup["restore_seconds"] > 0
    assert backup["verified"] is True
    assert backup["restored_memories"] == counts["memories_sql"]
    assert backup["restored_points"] == counts["memory_points"]

    outbox = artifact["outbox"]
    assert outbox["pending"] == 0
    assert outbox["acked"] >= 1

    assert artifact["disk"]["memories_db_bytes"] > 0
    assert artifact["rss"]["peak_rss_bytes"] > 0
    assert artifact["fingerprint"]["corpus"]["sha256"] == artifact["corpus"]["corpus_sha256"]
    assert len(artifact["fingerprint"]["git"]["head"]) == 40
    assert artifact["fingerprint"]["model"]["id"] == "arctic-xs"
    assert artifact["counts_run2"]["memories_sql"] == counts["memories_sql"], (
        "the reuse pass must see the same store it left behind")
    assert artifact["cold_process"]["first_recall_ms"] > 0
