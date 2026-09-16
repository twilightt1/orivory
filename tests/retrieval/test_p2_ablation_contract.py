"""T7 — the ablation artifact's CONTRACT: shape, arms, slices, verdicts. Never the numbers.

The numbers in ``eval/ablation_retrieval_p2.json`` are MEASUREMENT: they move with the
box, the model cache and the clock, and a test that pinned them would be the thing
that has to be edited when a number moves — which is exactly how a signed gate gets
walked backwards. What must not move silently is the SHAPE:

- every arm (dense / lexical / hybrid RRF / hybrid+rerank) with every slice and every
  metric (recall@{1,5,10}, MRR), plus the per-slice deltas vs dense-only;
- the fixture's seed and corpus hash, re-derived from the script's own builder — so
  "frozen" is checked, not asserted;
- the C1 evidence (the lexical leg ran in every arm that claims one, and the OFF arm
  never probed the index);
- the C2 skew number (a measured intra-tenant order flip count with a foreign size);
- the C3 hydration verdict with its measured share and the arm-(e) disposition;
- the enable rule's signed thresholds, its verdict, and that the verdict agrees with
  the arms and the deltas it was computed from — per slice, not just in aggregate.

The artifact and the script are read from the repository root, so this suite runs
from anywhere in the tree.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = ROOT / "eval" / "ablation_retrieval_p2.json"
SCRIPT_PATH = ROOT / "eval" / "ablation_retrieval_p2.py"

ARMS = ("dense_only", "lexical_only", "hybrid_rrf", "hybrid_rerank")
SLICES = (
    "exact_id",
    "vi_diacritics",
    "vi_no_diacritics",
    "en",
    "short_query",
    "long_query",
)
METRICS = ("recall@1", "recall@5", "recall@10", "mrr")
LEXICAL_ARMS = ("lexical_only", "hybrid_rrf", "hybrid_rerank")
HYDRATION_VERDICTS = ("not implemented: no measured benefit", "implemented: late hydration (arm e parity measured)")


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ablation():
    """The script itself — imported so the fixture can be REBUILT, not trusted."""
    spec = importlib.util.spec_from_file_location("p2_ablation_retrieval", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None, "the ablation script must exist"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_artifact_is_the_task7_ablation(artifact):
    assert artifact["artifact"] == "p2-task7-ablation-retrieval"
    assert artifact["plan"] == "2026-09-15-retrieval-correctness-hybrid-p2"
    assert artifact["generated_at"]
    # Fixture scale is part of the claim: the §12.2 budgets are signed there.
    assert artifact["scales"]["kind"] == "fixture"
    assert artifact["scales"]["memories"] > 0
    assert artifact["scales"]["queries"] > 0
    assert artifact["limitations"], "an artifact without its limits is a production claim"


def test_every_arm_carries_every_slice_and_every_metric(artifact):
    assert tuple(artifact["arms"]) == ARMS
    for arm, payload in artifact["arms"].items():
        assert payload["config"], f"{arm}: the arm's configuration is not recorded"
        assert set(payload["per_slice"]) == set(SLICES), f"{arm}: slice set moved"
        for slice_name in SLICES:
            slice_metrics = payload["per_slice"][slice_name]
            assert set(METRICS) <= set(slice_metrics), f"{arm}/{slice_name}: metric missing"
            # n is the fixture's own query count for the slice, not a second opinion.
            assert slice_metrics["n"] == artifact["fixture"]["queries_per_slice"][slice_name], (
                f"{arm}/{slice_name}: slice n is not the fixture's query count"
            )
            for metric in METRICS:
                assert 0.0 <= slice_metrics[metric] <= 1.0
        for metric in METRICS:
            assert metric in payload["overall"]
    # Every arm but the reference carries a delta for every slice.
    compare_arms = tuple(arm for arm in ARMS if arm != "dense_only")
    assert tuple(artifact["deltas_vs_dense_only"]) == compare_arms
    for arm in compare_arms:
        for slice_name in SLICES:
            delta = artifact["deltas_vs_dense_only"][arm]["per_slice"][slice_name]
            assert "recall@5_loss" in delta
            assert set(METRICS) <= set(delta)
        assert "recall@5_gain" in artifact["deltas_vs_dense_only"][arm]["overall"]


def test_the_fixture_is_frozen_and_reproducible(artifact, ablation):
    """The seed rebuilds the SAME corpus: the hash is re-derived, not re-read."""
    fixture_record = artifact["fixture"]
    assert fixture_record["seed"] == ablation.SEED
    assert fixture_record["slices"] == list(SLICES)
    assert fixture_record["queries_per_slice"] == {name: 8 for name in SLICES}

    rebuilt = ablation.build_fixture()
    assert ablation.corpus_hash(rebuilt) == fixture_record["corpus_hash"]
    assert ablation.query_set_hash(rebuilt) == fixture_record["query_set_hash"]
    assert len(rebuilt["rows"]) == artifact["scales"]["memories"]
    assert len(rebuilt["queries"]) == artifact["scales"]["queries"]
    # §7.4: two UUIDs, one text — the fixture must keep carrying that case.
    duplicate = fixture_record["duplicate_text_case"]
    assert len(duplicate["memory_ids"]) == 2
    assert len(set(duplicate["memory_ids"])) == 2
    assert duplicate["text"]
    # The FTS index over that corpus was complete when the arms ran (C1's floor).
    assert fixture_record["fts_coverage"]["missing"] == 0
    assert fixture_record["fts_coverage"]["orphan"] == 0


def test_c1_every_arm_that_claims_a_lexical_leg_ran_one(artifact):
    for arm in LEXICAL_ARMS:
        assert artifact["arms"][arm]["c1_lexical_leg_ran"] is True, arm
        assert "lexical" in artifact["arms"][arm]["trace"]["counts_keys_seen"], arm
    # The OFF path is dense-only: no lexical key ever appeared (R2(p2)).
    assert artifact["arms"]["dense_only"]["c1_lexical_leg_ran"] is None
    assert "lexical" not in artifact["arms"]["dense_only"]["trace"]["counts_keys_seen"]
    # The outage arm answered from the lexical leg and from nothing else.
    assert "lexical" in artifact["arms"]["lexical_only"]["trace"]["counts_keys_seen"]
    assert "dense" not in artifact["arms"]["lexical_only"]["trace"]["counts_keys_seen"]
    assert artifact["outage_arm_fallbacks"]["retrieval.vector_unavailable"] > 0
    # The rerank arm's stage really ran (a fallback would carry no `reranked`).
    assert "reranked" in artifact["arms"]["hybrid_rerank"]["trace"]["counts_keys_seen"]
    assert artifact["rerank_stage"]["orders_diff_from_hybrid_rrf"] > 0
    # The duplicate-text pair survives the fusion in the arms that can carry it.
    assert artifact["fixture"]["duplicate_text_case"]["golds_in_served_top10_per_arm"]["hybrid_rrf"] == 2


def test_c2_the_bm25_skew_is_measured_and_named_a_limitation(artifact):
    skew = artifact["bm25_skew"]
    assert "index-global" in skew["known_limitation"].lower()
    assert "none" in skew["authorization_impact"]
    measurement = skew["measurement"]
    assert measurement["probe_queries"] >= 2
    assert len(measurement["probes"]) == measurement["probe_queries"]
    assert all(probe["query"] and probe["baseline_rows"] >= 2 for probe in measurement["probes"])
    sizes = measurement["foreign_row_sizes"]
    assert len(sizes) >= 2
    for size in sizes:
        entry = measurement["results"][str(size)]
        assert entry["foreign_rows"] == size
        assert isinstance(entry["flipped_orders"], int) and entry["flipped_orders"] >= 0
        assert entry["probe_queries"] == measurement["probe_queries"]
    # A number, not a promise: at least one foreign size is recorded with its count.
    assert measurement["flips_at"], "no skew number recorded"


def test_c3_the_hydration_verdict_carries_its_measured_share(artifact):
    hydration = artifact["hydration"]
    assert hydration["signed_threshold"] == 0.25
    assert hydration["verdict"] in HYDRATION_VERDICTS
    deciding = hydration["deciding_measurement"]
    assert hydration["deciding_arm"] in ARMS
    assert set(deciding) == {"mean", "p95", "max", "n"}
    assert deciding["n"] > 0
    assert 0.0 <= deciding["mean"] <= 1.0 and 0.0 <= deciding["max"] <= 1.0
    # Every arm's share is on the record — a verdict may not average away a crossing arm.
    per_arm = hydration["measured"]["hydrate_ms_over_total"]["per_arm"]
    assert set(per_arm) == set(ARMS)
    under = all(
        deciding[measure] < hydration["signed_threshold"] for measure in ("mean", "p95", "max")
    )
    assert (hydration["verdict"] == HYDRATION_VERDICTS[0]) is under, (
        "the verdict must be decided the way the script decides it: mean AND p95 AND max "
        "under the trigger (a mean-only reading would flip it on one crossing percentile)"
    )
    assert hydration["parity_arm"]["status"] == ("not run" if under else "required")
    assert hydration["rationale"]


def test_the_enable_rule_is_signed_and_its_verdict_matches_the_deltas(artifact):
    rule = artifact["enable_rule"]
    assert rule["flag"] == "RETRIEVAL_HYBRID_ENABLED"
    thresholds = rule["signed_thresholds"]
    assert thresholds == {"max_slice_loss_recall@5": 0.02, "min_overall_gain_recall@5": 0.02}
    assert rule["verdicts"]["hybrid_rrf"]["decides_enable"] is True

    dense_slices = artifact["arms"]["dense_only"]["per_slice"]
    for arm, verdict in rule["verdicts"].items():
        assert verdict["verdict"] in ("PASS", "FAIL"), arm
        losses = verdict["per_slice_loss_recall@5"]
        assert set(losses) == set(SLICES), f"{arm}: the rule must be applied per slice"
        assert verdict["max_slice_loss_recall@5"] == max(losses.values())
        expected = (
            verdict["max_slice_loss_recall@5"] <= thresholds["max_slice_loss_recall@5"] + 1e-9
            and verdict["overall_gain_recall@5"] >= thresholds["min_overall_gain_recall@5"] - 1e-9
        )
        assert (verdict["verdict"] == "PASS") is expected, f"{arm}: verdict disagrees with its own numbers"
        assert set(verdict["failing_slices"]) <= set(SLICES)
        # The gain is the difference of the two recorded aggregates, not a third number.
        gain = (
            artifact["arms"][arm]["overall"]["recall@5"]
            - artifact["arms"]["dense_only"]["overall"]["recall@5"]
        )
        assert verdict["overall_gain_recall@5"] == pytest.approx(gain, abs=1e-6)
        # ... and the same arithmetic at SLICE scale: every per-slice loss the verdict
        # reads IS dense's recall@5 minus the arm's, and the delta block repeats that
        # number. One computation, three blocks (arms, deltas, verdict), tied here —
        # a fabricated per-slice recall@5 with the deltas and the verdict left intact
        # is exactly what this catches.
        arm_slices = artifact["arms"][arm]["per_slice"]
        for slice_name in SLICES:
            loss = round(
                dense_slices[slice_name]["recall@5"] - arm_slices[slice_name]["recall@5"], 6
            )
            assert losses[slice_name] == loss, (
                f"{arm}/{slice_name}: the enable rule is not reading the arms' own numbers"
            )
            delta = artifact["deltas_vs_dense_only"][arm]["per_slice"][slice_name]
            assert delta["recall@5_loss"] == loss, (
                f"{arm}/{slice_name}: the delta block and the enable rule disagree with the arms"
            )
            assert delta["recall@5"] == round(
                arm_slices[slice_name]["recall@5"] - dense_slices[slice_name]["recall@5"], 6
            ), f"{arm}/{slice_name}: the delta's recall@5 is not the arm difference"


def test_the_artifact_records_its_substitutions_and_stays_portable(artifact):
    substitutions = artifact["substitutions"]
    for key in ("embedding", "dense_store", "lexical", "rerank", "rewrite", "modifiers"):
        assert substitutions[key]
    raw = ARTIFACT_PATH.read_text(encoding="utf-8")
    assert "/Users/" not in raw and "/tmp/" not in raw, (
        "the artifact must not carry this box's paths — it is reviewed elsewhere"
    )
    # Stage timings and latency are measured per arm, so the hydration share is auditable.
    for arm, payload in artifact["arms"].items():
        assert "hydrate_ms" in payload["trace"]["stage_ms_mean"], arm
        assert payload["latency_ms"]["p95"] > 0
