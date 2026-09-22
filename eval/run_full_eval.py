#!/usr/bin/env python3
"""
Full Evaluation Pipeline

Combines:
1. Offline evaluation (deterministic retrieval metrics)
2. LLM-as-Judge evaluation (answer quality assessment)

Usage:
    # Offline only (fast, no API needed)
    python eval/run_full_eval.py --mode offline --dataset eval/orivory_extreme_eval_dataset.json

    # LLM-as-Judge (requires OpenAI API key)
    python eval/run_full_eval.py --mode judge --dataset eval/orivory_extreme_eval_dataset.json \
        --answers-file eval/extreme_results/latest_report.json

    # Full pipeline (offline + judge)
    python eval/run_full_eval.py --mode full --dataset eval/orivory_extreme_eval_dataset.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



def load_dataset(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_sample_docs(sample_docs_dir: Path) -> dict[str, str]:
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(sample_docs_dir.glob("*.md"))
    }


def run_offline_eval(
    dataset_path: Path,
    sample_docs_dir: Path,
    output_dir: Path,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run offline deterministic evaluation."""
    from eval.run_eval import run_evaluation
    return run_evaluation(
        dataset_path=dataset_path,
        sample_docs_dir=sample_docs_dir,
        output_dir=output_dir,
        top_k=top_k,
        fail_under_source_hit=0.0,
        fail_under_keyword_coverage=0.0,
        enable_ragas=False,
    )


async def run_judge_eval(
    dataset_path: Path,
    answers_file: Path,
    output_dir: Path,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run LLM-as-Judge evaluation."""
    from eval.llm_judge import (
        evaluate_case_offline,
        generate_judge_report,
        run_llm_judge_evaluation,
        summarize_judge_results,
    )

    dataset = load_dataset(dataset_path)

    # Load answers from offline eval results
    with open(answers_file) as f:
        offline_results = json.load(f)

    # Build answers and contexts dicts
    answers: dict[str, str] = {}
    contexts: dict[str, str] = {}

    for result in offline_results.get("results", []):
        case_id = result["id"]
        answers[case_id] = result.get("answer", "")

        # Build context from retrieved sources
        sources = result.get("returned_sources", [])
        if sources:
            docs = load_sample_docs(Path(offline_results["metadata"].get("sample_docs", "sample_docs")))
            context_parts = []
            for src in sources[:3]:  # Top 3 sources
                if src in docs:
                    context_parts.append(f"=== {src} ===\n{docs[src][:500]}")
            contexts[case_id] = "\n\n".join(context_parts)

    print("Running LLM-as-Judge evaluation...")
    print(f"Cases: {len(dataset)}")
    print(f"Model: {model}")

    try:
        results = await run_llm_judge_evaluation(
            dataset=dataset,
            answers=answers,
            contexts=contexts,
            model=model,
            api_key=api_key,
        )
    except Exception as e:
        print(f"LLM judge failed: {e}")
        print("Falling back to offline heuristics...")

        # Fallback to heuristic evaluation
        results = []
        for case in dataset:
            case_id = case.get("id", "unknown")
            answer = answers.get(case_id, "")
            context = contexts.get(case_id, "")
            results.append(evaluate_case_offline(case, answer, context))

    summary = summarize_judge_results(results)

    # Generate report
    report_md = generate_judge_report(results, summary)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "judge_report.md").write_text(report_md, encoding="utf-8")
    (output_dir / "judge_results.json").write_text(
        json.dumps({"summary": summary, "results": [vars(r) for r in results]}, indent=2, default=str),
        encoding="utf-8",
    )

    print("\n" + "=" * 50)
    print("LLM-as-Judge Results")
    print("=" * 50)
    print(f"Cases evaluated: {summary['total_cases']}")
    print(f"Pass rate: {summary['pass_rate']:.1%}")
    print(f"Faithfulness: {summary['avg_faithfulness']:.1%}")
    print(f"Answer Relevancy: {summary['avg_answer_relevancy']:.1%}")
    print(f"Reasoning Quality: {summary['avg_reasoning_quality']:.1%}")
    print(f"Overall Score: {summary['avg_overall_score']:.1%}")
    print(f"\nReport: {output_dir / 'judge_report.md'}")

    return {"summary": summary, "results": [vars(r) for r in results]}


def generate_full_report(
    offline_summary: dict[str, Any],
    judge_summary: dict[str, Any] | None,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate combined evaluation report."""

    report = {
        "evaluation_modes": [],
        "retrieval_quality": offline_summary,
        "answer_quality": judge_summary,
        "combined_score": 0.0,
        "recommendations": [],
    }

    # Calculate combined score
    scores = []

    # Retrieval metrics
    report["evaluation_modes"].append("offline")
    if offline_summary.get("source_hit_rate", 0) >= 0.9:
        scores.append(0.3)  # Retrieval weight
        report["retrieval_status"] = "✅ Excellent"
    elif offline_summary.get("source_hit_rate", 0) >= 0.7:
        scores.append(0.2)
        report["retrieval_status"] = "⚠️ Acceptable"
    else:
        scores.append(0.1)
        report["retrieval_status"] = "❌ Poor"

    # LLM judge metrics
    if judge_summary:
        report["evaluation_modes"].append("llm_judge")
        judge_score = judge_summary.get("avg_overall_score", 0.5)
        if judge_score >= 0.8:
            scores.append(0.4)  # Answer quality weight
            report["answer_quality_status"] = "✅ Excellent"
        elif judge_score >= 0.6:
            scores.append(0.3)
            report["answer_quality_status"] = "⚠️ Acceptable"
        else:
            scores.append(0.1)
            report["answer_quality_status"] = "❌ Poor"

        # Add reasoning quality specifically
        report["reasoning_quality"] = judge_summary.get("reasoning_quality_score", 0.0)

    report["combined_score"] = sum(scores)

    # Generate recommendations
    if offline_summary.get("source_hit_rate", 0) < 0.9:
        report["recommendations"].append(
            "Improve retrieval quality - source hit rate below 90%"
        )

    if judge_summary and judge_summary.get("avg_reasoning_quality", 1.0) < 0.7:
        report["recommendations"].append(
            "Improve reasoning quality - consider better prompting or model upgrade"
        )

    if judge_summary:
        # Check extreme cases
        extreme_score = None
        if "extreme" in judge_summary.get("by_difficulty", {}):
            extreme_score = judge_summary["by_difficulty"]["extreme"]["avg_score"]
            if extreme_score < 0.6:
                report["recommendations"].append(
                    f"Extreme reasoning cases need attention (score: {extreme_score:.1%})"
                )

    # Write combined report
    report_path = output_dir / "combined_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    return report


def print_full_summary(report: dict[str, Any]) -> None:
    """Print formatted summary to console."""
    print("\n" + "=" * 60)
    print("FULL EVALUATION PIPELINE SUMMARY")
    print("=" * 60)

    print(f"\nEvaluation Modes: {', '.join(report['evaluation_modes'])}")

    print("\n## Retrieval Quality (Offline)")
    print(f"  Source Hit Rate:     {report['retrieval_quality'].get('source_hit_rate', 0):.1%}")
    print(f"  Keyword Coverage:    {report['retrieval_quality'].get('keyword_coverage', 0):.1%}")
    print(f"  Fallback Accuracy:   {report['retrieval_quality'].get('fallback_accuracy', 0):.1%}")
    print(f"  Status:             {report.get('retrieval_status', 'N/A')}")

    if report.get("answer_quality"):
        print("\n## Answer Quality (LLM-as-Judge)")
        jq = report["answer_quality"]
        print(f"  Faithfulness:       {jq.get('avg_faithfulness', 0):.1%}")
        print(f"  Answer Relevancy:    {jq.get('avg_answer_relevancy', 0):.1%}")
        print(f"  Reasoning Quality:   {jq.get('reasoning_quality_score', 0):.1%}")
        print(f"  Overall Score:      {jq.get('avg_overall_score', 0):.1%}")
        print(f"  Pass Rate:          {jq.get('pass_rate', 0):.1%}")
        print(f"  Status:             {report.get('answer_quality_status', 'N/A')}")

        # By difficulty
        if jq.get("by_difficulty"):
            print("\n  Scores by Difficulty:")
            for diff, scores in jq["by_difficulty"].items():
                print(f"    {diff.capitalize()}: {scores['avg_score']:.1%}")

    print("\n## Combined Score")
    combined = report.get("combined_score", 0)
    if combined >= 0.7:
        status = "✅ Excellent"
    elif combined >= 0.5:
        status = "⚠️ Acceptable"
    else:
        status = "❌ Needs Improvement"
    print(f"  Score: {combined:.2f}/1.0 - {status}")

    if report.get("recommendations"):
        print("\n## Recommendations")
        for rec in report["recommendations"]:
            print(f"  • {rec}")


def main():
    parser = argparse.ArgumentParser(
        description="Full RAG Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Offline evaluation only
  python eval/run_full_eval.py --mode offline --dataset eval/orivory_extreme_eval_dataset.json

  # Offline + LLM judge
  python eval/run_full_eval.py --mode judge --dataset eval/orivory_extreme_eval_dataset.json

  # Full pipeline (offline + judge)
  python eval/run_full_eval.py --mode full --dataset eval/orivory_extreme_eval_dataset.json
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["offline", "judge", "full"],
        default="full",
        help="Evaluation mode",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "eval" / "orivory_extreme_eval_dataset.json",
    )
    parser.add_argument(
        "--sample-docs",
        type=Path,
        default=ROOT / "sample_docs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "eval" / "full_results",
    )
    parser.add_argument("--top-k", type=int, default=5)

    # LLM Judge options
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--openai-api-key", default=None)

    # For judge mode - where to get answers
    parser.add_argument(
        "--answers-file",
        type=Path,
        default=None,
        help="Path to offline eval results JSON for judge evaluation",
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting evaluation in '{args.mode}' mode")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {output_dir}")
    print()

    # Track results
    offline_summary = None
    judge_summary = None

    # 1. Offline evaluation (always runs)
    if args.mode in ("offline", "judge", "full"):
        print("=" * 50)
        print("PHASE 1: Offline Evaluation")
        print("=" * 50)

        offline_results = run_offline_eval(
            dataset_path=args.dataset,
            sample_docs_dir=args.sample_docs,
            output_dir=output_dir,
            top_k=args.top_k,
        )
        offline_summary = offline_results.get("summary", {})
        print()

    # 2. LLM-as-Judge evaluation
    if args.mode in ("judge", "full"):
        print("=" * 50)
        print("PHASE 2: LLM-as-Judge Evaluation")
        print("=" * 50)

        answers_file = args.answers_file or (output_dir / "latest_report.json")

        if not answers_file.exists():
            print(f"ERROR: Answers file not found: {answers_file}")
            print("Run --mode offline first to generate answers")
            return 1

        judge_results = asyncio.run(run_judge_eval(
            dataset_path=args.dataset,
            answers_file=answers_file,
            output_dir=output_dir,
            model=args.judge_model,
            api_key=args.openai_api_key,
        ))
        judge_summary = judge_results.get("summary")
        print()

    # Generate combined report
    if args.mode == "full":
        print("=" * 50)
        print("PHASE 3: Generating Combined Report")
        print("=" * 50)

        combined = generate_full_report(
            offline_summary=offline_summary or {},
            judge_summary=judge_summary,
            output_dir=output_dir,
        )

        print_full_summary(combined)

        print(f"\nFull report saved to: {output_dir / 'combined_report.json'}")

    print("\n" + "=" * 50)
    print("Evaluation complete!")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    import asyncio
    sys.exit(main())
