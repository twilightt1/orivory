"""
LLM-as-Judge Evaluation Module

Uses an LLM to evaluate RAG answer quality on:
1. Faithfulness - Does the answer stick to the retrieved context?
2. Answer Relevancy - Is the answer relevant to the question?
3. Context Precision - Are the retrieved documents relevant?
4. Reasoning Quality - Does the answer show correct reasoning?

Supports multiple evaluation modes:
- Self-contained: Judge evaluates with context only
- Comparison: Judge compares LLM answer vs ground truth
- Multi-dimensional: Judge scores multiple aspects separately
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# Default judge prompt templates
JUDGE_SYSTEM_PROMPT = """You are an expert RAG evaluator. Your task is to evaluate the quality of answers
from a Retrieval-Augmented Generation system. Be strict but fair in your evaluation.

Score scale:
- 1.0: Excellent - fully correct, complete, well-reasoned
- 0.8: Good - mostly correct, minor omissions
- 0.6: Acceptable - partially correct, some gaps
- 0.4: Poor - significant errors or omissions
- 0.2: Very poor - mostly incorrect
- 0.0: Completely wrong or harmful
"""


@dataclass
class JudgeResult:
    """Result from LLM judge evaluation."""

    faithfulness: float  # 0-1: Answer aligns with context
    answer_relevancy: float  # 0-1: Answer addresses the question
    reasoning_quality: float  # 0-1: Logic and reasoning quality
    overall_score: float  # 0-1: Weighted average
    reasoning: str  # Explanation of the scores
    errors: list[str] = field(default_factory=list)  # Identified issues
    suggestions: list[str] = field(default_factory=list)  # Improvement suggestions


@dataclass
class CaseJudgeResult:
    """Judge result for a single evaluation case."""

    case_id: str
    query: str
    answer: str
    context: str
    reasoning_type: str
    difficulty: str

    # Scores
    faithfulness: float
    answer_relevancy: float
    reasoning_quality: float
    overall_score: float

    # Reasoning trace
    judge_reasoning: str
    errors: list[str]
    suggestions: list[str]

    # Metadata
    model_used: str | None = None
    latency_ms: float | None = None
    # Provenance: "llm" for real judged results, "heuristic" for the offline
    # fallback. Aggregates must split on this — heuristic constants blended
    # into headline scores present fabricated numbers as judged quality.
    judged_by: str = "llm"


def _build_judge_prompt(
    query: str,
    answer: str,
    context: str,
    reasoning_type: str,
    difficulty: str,
    ground_truth: str | None = None,
) -> str:
    """Build the evaluation prompt for the judge."""

    # Add specific instructions based on reasoning type
    reasoning_hints = {
        "constraint_satisfying": "Look for whether the answer identifies the STRICTEST constraint.",
        "edge_case_inference": "Check if the answer handles edge cases correctly.",
        "tradeoff_reasoning": "Evaluate if the answer weighs tradeoffs appropriately.",
        "logical_inconsistency": "Look for whether the answer detects contradictions.",
        "comparative_definition": "Check if the answer distinguishes similar concepts.",
        "implication_chain": "Evaluate if the answer follows logical implications.",
        "negative_inference": "Look for whether the answer infers from absence.",
        "diagnostic_decomposition": "Check if the answer synthesizes multi-system diagnostics.",
        "exception_logic": "Look for whether the answer distinguishes allowed vs not allowed.",
        "version_comparison": "Check if the answer correctly compares version features.",
        "causal_chain": "Evaluate if the answer identifies indirect causation.",
        "numeric_calculation": "Look for whether the answer performs correct calculations.",
        "prohibition_inference": "Check if the answer identifies what NOT to do.",
        "abstraction_mapping": "Evaluate understanding of abstract relationships.",
        "boolean_logic": "Check if the answer correctly applies boolean logic.",
        "temporal_ordering": "Look for correct chronological reasoning.",
        "feature_separation": "Check if the answer distinguishes independent features.",
        "state_machine": "Look for understanding of state transitions.",
        "negation_understanding": "Check if the answer handles explicit negations.",
        "order_analysis": "Evaluate if the answer identifies step dependencies.",
        "boundary_value": "Check for correct boundary/limit understanding.",
        "direct_lookup": "Verify if correct values are retrieved from docs.",
        "irrelevant": "N/A - this is an out-of-scope query.",
    }

    reasoning_hint = reasoning_hints.get(reasoning_type, "General reasoning evaluation.")
    difficulty_multiplier = {"extreme": 3, "hard": 2, "medium": 1, "easy": 0.5}.get(difficulty, 1)

    prompt = f"""Evaluate the following RAG answer:

## Question
{query}

## Retrieved Context
{context[:3000] if context else '[No context retrieved]'}

## Generated Answer
{answer if answer else '[No answer generated]'}

## Evaluation Criteria
1. **Faithfulness** (0-1): Does the answer stay true to the retrieved context?
   - Deduct points for: hallucination, adding info not in context, contradicting context

2. **Answer Relevancy** (0-1): Does the answer directly address the question?
   - Deduct points for: off-topic content, incomplete answers, answering wrong question

3. **Reasoning Quality** (0-1): Is the reasoning logical and correct?
   - Reasoning type: {reasoning_type}
   - Hint: {reasoning_hint}
   - Deduct points for: logical errors, wrong conclusions, missing steps

## Ground Truth (if available)
{ground_truth if ground_truth else '[Not provided]'}

## Your Task
Provide a JSON response with:
{{
    "faithfulness": <0.0-1.0>,
    "answer_relevancy": <0.0-1.0>,
    "reasoning_quality": <0.0-1.0>,
    "overall_score": <weighted average>,
    "reasoning": "<2-3 sentence explanation>",
    "errors": ["<error1>", "<error2>"],
    "suggestions": ["<suggestion1>", "<suggestion2>"]
}}

IMPORTANT:
- Difficulty is {difficulty} (multiplier: {difficulty_multiplier}x stricter for harder cases)
- Be STRICT for extreme/hard cases
- Score must be a number between 0.0 and 1.0
"""
    return prompt


async def evaluate_with_llm_judge(
    query: str,
    answer: str,
    context: str,
    reasoning_type: str = "general",
    difficulty: str = "medium",
    ground_truth: str | None = None,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
) -> JudgeResult:
    """
    Evaluate answer quality using LLM-as-Judge.

    Args:
        query: The original question
        answer: The generated answer
        context: Retrieved context/documents
        reasoning_type: Type of reasoning required
        difficulty: Difficulty level (extreme, hard, medium, easy)
        ground_truth: Optional ground truth answer
        model: LLM model to use for judging
        api_key: API key for LLM provider

    Returns:
        JudgeResult with scores and reasoning
    """
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError("openai package required for LLM judge. Install with: pip install openai") from None

    api_key = api_key or __import__("os").getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable or api_key parameter required")

    client = AsyncOpenAI(api_key=api_key)

    prompt = _build_judge_prompt(
        query=query,
        answer=answer,
        context=context,
        reasoning_type=reasoning_type,
        difficulty=difficulty,
        ground_truth=ground_truth,
    )

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,  # Low temperature for consistent evaluation
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Empty response from LLM judge")

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON from LLM judge: {content[:200]}") from None

    return JudgeResult(
        faithfulness=float(data.get("faithfulness", 0.0)),
        answer_relevancy=float(data.get("answer_relevancy", 0.0)),
        reasoning_quality=float(data.get("reasoning_quality", 0.0)),
        overall_score=float(data.get("overall_score", 0.0)),
        reasoning=str(data.get("reasoning", "")),
        errors=data.get("errors", []),
        suggestions=data.get("suggestions", []),
    )


def evaluate_case_offline(
    case: dict[str, Any],
    answer: str,
    context: str,
) -> CaseJudgeResult:
    """
    Perform offline heuristic evaluation without LLM.

    This uses rule-based heuristics as a fallback when LLM is not available.
    Much less accurate than LLM-as-Judge but gives some signal.
    """
    from eval.metrics import calculate_keyword_coverage, is_fallback_answer

    # Heuristic scores
    faithfulness = 0.0
    answer_relevancy = 0.0
    reasoning_quality = 0.0

    # Faithfulness heuristic: check if answer uses context keywords
    expected_keywords = case.get("expected_keywords", [])
    if expected_keywords:
        kw_coverage = calculate_keyword_coverage(answer, expected_keywords)
        faithfulness = kw_coverage
    else:
        faithfulness = 1.0 if is_fallback_answer(answer) else 0.8

    # Answer relevancy: did we get sources?
    if case.get("should_fallback") and is_fallback_answer(answer):
        answer_relevancy = 1.0
    elif context:
        answer_relevancy = 0.7  # Got context but can't verify quality
    else:
        answer_relevancy = 0.3

    # Reasoning quality: the offline path cannot judge reasoning — there is no
    # model reading the answer. Historically this was a constant keyed on the
    # case's difficulty LABEL (easy=0.85 whatever the answer), which fabricated
    # scores. Now it mirrors the only measured offline signal (keyword
    # grounding); the result is tagged judged_by="heuristic" and headline
    # aggregates split heuristic rows out.
    reasoning_quality = faithfulness

    overall = (faithfulness + answer_relevancy + reasoning_quality) / 3

    errors = []
    suggestions = []

    if faithfulness < 0.5:
        errors.append("Low keyword coverage from context")
        suggestions.append("Verify retrieved documents match query intent")

    if answer_relevancy < 0.5:
        errors.append("Answer may not address the question")
        suggestions.append("Review retrieval strategy")

    return CaseJudgeResult(
        case_id=case.get("id", "unknown"),
        query=case.get("query", ""),
        answer=answer,
        context=context,
        reasoning_type=case.get("reasoning_type", "general"),
        difficulty=case.get("difficulty", "medium"),
        faithfulness=faithfulness,
        answer_relevancy=answer_relevancy,
        reasoning_quality=reasoning_quality,
        overall_score=overall,
        judge_reasoning="Heuristic evaluation (LLM not available)",
        errors=errors,
        suggestions=suggestions,
        judged_by="heuristic",
    )


async def run_llm_judge_evaluation(
    dataset: list[dict[str, Any]],
    answers: dict[str, str],  # case_id -> answer
    contexts: dict[str, str],  # case_id -> context
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
) -> list[CaseJudgeResult]:
    """Run LLM-as-Judge on a batch of cases."""
    import asyncio

    results: list[CaseJudgeResult] = []

    async def evaluate_one(case: dict[str, Any]) -> CaseJudgeResult:
        case_id = case.get("id", "unknown")
        answer = answers.get(case_id, "")
        context = contexts.get(case_id, "")

        import time
        start = time.perf_counter()

        try:
            judge_result = await evaluate_with_llm_judge(
                query=case.get("query", ""),
                answer=answer,
                context=context,
                reasoning_type=case.get("reasoning_type", "general"),
                difficulty=case.get("difficulty", "medium"),
                ground_truth=case.get("ground_truth_answer"),
                model=model,
                api_key=api_key,
            )

            return CaseJudgeResult(
                case_id=case_id,
                query=case.get("query", ""),
                answer=answer,
                context=context,
                reasoning_type=case.get("reasoning_type", "general"),
                difficulty=case.get("difficulty", "medium"),
                faithfulness=judge_result.faithfulness,
                answer_relevancy=judge_result.answer_relevancy,
                reasoning_quality=judge_result.reasoning_quality,
                overall_score=judge_result.overall_score,
                judge_reasoning=judge_result.reasoning,
                errors=judge_result.errors,
                suggestions=judge_result.suggestions,
                model_used=model,
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception:
            # Fallback to offline evaluation
            return evaluate_case_offline(case, answer, context)

    # Run evaluations concurrently
    results = await asyncio.gather(*[evaluate_one(case) for case in dataset])

    return list(results)


def summarize_judge_results(results: list[CaseJudgeResult]) -> dict[str, Any]:
    """Summarize LLM judge results."""
    if not results:
        return {
            "total_cases": 0,
            "heuristic_cases": 0,
            "llm_cases": 0,
            "avg_overall_score_llm": 0.0,
            "pass_rate_llm": 0.0,
            "avg_faithfulness": 0.0,
            "avg_answer_relevancy": 0.0,
            "avg_reasoning_quality": 0.0,
            "avg_overall_score": 0.0,
            "pass_rate": 0.0,
        }

    # Filter out fallback cases for reasoning metrics
    reasoning_cases = [r for r in results if r.reasoning_type != "irrelevant"]

    total = len(results)
    reasoning_total = len(reasoning_cases)

    # Provenance split: heuristic rows must never silently blend into headline
    # judged aggregates. LLM-only figures are the citable ones; the blended
    # avgs below are kept for backward compat alongside explicit counts.
    heuristic_cases = [r for r in results if r.judged_by == "heuristic"]
    llm_cases = [r for r in results if r.judged_by != "heuristic"]
    n_heuristic = len(heuristic_cases)
    n_llm = len(llm_cases)
    avg_overall_score_llm = (
        sum(r.overall_score for r in llm_cases) / n_llm if n_llm else 0.0
    )
    pass_rate_llm = (
        sum(1 for r in llm_cases if r.overall_score >= 0.7) / n_llm if n_llm else 0.0
    )

    # Overall scores
    avg_faithfulness = sum(r.faithfulness for r in results) / total
    avg_answer_relevancy = sum(r.answer_relevancy for r in results) / total
    avg_reasoning_quality = sum(r.reasoning_quality for r in results) / total
    avg_overall = sum(r.overall_score for r in results) / total

    # Pass rate (overall >= 0.7)
    passed = sum(1 for r in results if r.overall_score >= 0.7)
    pass_rate = passed / total

    # Reasoning-specific scores
    reasoning_faithfulness = (
        sum(r.faithfulness for r in reasoning_cases) / reasoning_total
        if reasoning_total > 0 else 0.0
    )
    reasoning_quality_score = (
        sum(r.reasoning_quality for r in reasoning_cases) / reasoning_total
        if reasoning_total > 0 else 0.0
    )

    # Scores by difficulty
    by_difficulty: dict[str, dict[str, float]] = {}
    for difficulty in ["extreme", "hard", "medium", "easy"]:
        diff_cases = [r for r in results if r.difficulty == difficulty]
        if diff_cases:
            by_difficulty[difficulty] = {
                "count": len(diff_cases),
                "avg_score": sum(r.overall_score for r in diff_cases) / len(diff_cases),
                "faithfulness": sum(r.faithfulness for r in diff_cases) / len(diff_cases),
                "reasoning_quality": sum(r.reasoning_quality for r in diff_cases) / len(diff_cases),
            }

    # Scores by reasoning type
    by_reasoning_type: dict[str, dict[str, float]] = {}
    for rtype in set(r.reasoning_type for r in reasoning_cases):
        type_cases = [r for r in reasoning_cases if r.reasoning_type == rtype]
        if type_cases:
            by_reasoning_type[rtype] = {
                "count": len(type_cases),
                "avg_score": sum(r.overall_score for r in type_cases) / len(type_cases),
            }

    return {
        "total_cases": total,
        "reasoning_cases": reasoning_total,
        "heuristic_cases": n_heuristic,
        "llm_cases": n_llm,
        "avg_overall_score_llm": avg_overall_score_llm,
        "pass_rate_llm": pass_rate_llm,
        "avg_faithfulness": avg_faithfulness,
        "avg_answer_relevancy": avg_answer_relevancy,
        "avg_reasoning_quality": avg_reasoning_quality,
        "avg_overall_score": avg_overall,
        "pass_rate": pass_rate,
        "passed_count": passed,
        "reasoning_faithfulness": reasoning_faithfulness,
        "reasoning_quality_score": reasoning_quality_score,
        "by_difficulty": by_difficulty,
        "by_reasoning_type": by_reasoning_type,
    }


def generate_judge_report(results: list[CaseJudgeResult], summary: dict[str, Any]) -> str:
    """Generate a markdown report from judge results."""
    lines = [
        "# LLM-as-Judge Evaluation Report",
        "",
        f"**Total Cases:** {summary['total_cases']}",
        f"**Reasoning Cases:** {summary['reasoning_cases']}",
        f"**Pass Rate:** {summary['pass_rate']:.1%} ({summary['passed_count']}/{summary['total_cases']})",
        "",
        "## Summary Scores",
        "",
        "| Metric | Score |",
        "|--------|-------|",
        f"| Faithfulness | {summary['avg_faithfulness']:.1%} |",
        f"| Answer Relevancy | {summary['avg_answer_relevancy']:.1%} |",
        f"| Reasoning Quality | {summary['avg_reasoning_quality']:.1%} |",
        f"| **Overall** | **{summary['avg_overall_score']:.1%}** |",
        "",
        "## Scores by Difficulty",
        "",
        "| Difficulty | Count | Overall | Faithfulness | Reasoning |",
        "|------------|-------|--------|--------------|-----------|",
    ]

    for diff, scores in summary.get("by_difficulty", {}).items():
        lines.append(
            f"| {diff.capitalize()} | {scores['count']} | "
            f"{scores['avg_score']:.1%} | {scores['faithfulness']:.1%} | "
            f"{scores['reasoning_quality']:.1%} |"
        )

    lines.extend(["", "## Scores by Reasoning Type", ""])

    for rtype, scores in summary.get("by_reasoning_type", {}).items():
        lines.append(
            f"- **{rtype}**: {scores['avg_score']:.1%} ({scores['count']} cases)"
        )

    lines.extend(["", "## Detailed Results", ""])

    for result in results:
        status = "✅" if result.overall_score >= 0.7 else "❌"
        lines.extend([
            "",
            f"### {status} {result.case_id}",
            "",
            f"**Query:** {result.query[:100]}...",
            "",
            "| Metric | Score |",
            "|--------|-------|",
            f"| Faithfulness | {result.faithfulness:.1%} |",
            f"| Answer Relevancy | {result.answer_relevancy:.1%} |",
            f"| Reasoning Quality | {result.reasoning_quality:.1%} |",
            f"| **Overall** | **{result.overall_score:.1%}** |",
            "",
            f"**Judge Reasoning:** {result.judge_reasoning}",
        ])

        if result.errors:
            lines.append(f"**Errors:** {', '.join(result.errors)}")
        if result.suggestions:
            lines.append(f"**Suggestions:** {', '.join(result.suggestions)}")

    return "\n".join(lines)
