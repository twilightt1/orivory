#!/usr/bin/env python3
"""Run LLM-as-Judge evaluation with heuristics fallback."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from eval.llm_judge import (
    evaluate_case_offline,
    generate_judge_report,
    summarize_judge_results,
)

# Load dataset
dataset = json.loads(Path('eval/orivory_extreme_eval_dataset.json').read_text())
offline_results = json.loads(Path('eval/extreme_results/latest_report.json').read_text())

# Build answers dict
answers = {}
contexts = {}
docs = {p.name: p.read_text() for p in Path('sample_docs').glob('*.md')}

for result in offline_results.get('results', []):
    case_id = result['id']
    answers[case_id] = result.get('answer', '')
    sources = result.get('returned_sources', [])
    if sources:
        context_parts = []
        for src in sources[:3]:
            if src in docs:
                context_parts.append(f'=== {src} ===\n{docs[src][:500]}')
        contexts[case_id] = '\n\n'.join(context_parts)

# Run heuristic evaluation
print('Running heuristic evaluation (no LLM needed)...')
print()

results = []
for case in dataset:
    case_id = case.get('id', 'unknown')
    result = evaluate_case_offline(
        case,
        answers.get(case_id, ''),
        contexts.get(case_id, '')
    )
    results.append(result)

summary = summarize_judge_results(results)

print('=' * 60)
print('LLM-AS-JUDGE EVALUATION (Heuristic Fallback)')
print('=' * 60)
print()
print(f'Total Cases: {summary["total_cases"]}')
print(f'Reasoning Cases: {summary["reasoning_cases"]}')
print()
print(f'Faithfulness:        {summary["avg_faithfulness"]:.1%}')
print(f'Answer Relevancy:    {summary["avg_answer_relevancy"]:.1%}')
print(f'Reasoning Quality:   {summary["avg_reasoning_quality"]:.1%}')
print(f'Overall Score:       {summary["avg_overall_score"]:.1%}')
print(f'Pass Rate:           {summary["pass_rate"]:.1%}')
print()

print('By Difficulty:')
for diff, scores in summary.get('by_difficulty', {}).items():
    print(f'  {diff.capitalize()}: {scores["avg_score"]:.1%} (n={scores["count"]})')

print()
print('By Reasoning Type:')
for rtype, scores in sorted(summary.get('by_reasoning_type', {}).items()):
    print(f'  {rtype}: {scores["avg_score"]:.1%}')

# Save results
output_dir = Path('eval/extreme_results')
output_dir.mkdir(parents=True, exist_ok=True)

report_md = generate_judge_report(results, summary)
(output_dir / 'judge_report.md').write_text(report_md, encoding='utf-8')

judge_results = {
    'summary': summary,
    'results': [vars(r) for r in results]
}
(output_dir / 'judge_results.json').write_text(
    json.dumps(judge_results, indent=2, default=str),
    encoding='utf-8'
)

print()
print(f'Report saved to: {output_dir / "judge_report.md"}')
