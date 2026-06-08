import argparse
import json
from datetime import datetime, UTC
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from prompt_eval import evaluate_generated_output, summarize_results


def load_cases(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_payload(results) -> list[dict]:
    payload: list[dict] = []
    for result in results:
        payload.append(
            {
                "id": result.case_id,
                "total_bullets": result.total_bullets,
                "grounded_bullets": result.grounded_bullets,
                "grounding_coverage": result.grounding_coverage,
                "hallucination_rate": result.hallucination_rate,
                "section_coverage": result.section_coverage,
                "keyword_coverage": result.keyword_coverage,
                "quality_score": result.quality_score,
                "passed": result.passed,
                "reasons": result.reasons,
            }
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generated prompt outputs for grounding coverage, hallucination rate, and overall quality."
        )
    )
    parser.add_argument("--cases", type=Path, default=Path("evals/prompt_eval_cases.json"))
    parser.add_argument("--report", type=Path, default=Path("evals/prompt_eval_report.json"))
    args = parser.parse_args()

    cases = load_cases(args.cases)
    results = [evaluate_generated_output(case) for case in cases]
    summary = summarize_results(results)

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": summary,
        "results": to_payload(results),
    }

    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Total cases: {summary['total']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Average grounding coverage: {summary['avg_grounding_coverage']}")
    print(f"Average hallucination rate: {summary['avg_hallucination_rate']}")
    print(f"Average quality score: {summary['avg_quality_score']}")
    print(f"Report written to: {args.report}")

    if summary["failed"] > 0:
        print("Failures:")
        for result in results:
            if result.passed:
                continue
            reasons = "; ".join(result.reasons)
            print(f"- {result.case_id}: {reasons}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
