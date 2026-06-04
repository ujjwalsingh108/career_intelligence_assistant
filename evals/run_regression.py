import argparse
import json
from pathlib import Path

from app.services.scorecards import build_fit_scorecard


def load_cases(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_cases(cases: list[dict]) -> dict:
    passed = 0
    failed = 0
    failures: list[str] = []

    for case in cases:
        structured = {
            "resume_skills": case["resume_skills"],
            "resume_domains": case["resume_domains"],
            "resume_years_experience": case["resume_years"],
            "job_skills": case["job_skills"],
            "job_must_have": case["job_must_have"],
            "job_domains": case["job_domains"],
            "job_years_required": case["job_years"],
        }
        scorecard = build_fit_scorecard(structured, chunks=[])
        missing = {item.lower() for item in scorecard["missing_required"]}
        expected_missing = {item.lower() for item in case["expected_missing"]}

        meets_missing = expected_missing.issuperset(missing) or missing.issuperset(expected_missing)
        meets_fit = scorecard["fit_score"] >= float(case["fit_min"])

        if meets_missing and meets_fit:
            passed += 1
            continue

        failed += 1
        failures.append(
            (
                f"{case['id']} failed: fit={scorecard['fit_score']} (expected >= {case['fit_min']}), "
                f"missing={sorted(missing)} expected~={sorted(expected_missing)}"
            )
        )

    return {"passed": passed, "failed": failed, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic scorecard regression checks.")
    parser.add_argument("--cases", type=Path, default=Path("evals/cases.json"))
    args = parser.parse_args()

    cases = load_cases(args.cases)
    result = run_cases(cases)

    print(f"Total cases: {len(cases)}")
    print(f"Passed: {result['passed']}")
    print(f"Failed: {result['failed']}")
    if result["failures"]:
        print("Failures:")
        for failure in result["failures"]:
            print(f"- {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
