import json
from pathlib import Path

from app.services.scorecards import build_fit_scorecard


def test_eval_cases_count() -> None:
    cases_path = Path("evals/cases.json")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    assert len(cases) >= 20


def test_scorecard_detects_missing_must_have() -> None:
    structured = {
        "resume_skills": ["python", "sql"],
        "resume_domains": ["fintech"],
        "resume_years_experience": 3,
        "job_skills": ["python", "docker"],
        "job_must_have": ["python", "kubernetes"],
        "job_domains": ["fintech"],
        "job_years_required": 2,
    }
    scorecard = build_fit_scorecard(structured, chunks=[])
    assert "kubernetes" in scorecard["missing_required"]
    assert scorecard["fit_score"] > 0
