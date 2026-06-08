from evals.prompt_eval import evaluate_generated_output, summarize_results


def test_prompt_eval_detects_grounding_failures() -> None:
    case = {
        "id": "bad-case",
        "required_sections": ["Professional Summary"],
        "required_keywords": ["python"],
        "forbidden_terms": ["google"],
        "generated_output": "## Professional Summary\n- Worked at Google and built everything",
        "expected": {
            "min_grounding_coverage": 1.0,
            "max_hallucination_rate": 0.0,
            "min_quality_score": 90,
        },
    }

    result = evaluate_generated_output(case)

    assert not result.passed
    assert result.grounding_coverage == 0.0
    assert result.hallucination_rate == 1.0
    assert any("grounding_coverage" in reason for reason in result.reasons)


def test_prompt_eval_summary_aggregates_metrics() -> None:
    passed_case = {
        "id": "good-case",
        "required_sections": ["Professional Summary"],
        "required_keywords": ["python"],
        "generated_output": "## Professional Summary\n- Built Python APIs [E1]",
        "expected": {
            "min_grounding_coverage": 0.9,
            "max_hallucination_rate": 0.1,
            "min_quality_score": 60,
        },
    }
    failed_case = {
        "id": "bad-case",
        "required_sections": ["Professional Summary"],
        "required_keywords": ["python"],
        "generated_output": "## Summary\n- Generic work",
        "expected": {
            "min_grounding_coverage": 0.9,
            "max_hallucination_rate": 0.1,
            "min_quality_score": 60,
        },
    }

    results = [evaluate_generated_output(passed_case), evaluate_generated_output(failed_case)]
    summary = summarize_results(results)

    assert summary["total"] == 2
    assert summary["passed"] == 1
    assert summary["failed"] == 1
