from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass
class PromptEvalResult:
    case_id: str
    total_bullets: int
    grounded_bullets: int
    grounding_coverage: float
    hallucination_rate: float
    section_coverage: float
    keyword_coverage: float
    quality_score: float
    passed: bool
    reasons: list[str]


BULLET_PATTERN = re.compile(r"^\s*[-*]\s+", re.MULTILINE)
CITATION_PATTERN = re.compile(r"\[E\d+(?:\s*,\s*E\d+)*\]")


def _extract_bullets(text: str) -> list[str]:
    bullet_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if BULLET_PATTERN.match(stripped):
            bullet_lines.append(stripped)
    return bullet_lines


def _normalize_set(items: list[str] | None) -> set[str]:
    if not items:
        return set()
    return {item.strip().lower() for item in items if item.strip()}


def evaluate_generated_output(case: dict) -> PromptEvalResult:
    case_id = str(case["id"])
    output = str(case["generated_output"])
    expected = case.get("expected", {})

    required_sections = case.get("required_sections", [])
    required_keywords = _normalize_set(case.get("required_keywords", []))
    forbidden_terms = _normalize_set(case.get("forbidden_terms", []))

    bullets = _extract_bullets(output)
    total_bullets = len(bullets)
    grounded_bullets = 0
    hallucinated_bullets = 0

    lowered_output = output.lower()

    for bullet in bullets:
        has_citation = bool(CITATION_PATTERN.search(bullet))
        if has_citation:
            grounded_bullets += 1

        lowered_bullet = bullet.lower()
        forbidden_hit = any(term in lowered_bullet for term in forbidden_terms)
        if (not has_citation) or forbidden_hit:
            hallucinated_bullets += 1

    grounding_coverage = grounded_bullets / total_bullets if total_bullets else 0.0
    hallucination_rate = hallucinated_bullets / total_bullets if total_bullets else 1.0

    found_sections = 0
    for section in required_sections:
        if section.lower() in lowered_output:
            found_sections += 1
    section_coverage = found_sections / len(required_sections) if required_sections else 1.0

    keyword_hits = 0
    for keyword in required_keywords:
        if keyword in lowered_output:
            keyword_hits += 1
    keyword_coverage = keyword_hits / len(required_keywords) if required_keywords else 1.0

    quality_score = round((section_coverage * 40.0) + (grounding_coverage * 40.0) + (keyword_coverage * 20.0), 2)

    min_grounding_coverage = float(expected.get("min_grounding_coverage", 0.0))
    max_hallucination_rate = float(expected.get("max_hallucination_rate", 1.0))
    min_quality_score = float(expected.get("min_quality_score", 0.0))

    reasons: list[str] = []
    if grounding_coverage < min_grounding_coverage:
        reasons.append(
            f"grounding_coverage={grounding_coverage:.2f} below minimum {min_grounding_coverage:.2f}"
        )
    if hallucination_rate > max_hallucination_rate:
        reasons.append(
            f"hallucination_rate={hallucination_rate:.2f} above maximum {max_hallucination_rate:.2f}"
        )
    if quality_score < min_quality_score:
        reasons.append(f"quality_score={quality_score:.2f} below minimum {min_quality_score:.2f}")

    passed = len(reasons) == 0

    return PromptEvalResult(
        case_id=case_id,
        total_bullets=total_bullets,
        grounded_bullets=grounded_bullets,
        grounding_coverage=round(grounding_coverage, 4),
        hallucination_rate=round(hallucination_rate, 4),
        section_coverage=round(section_coverage, 4),
        keyword_coverage=round(keyword_coverage, 4),
        quality_score=quality_score,
        passed=passed,
        reasons=reasons,
    )


def summarize_results(results: list[PromptEvalResult]) -> dict:
    case_count = len(results)
    passed = sum(1 for result in results if result.passed)
    failed = case_count - passed

    avg_grounding = round(sum(result.grounding_coverage for result in results) / case_count, 4) if case_count else 0.0
    avg_hallucination = round(sum(result.hallucination_rate for result in results) / case_count, 4) if case_count else 0.0
    avg_quality = round(sum(result.quality_score for result in results) / case_count, 2) if case_count else 0.0

    return {
        "total": case_count,
        "passed": passed,
        "failed": failed,
        "avg_grounding_coverage": avg_grounding,
        "avg_hallucination_rate": avg_hallucination,
        "avg_quality_score": avg_quality,
    }
