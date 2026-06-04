from app.services.retrieval import RetrievedChunk


def _normalize_list(values: list[str] | None) -> set[str]:
    if not values:
        return set()
    normalized = set()
    for value in values:
        cleaned = value.strip().lower()
        if cleaned:
            normalized.add(cleaned)
    return normalized


def build_fit_scorecard(structured: dict, chunks: list[RetrievedChunk]) -> dict:
    resume_skills = _normalize_list(structured.get("resume_skills"))
    resume_domains = _normalize_list(structured.get("resume_domains"))
    job_skills = _normalize_list(structured.get("job_skills"))
    job_must_have = _normalize_list(structured.get("job_must_have"))
    job_domains = _normalize_list(structured.get("job_domains"))

    resume_years = float(structured.get("resume_years_experience") or 0)
    job_years = float(structured.get("job_years_required") or 0)

    required_pool = job_skills | job_must_have
    matched_required = sorted(job_must_have & resume_skills)
    missing_required = sorted(job_must_have - resume_skills)
    matched_skills = sorted((job_skills & resume_skills) - set(matched_required))

    skill_coverage = (len(required_pool & resume_skills) / max(1, len(required_pool)))
    years_ratio = min(1.0, (resume_years / job_years)) if job_years > 0 else 1.0
    domain_alignment = (len(job_domains & resume_domains) / max(1, len(job_domains)))

    fit_score = round(((skill_coverage * 0.60) + (years_ratio * 0.25) + (domain_alignment * 0.15)) * 100, 1)

    confidence_raw = 0.0
    if chunks:
        confidence_raw = sum(chunk.fused_score for chunk in chunks[:4]) / min(4, len(chunks))
    confidence = round(max(35.0, min(99.0, confidence_raw * 100)), 1)

    evidence_chips = [
        {
            "id": chunk.evidence_id,
            "source": chunk.document_label,
            "score": round(chunk.fused_score, 3),
            "text": chunk.content[:180].strip(),
        }
        for chunk in chunks[:5]
    ]

    return {
        "fit_score": fit_score,
        "confidence": confidence,
        "matched_required": matched_required,
        "missing_required": missing_required,
        "matched_skills": matched_skills,
        "years_alignment": {
            "resume_years": resume_years,
            "required_years": job_years,
            "ratio": round(years_ratio, 3),
        },
        "domain_overlap": sorted(job_domains & resume_domains),
        "evidence_chips": evidence_chips,
    }
