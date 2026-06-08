from app.models import JobPosting, Resume
from app.services.groq_client import GroqService, LLMCompletion
from app.services.retrieval import RetrievedChunk


def _format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No retrieval context available."
    return "\n\n".join(
        f"[{chunk.evidence_id} | {chunk.document_label} | score={chunk.fused_score:.3f}]\n{chunk.content}"
        for chunk in chunks
    )


def generate_tailored_resume(
    groq_service: GroqService,
    resume: Resume,
    job: JobPosting,
    chunks: list[RetrievedChunk],
    scorecard: dict,
    structured_profile: dict,
    analysis: str,
) -> LLMCompletion:
    prompt = f"""
Rewrite the resume for this role using only the evidence below.

Resume filename: {resume.filename}
Job title: {job.title}
Job source: {job.source_name}

Current resume text:
{resume.raw_text}

Job description:
{job.raw_text}

Structured extraction JSON:
{structured_profile}

Scorecard JSON:
{scorecard}

Analysis report:
{analysis}

Retrieved evidence:
{_format_context(chunks)}

Task:
Create a section-wise, ATS-aware tailored resume draft as markdown with these sections in order:
1. Professional Summary (3-4 lines)
2. Core Skills (10-14 bullets)
3. Experience Bullet Rewrites (at least 8 bullets, grouped by likely role scope)
4. Project/Impact Highlights (3-5 bullets)
5. Keyword Coverage Notes (matched vs missing)

Constraints:
- Every bullet must include at least one evidence ID, formatted like [E1].
- Do not invent employers, timelines, technologies, or outcomes.
- If evidence is missing for a useful point, put it under a "Needs evidence" subsection.
- Keep the language specific and concise.
""".strip()
    return groq_service.complete(prompt=prompt, fast=False)