import json
import re

from app.models import JobPosting, Resume
from app.services.groq_client import GroqService, LLMCompletion
from app.services.retrieval import RetrievedChunk


def _extract_json_block(text: str) -> dict:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def extract_structured_profile(
    groq_service: GroqService,
    resume: Resume,
    job: JobPosting,
    chunks: list[RetrievedChunk],
) -> tuple[dict, LLMCompletion]:
    context = "\n\n".join(f"[{chunk.document_label}]\n{chunk.content}" for chunk in chunks)
    prompt = f"""
Extract a strict JSON object from the resume and job posting.

Required JSON keys:
- resume_skills: string[]
- resume_domains: string[]
- resume_years_experience: number
- job_skills: string[]
- job_must_have: string[]
- job_domains: string[]
- job_years_required: number

Rules:
- Output JSON only.
- Use empty arrays or 0 if not present.
- Keep skill names concise and normalized.

Resume text:
{resume.raw_text}

Job description:
{job.raw_text}

Retrieved evidence:
{context}
""".strip()

    completion = groq_service.complete(prompt=prompt, fast=False)
    parsed = _extract_json_block(completion.text)
    if not parsed:
        parsed = {
            "resume_skills": [],
            "resume_domains": [],
            "resume_years_experience": 0,
            "job_skills": [],
            "job_must_have": [],
            "job_domains": [],
            "job_years_required": 0,
        }
    return parsed, completion
