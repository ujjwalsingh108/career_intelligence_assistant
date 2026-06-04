from app.models import JobPosting, Resume
from app.services.groq_client import GroqService
from app.services.retrieval import RetrievedChunk


def _format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No retrieval context available."
    return "\n\n".join(f"[{chunk.document_label}]\n{chunk.content}" for chunk in chunks)


def build_job_summary(
    groq_service: GroqService,
    resume: Resume,
    job: JobPosting,
    chunks: list[RetrievedChunk],
) -> str:
    prompt = f"""
Resume filename: {resume.filename}
Job title: {job.title}
Job source: {job.source_name}

Resume text:
{resume.raw_text}

Job description:
{job.raw_text}

Retrieved evidence:
{_format_context(chunks)}

Write a concise but detailed assessment with these sections:
1. Fit summary
2. Strengths aligned to the job
3. Gaps or missing evidence
4. Interview preparation areas
5. Resume improvements to target this role

Every section must be grounded in the provided evidence. Do not invent qualifications.
""".strip()
    return groq_service.complete(prompt=prompt, fast=False)


def answer_user_question(
    groq_service: GroqService,
    question: str,
    resume: Resume,
    job: JobPosting | None,
    chunks: list[RetrievedChunk],
) -> str:
    job_header = f"Job title: {job.title}\nJob description:\n{job.raw_text}" if job else "No job selected."
    prompt = f"""
Question: {question}

Resume filename: {resume.filename}
Resume text:
{resume.raw_text}

{job_header}

Retrieved evidence:
{_format_context(chunks)}

Answer the question using only the available evidence. If evidence is incomplete, say what is missing.
Include a short evidence section at the end citing the most relevant snippets.
""".strip()
    return groq_service.complete(prompt=prompt, fast=True)
