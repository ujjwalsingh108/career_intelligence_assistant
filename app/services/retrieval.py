from dataclasses import dataclass

from sqlalchemy import delete, or_, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import DocumentChunk, JobPosting, Resume
from app.services.chunking import chunk_text
from app.services.embeddings import embed_text, embed_texts


settings = get_settings()


@dataclass
class RetrievedChunk:
    document_label: str
    content: str


def replace_resume(session: Session, filename: str, raw_text: str) -> Resume:
    session.execute(delete(DocumentChunk).where(DocumentChunk.resume_id.is_not(None)))
    session.execute(delete(Resume))

    resume = Resume(filename=filename, raw_text=raw_text)
    session.add(resume)
    session.flush()

    chunks = chunk_text(raw_text, settings.chunk_size, settings.chunk_overlap)
    embeddings = embed_texts(chunks)
    for index, (content, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
        session.add(
            DocumentChunk(
                document_label=f"Resume: {filename}",
                chunk_index=index,
                content=content,
                resume_id=resume.id,
                embedding=embedding,
            )
        )

    session.commit()
    session.refresh(resume)
    return resume


def add_job_posting(session: Session, title: str, source_name: str, raw_text: str) -> JobPosting:
    job = JobPosting(title=title, source_name=source_name, raw_text=raw_text)
    session.add(job)
    session.flush()

    chunks = chunk_text(raw_text, settings.chunk_size, settings.chunk_overlap)
    embeddings = embed_texts(chunks)
    for index, (content, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
        session.add(
            DocumentChunk(
                document_label=f"Job: {title}",
                chunk_index=index,
                content=content,
                job_posting_id=job.id,
                embedding=embedding,
            )
        )

    session.commit()
    session.refresh(job)
    return job


def list_jobs(session: Session) -> list[JobPosting]:
    statement = select(JobPosting).order_by(JobPosting.created_at.desc(), JobPosting.id.desc())
    return list(session.scalars(statement))


def get_latest_resume(session: Session) -> Resume | None:
    statement = select(Resume).order_by(Resume.created_at.desc(), Resume.id.desc()).limit(1)
    return session.scalar(statement)


def get_job(session: Session, job_id: int) -> JobPosting | None:
    return session.get(JobPosting, job_id)


def retrieve_relevant_chunks(session: Session, question: str, job_id: int | None) -> list[RetrievedChunk]:
    embedding = embed_text(question)
    filters = [DocumentChunk.resume_id.is_not(None)]
    if job_id is not None:
        filters.append(DocumentChunk.job_posting_id == job_id)

    statement = (
        select(DocumentChunk)
        .where(or_(*filters))
        .order_by(DocumentChunk.embedding.cosine_distance(embedding))
        .limit(settings.max_context_chunks)
    )

    return [
        RetrievedChunk(document_label=chunk.document_label, content=chunk.content)
        for chunk in session.scalars(statement)
    ]
