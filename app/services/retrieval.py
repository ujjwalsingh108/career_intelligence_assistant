from dataclasses import dataclass
import re

from sqlalchemy import delete, or_, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import DocumentChunk, JobPosting, Resume
from app.services.chunking import chunk_text
from app.services.embeddings import embed_text, embed_texts


settings = get_settings()


@dataclass
class RetrievedChunk:
    evidence_id: str
    document_label: str
    content: str
    fused_score: float
    vector_score: float
    lexical_score: float


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9+#\.]{2,}", text.lower())
    return set(tokens)


def _lexical_score(query: str, content: str) -> float:
    query_tokens = _tokenize(query)
    content_tokens = _tokenize(content)
    if not query_tokens or not content_tokens:
        return 0.0
    overlap = len(query_tokens & content_tokens)
    return overlap / len(query_tokens)


def _rerank_score(query: str, chunk: RetrievedChunk) -> float:
    query_tokens = _tokenize(query)
    content_tokens = _tokenize(chunk.content)
    if not query_tokens:
        return chunk.fused_score
    coverage = len(query_tokens & content_tokens) / len(query_tokens)
    return (chunk.fused_score * 0.85) + (coverage * 0.15)


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

    distance = DocumentChunk.embedding.cosine_distance(embedding).label("distance")

    statement = (
        select(DocumentChunk, distance)
        .where(or_(*filters))
        .order_by(distance)
        .limit(settings.retrieval_candidate_pool)
    )
    rows = list(session.execute(statement))

    weighted: list[RetrievedChunk] = []
    for row in rows:
        chunk = row[0]
        chunk_distance = float(row[1] or 1.0)
        vector_score = max(0.0, 1.0 - min(chunk_distance, 2.0) / 2.0)
        lexical_score = _lexical_score(question, chunk.content)
        fused_score = (vector_score * settings.hybrid_vector_weight) + (lexical_score * settings.hybrid_lexical_weight)
        weighted.append(
            RetrievedChunk(
                evidence_id="",
                document_label=chunk.document_label,
                content=chunk.content,
                fused_score=fused_score,
                vector_score=vector_score,
                lexical_score=lexical_score,
            )
        )

    weighted.sort(key=lambda item: item.fused_score, reverse=True)
    reranked = sorted(weighted[: settings.rerank_top_k], key=lambda item: _rerank_score(question, item), reverse=True)
    trimmed = reranked[: settings.max_context_chunks]

    output: list[RetrievedChunk] = []
    for index, chunk in enumerate(trimmed, start=1):
        output.append(
            RetrievedChunk(
                evidence_id=f"E{index}",
                document_label=chunk.document_label,
                content=chunk.content,
                fused_score=chunk.fused_score,
                vector_score=chunk.vector_score,
                lexical_score=chunk.lexical_score,
            )
        )
    return output
