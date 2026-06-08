from functools import lru_cache
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from app.models import JobPosting, Resume
from app.services.groq_client import GroqService, LLMCompletion
from app.services.retrieval import RetrievedChunk, retrieve_relevant_chunks
from app.services.scorecards import build_fit_scorecard
from app.services.structured import extract_structured_profile


class AnalysisState(TypedDict, total=False):
    db: Session
    groq_service: GroqService
    resume: Resume
    job: JobPosting
    query: str
    chunks: list[RetrievedChunk]
    structured_profile: dict
    structured_completion: LLMCompletion
    scorecard: dict
    analysis: str
    analysis_completion: LLMCompletion


def _retrieve_node(state: AnalysisState) -> AnalysisState:
    db = state["db"]
    job = state["job"]
    query = state.get("query") or f"Analyze fit between resume and {job.title}"
    chunks = retrieve_relevant_chunks(db, query, job.id)
    return {"chunks": chunks}


def _structured_node(state: AnalysisState) -> AnalysisState:
    groq_service = state["groq_service"]
    resume = state["resume"]
    job = state["job"]
    chunks = state.get("chunks", [])
    structured, completion = extract_structured_profile(groq_service, resume, job, chunks)
    return {
        "structured_profile": structured,
        "structured_completion": completion,
    }


def _scorecard_node(state: AnalysisState) -> AnalysisState:
    structured = state.get("structured_profile", {})
    chunks = state.get("chunks", [])
    return {"scorecard": build_fit_scorecard(structured, chunks)}


def _analysis_node(state: AnalysisState) -> AnalysisState:
    groq_service = state["groq_service"]
    resume = state["resume"]
    job = state["job"]
    chunks = state.get("chunks", [])
    structured = state.get("structured_profile", {})
    scorecard = state.get("scorecard", {})

    context = "\n\n".join(
        f"[{chunk.evidence_id} | {chunk.document_label} | score={chunk.fused_score:.3f}]\n{chunk.content}"
        for chunk in chunks
    )
    prompt = f"""
Write a career-fit report using only the evidence below.

Resume filename: {resume.filename}
Job title: {job.title}
Job source: {job.source_name}

Structured extraction JSON:
{structured}

Scorecard JSON:
{scorecard}

Evidence context:
{context}

Write sections:
1. Executive fit summary
2. Strengths mapped to role needs
3. Critical gaps and risk flags
4. Interview prep strategy
5. Resume rewrite priorities
6. Evidence citations (use evidence IDs like E1, E2)

Rules:
- Do not make claims that are not supported by the evidence.
- If the evidence is incomplete, say so clearly.
""".strip()

    completion = groq_service.complete(prompt=prompt, fast=False)
    return {"analysis": completion.text, "analysis_completion": completion}


@lru_cache(maxsize=1)
def _build_graph():
    graph = StateGraph(AnalysisState)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("extract_structured", _structured_node)
    graph.add_node("scorecard", _scorecard_node)
    graph.add_node("compose_analysis", _analysis_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "extract_structured")
    graph.add_edge("extract_structured", "scorecard")
    graph.add_edge("scorecard", "compose_analysis")
    graph.add_edge("compose_analysis", END)
    return graph.compile()


def run_analysis_graph(
    db: Session,
    groq_service: GroqService,
    resume: Resume,
    job: JobPosting,
) -> AnalysisState:
    graph = _build_graph()
    return graph.invoke(
        {
            "db": db,
            "groq_service": groq_service,
            "resume": resume,
            "job": job,
            "query": f"Analyze fit between resume and {job.title}",
        }
    )
