from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db, initialize_database
from app.services.analysis import answer_user_question, build_job_summary
from app.services.documents import extract_text
from app.services.groq_client import GroqService
from app.services.retrieval import add_job_posting, get_job, get_latest_resume, list_jobs, replace_resume, retrieve_relevant_chunks


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def build_context(
    request: Request,
    db: Session,
    *,
    selected_job_id: int | None = None,
    analysis: str | None = None,
    answer: str | None = None,
    error: str | None = None,
) -> dict:
    jobs = list_jobs(db)
    resume = get_latest_resume(db)
    selected_job = get_job(db, selected_job_id) if selected_job_id else None
    return {
        "request": request,
        "resume": resume,
        "jobs": jobs,
        "selected_job": selected_job,
        "analysis": analysis,
        "answer": answer,
        "error": error,
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request, job_id: int | None = None, db: Session = Depends(get_db)) -> HTMLResponse:
    return templates.TemplateResponse("index.html", build_context(request, db, selected_job_id=job_id))


@app.post("/resume/upload", response_class=HTMLResponse)
async def upload_resume(
    request: Request,
    resume_file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    try:
        filename, text = await extract_text(resume_file)
        replace_resume(db, filename, text)
        return templates.TemplateResponse(
            "index.html",
            build_context(request, db, answer="Resume indexed successfully."),
        )
    except HTTPException as exc:
        return templates.TemplateResponse(
            "index.html",
            build_context(request, db, error=exc.detail),
            status_code=exc.status_code,
        )


@app.post("/jobs/upload", response_class=HTMLResponse)
async def upload_jobs(
    request: Request,
    title: str = Form(""),
    pasted_job_description: str = Form(""),
    job_files: list[UploadFile] = File(default=[]),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    try:
        created = 0
        cleaned_title = title.strip()
        cleaned_text = pasted_job_description.strip()
        if cleaned_text:
            add_job_posting(
                db,
                title=cleaned_title or "Pasted Job Description",
                source_name="pasted-input",
                raw_text=cleaned_text,
            )
            created += 1

        for upload in job_files:
            filename, text = await extract_text(upload)
            inferred_title = Path(filename).stem.replace("_", " ").replace("-", " ").strip().title()
            add_job_posting(db, title=inferred_title or filename, source_name=filename, raw_text=text)
            created += 1

        if created == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Provide a job description or at least one file.")

        return templates.TemplateResponse(
            "index.html",
            build_context(request, db, answer=f"Indexed {created} job posting(s)."),
        )
    except HTTPException as exc:
        return templates.TemplateResponse(
            "index.html",
            build_context(request, db, error=exc.detail),
            status_code=exc.status_code,
        )


@app.post("/jobs/{job_id}/analyze", response_class=HTMLResponse)
def analyze_job(job_id: int, request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    resume = get_latest_resume(db)
    job = get_job(db, job_id)
    if not resume or not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resume or job posting not found.")

    try:
        groq_service = GroqService()
        chunks = retrieve_relevant_chunks(db, f"Analyze fit between resume and {job.title}", job_id)
        analysis = build_job_summary(groq_service, resume, job, chunks)
        return templates.TemplateResponse(
            "index.html",
            build_context(request, db, selected_job_id=job_id, analysis=analysis),
        )
    except RuntimeError as exc:
        return templates.TemplateResponse(
            "index.html",
            build_context(request, db, selected_job_id=job_id, error=str(exc)),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


@app.post("/ask", response_class=HTMLResponse)
def ask_question(
    request: Request,
    question: str = Form(...),
    job_id: int | None = Form(default=None),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    resume = get_latest_resume(db)
    if not resume:
        return templates.TemplateResponse(
            "index.html",
            build_context(request, db, error="Upload a resume before asking questions."),
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    job = get_job(db, job_id) if job_id else None

    try:
        groq_service = GroqService()
        chunks = retrieve_relevant_chunks(db, question, job_id)
        answer = answer_user_question(groq_service, question, resume, job, chunks)
        return templates.TemplateResponse(
            "index.html",
            build_context(request, db, selected_job_id=job_id, answer=answer),
        )
    except RuntimeError as exc:
        return templates.TemplateResponse(
            "index.html",
            build_context(request, db, selected_job_id=job_id, error=str(exc)),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset() -> RedirectResponse:
    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
