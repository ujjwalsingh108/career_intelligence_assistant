from types import SimpleNamespace

from fastapi.testclient import TestClient

import app.main as main_module
from app.services.groq_client import LLMCompletion


class _FakeSession:
    def add(self, _obj) -> None:
        pass

    def commit(self) -> None:
        pass


def test_tailor_resume_route_returns_tailored_output(monkeypatch) -> None:
    monkeypatch.setattr("app.main.initialize_database", lambda: None)
    monkeypatch.setattr("app.main.get_latest_resume", lambda _db: SimpleNamespace(filename="resume.txt", raw_text="Python backend work"))
    monkeypatch.setattr(
        "app.main.get_job",
        lambda _db, _job_id: SimpleNamespace(id=1, title="Backend Engineer", source_name="job.txt", raw_text="Need Python and APIs"),
    )
    monkeypatch.setattr(
        "app.main.list_jobs",
        lambda _db: [SimpleNamespace(id=1, title="Backend Engineer", source_name="job.txt")],
    )

    monkeypatch.setattr(
        "app.main.run_analysis_graph",
        lambda *_args, **_kwargs: {
            "chunks": [SimpleNamespace(evidence_id="E1")],
            "scorecard": {"fit_score": 80, "confidence": 85, "matched_required": [], "missing_required": [], "evidence_chips": []},
            "structured_profile": {"resume_skills": ["python"]},
            "analysis": "Strong fit with API delivery evidence.",
        },
    )

    fake_completion = LLMCompletion(
        text="## Professional Summary\n- API-focused backend engineer [E1]",
        model_name="mock-model",
        prompt_tokens=10,
        completion_tokens=12,
        total_tokens=22,
        estimated_cost_usd=0.0,
        latency_ms=1.2,
    )
    monkeypatch.setattr("app.main.generate_tailored_resume", lambda *_args, **_kwargs: fake_completion)
    monkeypatch.setattr("app.main.log_llm_completion", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("app.main.GroqService", lambda: object())

    def _fake_get_db():
        yield _FakeSession()

    main_module.app.dependency_overrides = {main_module.get_db: _fake_get_db}

    with TestClient(main_module.app) as client:
        response = client.post("/jobs/1/tailor")

    assert response.status_code == 200
    assert "Tailored Resume" in response.text
    assert "API-focused backend engineer" in response.text

    main_module.app.dependency_overrides = {}
