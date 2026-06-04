import json

from sqlalchemy.orm import Session

from app.models import LlmRequestLog
from app.services.groq_client import LLMCompletion


def log_llm_completion(
    session: Session,
    *,
    request_id: str,
    route_name: str,
    completion: LLMCompletion,
    metadata: dict | None = None,
) -> None:
    session.add(
        LlmRequestLog(
            request_id=request_id,
            route_name=route_name,
            model_name=completion.model_name,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            total_tokens=completion.total_tokens,
            estimated_cost_usd=completion.estimated_cost_usd,
            latency_ms=completion.latency_ms,
            metadata_json=json.dumps(metadata or {}),
        )
    )
    session.commit()
