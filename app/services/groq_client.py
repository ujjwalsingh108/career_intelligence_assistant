from dataclasses import dataclass
from time import perf_counter

from groq import Groq

from app.config import get_settings


settings = get_settings()


@dataclass
class LLMCompletion:
    text: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    latency_ms: float


class GroqService:
    def __init__(self) -> None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")
        self.client = Groq(api_key=settings.groq_api_key)

    def _estimate_cost(self, model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
        if model_name == settings.groq_analysis_model:
            input_rate = settings.groq_analysis_input_cost_per_million
            output_rate = settings.groq_analysis_output_cost_per_million
        else:
            input_rate = settings.groq_chat_input_cost_per_million
            output_rate = settings.groq_chat_output_cost_per_million
        return round(((prompt_tokens / 1_000_000) * input_rate) + ((completion_tokens / 1_000_000) * output_rate), 8)

    def complete(self, *, prompt: str, fast: bool = False) -> LLMCompletion:
        model = settings.groq_chat_model if fast else settings.groq_analysis_model
        started_at = perf_counter()
        response = self.client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful career intelligence assistant. Ground every conclusion in the provided "
                        "resume and job description evidence. If evidence is missing, say so explicitly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        latency_ms = (perf_counter() - started_at) * 1000
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0
        return LLMCompletion(
            text=response.choices[0].message.content or "No response returned.",
            model_name=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=self._estimate_cost(model, prompt_tokens, completion_tokens),
            latency_ms=round(latency_ms, 2),
        )
