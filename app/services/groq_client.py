from groq import Groq

from app.config import get_settings


settings = get_settings()


class GroqService:
    def __init__(self) -> None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")
        self.client = Groq(api_key=settings.groq_api_key)

    def complete(self, *, prompt: str, fast: bool = False) -> str:
        model = settings.groq_chat_model if fast else settings.groq_analysis_model
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
        return response.choices[0].message.content or "No response returned."
