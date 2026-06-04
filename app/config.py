from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Career Intelligence Assistant"
    database_url: str = "postgresql+psycopg://career:career@localhost:5432/career_intelligence"
    groq_api_key: str = ""
    groq_analysis_model: str = "llama-3.3-70b-versatile"
    groq_chat_model: str = "llama-3.1-8b-instant"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384
    chunk_size: int = 900
    chunk_overlap: int = 180
    max_context_chunks: int = 6

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
