from functools import lru_cache

from fastembed import TextEmbedding

from app.config import get_settings


settings = get_settings()


@lru_cache(maxsize=1)
def get_embedding_model() -> TextEmbedding:
    return TextEmbedding(model_name=settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    vectors = list(model.embed(texts))
    return [vector.tolist() for vector in vectors]


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]
