import asyncio

from langchain.embeddings import init_embeddings

from app.core import config
from app.core.errors import AiUnavailableError

TIMEOUT_SECONDS = 20.0


def _embedder():
    if not config.embedding_api_key() or not config.embedding_model():
        raise AiUnavailableError("EMBEDDING_API_KEY 또는 EMBEDDING_MODEL이 설정되지 않았습니다.")
    return init_embeddings(
        config.embedding_model(),
        api_key=config.embedding_api_key(),
        dimensions=config.embedding_dimensions(),
    )


async def embed_query(text: str) -> list[float]:
    return await asyncio.wait_for(_embedder().aembed_query(text), TIMEOUT_SECONDS)


async def embed_documents(texts: list[str]) -> list[list[float]]:
    return await asyncio.wait_for(_embedder().aembed_documents(texts), TIMEOUT_SECONDS)
