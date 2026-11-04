import asyncio

from langchain.embeddings import init_embeddings

from app.core import config
from app.core.errors import AiUnavailableError

TIMEOUT_SECONDS = 20.0


def _embedder():
    model = config.embedding_model()
    if not model:
        raise AiUnavailableError("EMBEDDING_MODEL이 설정되지 않았습니다.")
    dimensions = config.embedding_dimensions()
    # Bedrock은 API 키 대신 실행 환경의 AWS 자격 증명을 쓰고, 차원을 model_kwargs로 받는다
    if config.uses_aws_credentials(model):
        return init_embeddings(model, model_kwargs={"dimensions": dimensions})
    if not config.embedding_api_key():
        raise AiUnavailableError("EMBEDDING_API_KEY가 설정되지 않았습니다.")
    return init_embeddings(model, api_key=config.embedding_api_key(), dimensions=dimensions)


async def embed_query(text: str) -> list[float]:
    return await asyncio.wait_for(_embedder().aembed_query(text), TIMEOUT_SECONDS)


async def embed_documents(texts: list[str]) -> list[list[float]]:
    return await asyncio.wait_for(_embedder().aembed_documents(texts), TIMEOUT_SECONDS)
