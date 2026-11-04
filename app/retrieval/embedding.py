import asyncio
import logging
from functools import lru_cache

from langchain.embeddings import init_embeddings

from app.core import config
from app.core.errors import AiUnavailableError, describe_exception

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 20.0
BATCH_SIZE = 64
BATCH_ATTEMPTS = 2


@lru_cache(maxsize=1)
def _build_embedder(model: str, api_key: str | None, dimensions: int):
    # Bedrock은 API 키 대신 실행 환경의 AWS 자격 증명을 쓰고, 차원을 model_kwargs로 받는다
    if config.uses_aws_credentials(model):
        return init_embeddings(model, model_kwargs={"dimensions": dimensions})
    return init_embeddings(model, api_key=api_key, dimensions=dimensions)


def _embedder():
    model = config.embedding_model()
    if not model:
        raise AiUnavailableError("EMBEDDING_MODEL이 설정되지 않았습니다.")
    if not config.uses_aws_credentials(model) and not config.embedding_api_key():
        raise AiUnavailableError("EMBEDDING_API_KEY가 설정되지 않았습니다.")
    return _build_embedder(model, config.embedding_api_key(), config.embedding_dimensions())


def _check_dimensions(vectors: list[list[float]]) -> None:
    # 차원이 다른 벡터는 검색이나 저장 단계에서 늦게 실패하므로 받자마자 막는다
    expected = config.embedding_dimensions()
    if any(len(vector) != expected for vector in vectors):
        raise AiUnavailableError(f"임베딩 차원이 설정값 {expected}과 다릅니다.")


async def embed_query(text: str) -> list[float]:
    vector = await asyncio.wait_for(_embedder().aembed_query(text), TIMEOUT_SECONDS)
    _check_dimensions([vector])
    return vector


async def embed_documents(texts: list[str]) -> list[list[float]]:
    # 큰 문서가 하나의 시간 제한에 걸리지 않도록 배치마다 제한과 재시도를 둔다
    embedder = _embedder()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        for attempt in range(1, BATCH_ATTEMPTS + 1):
            try:
                batch_vectors = await asyncio.wait_for(embedder.aembed_documents(batch), TIMEOUT_SECONDS)
                break
            except Exception as exc:
                if attempt == BATCH_ATTEMPTS:
                    raise
                logger.warning("Embedding batch retry: %s", describe_exception(exc))
        _check_dimensions(batch_vectors)
        vectors.extend(batch_vectors)
    return vectors
