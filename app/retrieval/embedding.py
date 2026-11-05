import asyncio
import logging
from collections import OrderedDict
from functools import lru_cache

import httpx
from langchain.embeddings import init_embeddings

from app.core import config
from app.core.errors import AiUnavailableError, describe_exception

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 20.0
# 중계 API의 전체 제한 30초보다 길어야 중계의 오류 응답을 받을 수 있다
BATCH_TIMEOUT_SECONDS = 35.0
# 중계 API가 한 번에 받는 최대 개수에 맞춘다
BATCH_SIZE = 32
BATCH_ATTEMPTS = 2
QUERY_CACHE_SIZE = 256

_query_cache: OrderedDict[tuple[str | None, int, str], list[float]] = OrderedDict()


class ProxyEmbedder:
    """docs199-Server의 임시 임베딩 중계 API를 호출한다. 질문과 문서가 같은 Titan 설정을 쓴다."""

    def __init__(self, url: str, secret: str) -> None:
        self._url = url
        self._secret = secret

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=BATCH_TIMEOUT_SECONDS) as client:
            response = await client.post(self._url, headers={"X-Internal-Secret": self._secret}, json={"texts": texts})
        response.raise_for_status()
        vectors = response.json()["embeddings"]
        if len(vectors) != len(texts):
            raise AiUnavailableError("임베딩 개수가 요청과 다릅니다.")
        return vectors

    async def aembed_query(self, text: str) -> list[float]:
        return (await self.aembed_documents([text]))[0]


@lru_cache(maxsize=1)
def _build_embedder(model: str, api_key: str | None, dimensions: int):
    # Bedrock은 API 키 대신 실행 환경의 AWS 자격 증명을 쓰고, 차원을 model_kwargs로 받는다
    # 질문과 문서가 같은 설정이어야 검색이 맞으므로 서버 계약의 normalize를 명시한다
    if config.uses_aws_credentials(model):
        return init_embeddings(
            model, model_kwargs={"dimensions": dimensions, "normalize": True}, **config.bedrock_options()
        )
    return init_embeddings(model, api_key=api_key, dimensions=dimensions)


def _embedder():
    proxy_url = config.embedding_proxy_url()
    if proxy_url:
        if not config.internal_secret():
            raise AiUnavailableError("EMBEDDING_PROXY_URL에는 INTERNAL_SECRET이 필요합니다.")
        return ProxyEmbedder(proxy_url, config.internal_secret())
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
    # 같은 질문의 임베딩 비용을 줄인다. 벡터에는 권한 정보가 없어 대화끼리 공유해도 된다
    key = (config.embedding_proxy_url() or config.embedding_model(), config.embedding_dimensions(), text)
    cached = _query_cache.get(key)
    if cached is not None:
        _query_cache.move_to_end(key)
        return cached
    vector = await asyncio.wait_for(_embedder().aembed_query(text), TIMEOUT_SECONDS)
    _check_dimensions([vector])
    _query_cache[key] = vector
    if len(_query_cache) > QUERY_CACHE_SIZE:
        _query_cache.popitem(last=False)
    return vector


async def embed_documents(texts: list[str]) -> list[list[float]]:
    # 큰 문서가 하나의 시간 제한에 걸리지 않도록 배치마다 제한과 재시도를 둔다
    embedder = _embedder()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        for attempt in range(1, BATCH_ATTEMPTS + 1):
            try:
                batch_vectors = await asyncio.wait_for(embedder.aembed_documents(batch), BATCH_TIMEOUT_SECONDS)
                break
            except Exception as exc:
                if attempt == BATCH_ATTEMPTS:
                    raise
                logger.warning("Embedding batch retry: %s", describe_exception(exc))
        _check_dimensions(batch_vectors)
        vectors.extend(batch_vectors)
    return vectors
