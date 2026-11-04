import asyncio
import logging
from functools import cache

from typing_extensions import TypedDict

from app.core import config
from app.core.errors import AiUnavailableError, describe_exception
from app.retrieval.embedding import embed_query
from app.retrieval.fusion import rrf_fuse
from app.retrieval.schemas import Chunk
from app.retrieval.stores import GraphStore, SampleGraphStore, SampleVectorStore, VectorStore

logger = logging.getLogger(__name__)

SOURCE_LIMIT = 10
CONTEXT_LIMIT = 5


class SearchResult(TypedDict):
    chunks: list[Chunk]
    failed_sources: list[str]


@cache
def _sample_stores() -> tuple[VectorStore, GraphStore]:
    return SampleVectorStore(), SampleGraphStore()


def get_stores() -> tuple[VectorStore, GraphStore]:
    if config.use_sample_data():
        return _sample_stores()
    # TODO: 벡터 저장소 확정(#9)과 Neo4j 연동(#3) 뒤 실제 저장소를 반환한다. 샘플로 대체하지 않는다
    raise AiUnavailableError("운영 저장소가 아직 연결되지 않았습니다.")


async def search(query: str) -> SearchResult:
    # 한 DB가 실패해도 나머지 결과로 답변하고, 모두 실패하면 이전 컨텍스트를 덮어쓰지 않도록 중단한다
    vector_store, graph_store = get_stores()

    async def search_vector() -> list[Chunk]:
        return await vector_store.search(await embed_query(query), SOURCE_LIMIT)

    names = ["vector", "graph"]
    results = await asyncio.gather(
        search_vector(),
        graph_store.search(query, SOURCE_LIMIT),
        return_exceptions=True,
    )
    rankings = []
    failed = []
    for name, result in zip(names, results):
        if isinstance(result, Exception):
            logger.error("%s search failed: %s", name, describe_exception(result))
            failed.append(name)
        else:
            rankings.append(result)
    if not rankings:
        raise AiUnavailableError("벡터 DB와 그래프 DB 검색이 모두 실패했습니다.")
    return {"chunks": rrf_fuse(rankings, CONTEXT_LIMIT), "failed_sources": failed}
