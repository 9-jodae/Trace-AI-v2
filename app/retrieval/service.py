import asyncio
import logging

from typing_extensions import TypedDict

from app.core import config
from app.core.errors import AiUnavailableError
from app.retrieval.embedding import embed_query
from app.retrieval.fusion import rrf_fuse
from app.retrieval.schemas import Chunk
from app.retrieval.stores import GraphStore, PostgresVectorStore, SampleGraphStore, SampleVectorStore, VectorStore

logger = logging.getLogger(__name__)

SOURCE_LIMIT = 10
CONTEXT_LIMIT = 5

vector_store: VectorStore = (
    PostgresVectorStore(config.database_url()) if config.database_url() else SampleVectorStore()
)
graph_store: GraphStore = SampleGraphStore()


class SearchResult(TypedDict):
    chunks: list[Chunk]
    failed_sources: list[str]


async def _search_vector(query: str) -> list[Chunk]:
    return await vector_store.search(await embed_query(query), SOURCE_LIMIT)


async def search(query: str) -> SearchResult:
    # 한 DB가 실패해도 나머지 결과로 답변하고, 모두 실패하면 이전 컨텍스트를 덮어쓰지 않도록 중단한다
    names = ["vector", "graph"]
    results = await asyncio.gather(
        _search_vector(query),
        graph_store.search(query, SOURCE_LIMIT),
        return_exceptions=True,
    )
    rankings = []
    failed = []
    for name, result in zip(names, results):
        if isinstance(result, Exception):
            logger.error("%s search failed: %r", name, result)
            failed.append(name)
        else:
            rankings.append(result)
    if not rankings:
        raise AiUnavailableError("벡터 DB와 그래프 DB 검색이 모두 실패했습니다.")
    return {"chunks": rrf_fuse(rankings, CONTEXT_LIMIT), "failed_sources": failed}
