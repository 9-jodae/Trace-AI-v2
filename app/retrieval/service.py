import asyncio
import logging
from functools import cache

from typing_extensions import TypedDict

from app.core import config
from app.core.errors import AiUnavailableError, describe_exception
from app.retrieval.embedding import embed_query
from app.retrieval.fusion import rrf_fuse
from app.retrieval.neo4j_store import Neo4jGraphStore
from app.retrieval.schemas import Chunk
from app.retrieval.stores import (
    GraphStore,
    PostgresVectorStore,
    SampleGraphStore,
    SampleVectorStore,
    VectorStore,
)

logger = logging.getLogger(__name__)

SOURCE_LIMIT = 10
CONTEXT_LIMIT = 5


class SearchResult(TypedDict):
    chunks: list[Chunk]
    failed_sources: list[str]


@cache
def _sample_stores() -> tuple[VectorStore, GraphStore]:
    return SampleVectorStore(), SampleGraphStore()


@cache
def _real_stores(database_url: str, neo4j_settings: tuple) -> tuple[VectorStore, GraphStore]:
    return PostgresVectorStore(database_url), Neo4jGraphStore.from_uri(*neo4j_settings)


def get_real_stores() -> tuple[VectorStore, GraphStore]:
    if not config.database_url() or not config.neo4j_uri() or not config.neo4j_password():
        raise AiUnavailableError("운영 저장소 설정이 없습니다.")
    return _real_stores(
        config.database_url(),
        (config.neo4j_uri(), config.neo4j_user(), config.neo4j_password(), config.neo4j_database()),
    )


def get_stores() -> tuple[VectorStore, GraphStore]:
    # 샘플은 로컬에서 명시적으로 켰을 때만 쓴다. 설정이 없으면 대체하지 않고 실패한다
    return _sample_stores() if config.use_sample_data() else get_real_stores()


async def close_stores() -> None:
    if _real_stores.cache_info().currsize:
        vector_store, graph_store = get_real_stores()
        await vector_store.close()
        await graph_store.close()
    _real_stores.cache_clear()


async def search(query: str) -> SearchResult:
    # 한 DB가 실패해도 나머지 결과로 답변하고, 모두 실패하면 이전 컨텍스트를 덮어쓰지 않도록 중단한다
    vector_store, graph_store = get_stores()
    allowed_groups = config.allowed_group_ids()
    if not allowed_groups and not config.use_sample_data():
        # 권한 범위를 모르면 어떤 문서도 근거로 내보내지 않는다. 임베딩 비용도 쓰지 않는다
        logger.warning("Search skipped: no allowed groups")
        return {"chunks": [], "failed_sources": []}

    async def search_vector() -> list[Chunk]:
        return await vector_store.search(await embed_query(query), SOURCE_LIMIT, allowed_groups)

    names = ["vector", "graph"]
    results = await asyncio.gather(
        search_vector(),
        graph_store.search(query, SOURCE_LIMIT, allowed_groups),
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
