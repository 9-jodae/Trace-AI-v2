import asyncio
import logging
from functools import cache

from typing_extensions import TypedDict

from app.core import config
from app.core.errors import AiUnavailableError, describe_exception
from app.core.schemas import SearchFilters
from app.retrieval.embedding import embed_query
from app.retrieval.fusion import rrf_fuse
from app.retrieval.neo4j_store import Neo4jGraphStore
from app.retrieval.schemas import Chunk
from app.retrieval.stores import (
    CHUNK_EMBEDDING_DIMENSIONS,
    GraphStore,
    PostgresVectorStore,
    SampleGraphStore,
    SampleVectorStore,
    VectorStore,
)

logger = logging.getLogger(__name__)

SOURCE_LIMIT = 10
# 권한 필터로 줄어들어도 SOURCE_LIMIT을 채울 수 있도록 그래프 후보를 더 가져온다
GRAPH_CANDIDATE_LIMIT = 50
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
    if config.embedding_dimensions() != CHUNK_EMBEDDING_DIMENSIONS:
        raise AiUnavailableError(f"EMBEDDING_DIMENSIONS는 {CHUNK_EMBEDDING_DIMENSIONS}이어야 합니다.")
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


async def filter_accessible_context(chunks: list[Chunk]) -> list[Chunk]:
    # 이전 근거를 재사용하기 전에 권한 변경, 문서 삭제, 개정을 다시 확인한다
    vector_store, _ = get_stores()
    accessible = await vector_store.filter_accessible([chunk["id"] for chunk in chunks], config.access_scope())
    return [chunk for chunk in chunks if chunk["id"] in accessible]


async def search(query: str, filters: SearchFilters | None = None) -> SearchResult:
    # 한 DB가 실패해도 나머지 결과로 답변하고, 모두 실패하면 이전 컨텍스트를 덮어쓰지 않도록 중단한다
    vector_store, graph_store = get_stores()
    scope = config.access_scope()
    if scope is None and not config.use_sample_data():
        # 권한 범위를 모르면 어떤 문서도 근거로 내보내지 않는다. 임베딩 비용도 쓰지 않는다
        logger.warning("Search skipped: no access scope")
        return {"chunks": [], "failed_sources": []}

    async def search_vector() -> list[Chunk]:
        return await vector_store.search(await embed_query(query), SOURCE_LIMIT, scope, filters)

    async def search_graph() -> list[Chunk]:
        # 권한을 확인하지 못하면 그래프 결과도 내보내지 않도록 실패로 처리한다
        candidates = await graph_store.search(query, GRAPH_CANDIDATE_LIMIT)
        accessible = await vector_store.filter_accessible([chunk["id"] for chunk in candidates], scope, filters)
        return [chunk for chunk in candidates if chunk["id"] in accessible][:SOURCE_LIMIT]

    names = ["vector", "graph"]
    results = await asyncio.gather(
        search_vector(),
        search_graph(),
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
