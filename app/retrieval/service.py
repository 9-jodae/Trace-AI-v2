import asyncio
import logging
from functools import cache

from typing_extensions import TypedDict

from app.core import config
from app.core.errors import AiUnavailableError, describe_exception
from app.core.schemas import AccessScope, SearchFilters
from app.retrieval.embedding import embed_query
from app.retrieval.fusion import rrf_fuse
from app.retrieval.neo4j_store import Neo4jGraphStore
from app.retrieval.schemas import Chunk
from app.retrieval.stores import (
    CHUNK_EMBEDDING_DIMENSIONS,
    GraphStore,
    NoGraphStore,
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
def _real_stores(database_url: str, neo4j_settings: tuple | None) -> tuple[VectorStore, GraphStore]:
    if not neo4j_settings:
        # 설정 누락과 구분할 수 있도록 저장소를 만들 때 한 번 남긴다
        logger.warning("Graph search disabled: NEO4J_URI is not set")
    graph_store = Neo4jGraphStore.from_uri(*neo4j_settings) if neo4j_settings else NoGraphStore()
    return PostgresVectorStore(database_url), graph_store


def get_real_stores() -> tuple[VectorStore, GraphStore]:
    # 그래프는 선택이다. NEO4J_URI를 비우면 벡터 검색만 쓴다
    if not config.database_url() or (config.neo4j_uri() and not config.neo4j_password()):
        raise AiUnavailableError("운영 저장소 설정이 없습니다.")
    if config.embedding_dimensions() != CHUNK_EMBEDDING_DIMENSIONS:
        raise AiUnavailableError(f"EMBEDDING_DIMENSIONS는 {CHUNK_EMBEDDING_DIMENSIONS}이어야 합니다.")
    neo4j_settings = (config.neo4j_uri(), config.neo4j_user(), config.neo4j_password(), config.neo4j_database())
    return _real_stores(config.database_url(), neo4j_settings if config.neo4j_uri() else None)


def get_stores() -> tuple[VectorStore, GraphStore]:
    # 샘플은 로컬에서 명시적으로 켰을 때만 쓴다. 설정이 없으면 대체하지 않고 실패한다
    return _sample_stores() if config.use_sample_data() else get_real_stores()


async def close_stores() -> None:
    if _real_stores.cache_info().currsize:
        vector_store, graph_store = get_real_stores()
        await vector_store.close()
        await graph_store.close()
    _real_stores.cache_clear()


async def filter_accessible_context(chunks: list[Chunk], scope: AccessScope | None = None) -> list[Chunk]:
    # 이전 근거를 재사용하기 전에 권한 변경, 문서 삭제, 개정을 다시 확인한다
    vector_store, _ = get_stores()
    accessible = await vector_store.filter_accessible([chunk["id"] for chunk in chunks], scope or config.access_scope())
    return [chunk for chunk in chunks if chunk["id"] in accessible]


async def search(
    query: str, filters: SearchFilters | None = None, request_scope: AccessScope | None = None
) -> SearchResult:
    # 한 DB가 실패해도 나머지 결과로 답변하고, 모두 실패하면 이전 컨텍스트를 덮어쓰지 않도록 중단한다
    vector_store, graph_store = get_stores()
    # 운영의 config.access_scope()는 항상 None이므로 요청에 권한 범위가 없으면 문서를 내보내지 않는다
    scope = request_scope or config.access_scope()
    if scope is None and not config.use_sample_data():
        # 권한 범위를 모르면 어떤 문서도 근거로 내보내지 않는다. 임베딩 비용도 쓰지 않는다
        logger.warning("Search skipped: no access scope")
        return {"chunks": [], "failed_sources": []}
    # 벡터 검색과 그래프 후보 관련도 검증이 같은 질문 벡터를 공유한다.
    query_embedding = asyncio.create_task(embed_query(query))

    async def search_vector() -> list[Chunk]:
        return await vector_store.search(await query_embedding, SOURCE_LIMIT, scope, filters)

    async def search_fulltext() -> list[Chunk]:
        return await vector_store.fulltext(query, SOURCE_LIMIT, scope, filters)

    async def search_graph() -> list[Chunk]:
        if isinstance(graph_store, NoGraphStore):
            return []
        # 키워드·벡터 검색의 문서 후보를 합쳐 Graph seed로 삼는다.
        vector_result, fulltext_result = await asyncio.gather(
            vector_task, fulltext_task, return_exceptions=True
        )
        if isinstance(vector_result, Exception) and isinstance(fulltext_result, Exception):
            raise RuntimeError("vector and full-text seed searches failed")

        if config.use_sample_data() and isinstance(graph_store, SampleGraphStore):
            # 샘플 저장소는 문서 ID가 없는 고정 데이터이므로 기존 제목 seed를 사용한다.
            seed_ids = await graph_store.seed_documents(query)
        else:
            seed_ids = list(
                dict.fromkeys(
                    chunk.get("document_id") or chunk["id"]
                    for result in (vector_result, fulltext_result)
                    if not isinstance(result, Exception)
                    for chunk in result
                )
            )
        if not seed_ids:
            return []
        # seed 문서의 권한·현재 버전·필터를 확인한 뒤에만 관계를 확장한다.
        allowed_seed_ids = await vector_store.accessible_document_ids(seed_ids, scope, filters)
        if not allowed_seed_ids:
            return []
        candidate_ids = await graph_store.search_from_documents(
            [document_id for document_id in seed_ids if document_id in allowed_seed_ids], GRAPH_CANDIDATE_LIMIT
        )
        try:
            embedding = await query_embedding
        except Exception:
            # 질문 임베딩 장애에도 키워드 검색으로 찾은 명시 관계 경로는 계속 제공한다.
            embedding = None
        return await vector_store.fetch_accessible(
            candidate_ids[:GRAPH_CANDIDATE_LIMIT], scope, filters, embedding
        )

    vector_task = asyncio.create_task(search_vector())
    fulltext_task = asyncio.create_task(search_fulltext())
    names = ["vector", "graph", "fulltext"]
    results = await asyncio.gather(vector_task, search_graph(), fulltext_task, return_exceptions=True)
    rankings = []
    failed = []
    for name, result in zip(names, results):
        if isinstance(result, Exception):
            logger.error("%s search failed: %s", name, describe_exception(result))
            failed.append(name)
        else:
            rankings.append(result)
    if not rankings:
        raise AiUnavailableError("벡터, 그래프, 전문 검색이 모두 실패했습니다.")
    return {"chunks": rrf_fuse(rankings, CONTEXT_LIMIT), "failed_sources": failed}
