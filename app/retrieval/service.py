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
# 연쇄 참조를 따라갈 시작 문서 수, 단계 수, 답변 근거에 넣을 청크 수
CHAIN_SEED_DOCUMENTS = 3
CHAIN_MAX_HOPS = 2
CHAIN_CONTEXT_LIMIT = 2


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


async def _follow_chain(
    vector_store: VectorStore,
    graph_store: GraphStore,
    chunks: list[Chunk],
    scope: AccessScope | None,
    filters: SearchFilters | None,
    embedding: list[float] | None,
) -> list[Chunk]:
    """검색된 문서가 가리키는 문서를 단계별로 따라간다. 볼 수 없는 문서는 지나가지 않는다."""
    seeds = [chunk["document_id"] for chunk in chunks if chunk.get("document_id")]
    frontier = list(dict.fromkeys(seeds))[:CHAIN_SEED_DOCUMENTS]
    visited = set(frontier)
    reached: list[str] = []
    for _ in range(CHAIN_MAX_HOPS):
        targets = [target for target in await graph_store.follow_references(frontier) if target not in visited]
        visited.update(targets)
        allowed = await vector_store.accessible_document_ids(targets, scope, filters) if targets else set()
        frontier = [target for target in targets if target in allowed]
        if not frontier:
            break
        reached.extend(frontier)
    present = {chunk["id"] for chunk in chunks}
    chunk_ids = [chunk_id for chunk_id in await graph_store.document_chunks(reached) if chunk_id not in present]
    if embedding is None:
        # 질문 임베딩이 실패해도 전문 검색으로 찾은 문서의 참조는 쓸 수 있다. 이때는 가까운 순서를 모른다
        return (await vector_store.fetch_accessible(chunk_ids, scope, filters))[:CHAIN_CONTEXT_LIMIT]
    # 질문과 표현이 다른 문서를 찾는 경로라 거리로 잘라내지 않고, 가까운 순서로 몇 개만 고른다
    return await vector_store.nearest_accessible(chunk_ids, scope, filters, embedding, CHAIN_CONTEXT_LIMIT)


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
        if isinstance(graph_store, Neo4jGraphStore) and not config.graph_search_enabled():
            return []
        # 먼저 PostgreSQL 하이브리드 검색 결과를 문서 단위 Neo4j 탐색 시작점으로 쓴다.
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
            # 샘플 그래프는 제목 매칭으로 시작점을 만들므로 빈 결과여도 아래 엔티티 검색은 실행한다.
            allowed_seed_ids = set()
        else:
            # PostgreSQL에서 권한·현재 버전·문서 필터를 재확인한 시작 문서만 Neo4j에 전달한다.
            allowed_seed_ids = await vector_store.accessible_document_ids(seed_ids, scope, filters)
        reference_ids = []
        if allowed_seed_ids:
            reference_ids = await graph_store.search_from_documents(
                [document_id for document_id in seed_ids if document_id in allowed_seed_ids], GRAPH_CANDIDATE_LIMIT
            )
        # 기존 Entity 경로도 유지하되, 반환 청크는 PostgreSQL에서 다시 권한 확인한다.
        entity_chunk_ids = await graph_store.search_from_entities(query, GRAPH_CANDIDATE_LIMIT)
        candidate_ids = list(dict.fromkeys([*reference_ids, *entity_chunk_ids]))
        if not candidate_ids:
            return []
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
    chunks = rrf_fuse(rankings, CONTEXT_LIMIT)
    if chunks and isinstance(graph_store, Neo4jGraphStore) and config.graph_search_enabled():
        try:
            embedding = None if query_embedding.exception() else query_embedding.result()
            chain = await _follow_chain(vector_store, graph_store, chunks, scope, filters, embedding)
        except Exception as exc:
            logger.error("chain search failed: %s", describe_exception(exc))
            failed.append("chain")
        else:
            # 기존 결과를 밀어내지 않도록 뒤에 더한다
            chunks = [*chunks, *chain]
    return {"chunks": chunks, "failed_sources": failed}
