import asyncio
import logging
from functools import cache
from typing import NamedTuple
from uuid import UUID

from typing_extensions import TypedDict

from app.core import config
from app.core.errors import AiUnavailableError, describe_exception
from app.core.schemas import AccessScope, SearchFilters
from app.retrieval.embedding import embed_query
from app.retrieval.fusion import distinct_documents_first, rrf_fuse
from app.retrieval.neo4j_store import Neo4jGraphStore
from app.retrieval.schemas import Chunk, RegisteredConflict
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
# 그래프 후보가 이보다 적으면 전문 검색으로 보완한다
FULLTEXT_FALLBACK_BELOW = 5
# 권한 필터로 줄어들어도 SOURCE_LIMIT을 채울 수 있도록 그래프 후보를 더 가져온다
GRAPH_CANDIDATE_LIMIT = 50
CONTEXT_LIMIT = 5
# 문서마다 하나씩 먼저 채우기 전에 넉넉히 합쳐 둘 후보 수
FUSION_CANDIDATE_LIMIT = CONTEXT_LIMIT * 4
# 연쇄 참조를 따라갈 시작 문서 수, 단계 수, 답변 근거에 넣을 청크 수
CHAIN_SEED_DOCUMENTS = 3
CHAIN_MAX_HOPS = 2
CHAIN_CONTEXT_LIMIT = 2
# 충돌 탐색 질문에 근거로 넣을 등록 충돌 수와 문서마다 넣을 청크 수
KNOWN_CONFLICT_LIMIT = 5
KNOWN_CONFLICT_CHUNKS = 2


class SearchResult(TypedDict):
    chunks: list[Chunk]
    failed_sources: list[str]


class KnownConflict(NamedTuple):
    document_id: UUID
    other_document_id: UUID
    item: str


class KnownConflictResult(TypedDict):
    chunks: list[Chunk]
    conflicts: list[RegisteredConflict]


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


async def _nearest_in_document(
    document_id: UUID,
    scope: AccessScope | None,
    filters: SearchFilters | None,
    embedding: list[float],
) -> list[Chunk]:
    vector_store, graph_store = get_stores()
    chunk_ids = await graph_store.document_chunks([str(document_id)])
    return await vector_store.nearest_accessible(chunk_ids, scope, filters, embedding, KNOWN_CONFLICT_CHUNKS)


async def _known_conflict_pair(
    conflict: KnownConflict, scope: AccessScope | None, filters: SearchFilters | None
) -> tuple[list[Chunk], list[Chunk]]:
    embedding = await embed_query(conflict.item)
    first, second = await asyncio.gather(
        _nearest_in_document(conflict.document_id, scope, filters, embedding),
        _nearest_in_document(conflict.other_document_id, scope, filters, embedding),
    )
    return first, second


async def fetch_known_conflicts(
    conflicts: list[KnownConflict], filters: SearchFilters | None = None, request_scope: AccessScope | None = None
) -> KnownConflictResult:
    """등록 충돌의 두 문서에서 항목과 가까운 청크를 찾는다. 권한과 필터는 일반 검색과 같이 다시 확인한다."""
    scope = request_scope or config.access_scope()
    empty: KnownConflictResult = {"chunks": [], "conflicts": []}
    if scope is None and not config.use_sample_data():
        return empty
    selected = conflicts[:KNOWN_CONFLICT_LIMIT]
    try:
        pairs = await asyncio.gather(*(_known_conflict_pair(conflict, scope, filters) for conflict in selected))
    except Exception as exc:
        # 등록 충돌을 못 읽어도 일반 검색 결과로 답한다. 답변은 검색 범위를 밝힌다
        logger.warning("Known conflict lookup failed: %s", describe_exception(exc))
        return empty
    chunks: dict[str, Chunk] = {}
    registered: list[RegisteredConflict] = []
    for conflict, (first, second) in zip(selected, pairs, strict=True):
        # 한쪽이라도 필터 밖이거나 청크가 없으면 비교할 수 없으므로 뺀다
        if not first or not second:
            continue
        chunks.update({chunk["id"]: chunk for chunk in [*first, *second]})
        registered.append(
            {
                "item": conflict.item,
                "source": first[0].get("source") or "출처 미상",
                "other_source": second[0].get("source") or "출처 미상",
            }
        )
    return {"chunks": list(chunks.values()), "conflicts": registered}


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
    # 하나의 질문 벡터를 벡터 검색과 그래프 후보 관련도 검증에 공유한다.
    query_embedding = asyncio.create_task(embed_query(query))

    async def search_vector() -> list[Chunk]:
        return await vector_store.search(await query_embedding, SOURCE_LIMIT, scope, filters)

    async def search_graph() -> list[Chunk]:
        if isinstance(graph_store, Neo4jGraphStore) and not config.graph_search_enabled():
            return []
        # Neo4j 제목 매칭 결과를 먼저 PostgreSQL에서 허용된 현재 문서인지 확인한 뒤에만 관계를 확장한다.
        seed_ids = await graph_store.seed_documents(query)
        # 엔티티 경로는 청크 ID를 돌려주므로 문서 시작점에 섞지 않고 후보에 바로 넣는다
        entity_chunk_ids = await graph_store.search_from_entities(query, GRAPH_CANDIDATE_LIMIT)
        if not seed_ids and not entity_chunk_ids:
            return []
        allowed_seed_ids = await vector_store.accessible_document_ids(seed_ids, scope, filters) if seed_ids else []
        reference_ids = []
        if allowed_seed_ids:
            reference_ids = await graph_store.search_from_documents(
                [document_id for document_id in seed_ids if document_id in allowed_seed_ids], GRAPH_CANDIDATE_LIMIT
            )
        candidate_ids = list(dict.fromkeys([*entity_chunk_ids, *reference_ids]))
        if not candidate_ids:
            return []
        try:
            embedding = await query_embedding
        except Exception:
            # 질문 임베딩 장애에도 명시적 문서 관계 검색은 계속 제공한다.
            embedding = None
        return await vector_store.fetch_accessible(
            candidate_ids[:GRAPH_CANDIDATE_LIMIT], scope, filters, embedding
        )

    names = ["vector", "graph"]
    results = await asyncio.gather(search_vector(), search_graph(), return_exceptions=True)
    rankings = []
    failed = []
    graph_count = 0
    for name, result in zip(names, results):
        if isinstance(result, Exception):
            logger.error("%s search failed: %s", name, describe_exception(result))
            failed.append(name)
        else:
            rankings.append(result)
            if name == "graph":
                graph_count = len(result)
    if graph_count < FULLTEXT_FALLBACK_BELOW:
        try:
            rankings.append(await vector_store.fulltext(query, SOURCE_LIMIT, scope, filters))
        except Exception as exc:
            logger.error("fulltext search failed: %s", describe_exception(exc))
            failed.append("fulltext")
    if not rankings:
        raise AiUnavailableError("벡터, 그래프, 전문 검색이 모두 실패했습니다.")
    # 문서가 147개로 늘자 같은 문서의 청크가 근거를 여러 칸 차지해 기대 문서를 밀어냈다(2026-11-06 측정)
    chunks = distinct_documents_first(rrf_fuse(rankings, FUSION_CANDIDATE_LIMIT), CONTEXT_LIMIT)
    # 본문은 PostgreSQL에만 있어 벡터와 전문 검색이 모두 실패하면 찾지 못한 것이 아니라 장애다.
    # 그래프 후보가 비어 성공으로 끝나도 "문서 없음"으로 답하지 않는다
    if not chunks and {"vector", "fulltext"} <= set(failed):
        raise AiUnavailableError("문서 검색을 사용할 수 없습니다.")
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
