import asyncio
import logging
import time
from functools import cache

from typing_extensions import NotRequired, TypedDict

from app.core import config
from app.core.errors import AiUnavailableError, describe_exception
from app.core.schemas import AccessScope, SearchFilters
from app.retrieval.embedding import embed_query
from app.retrieval.fusion import rrf_fuse_with_trace
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
    retrieval_trace: NotRequired[dict]


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
    trace: dict | None = None,
) -> list[Chunk]:
    """검색된 문서가 가리키는 문서를 단계별로 따라간다. 볼 수 없는 문서는 지나가지 않는다."""
    seeds = [chunk["document_id"] for chunk in chunks if chunk.get("document_id")]
    frontier = list(dict.fromkeys(seeds))[:CHAIN_SEED_DOCUMENTS]
    visited = set(frontier)
    reached: list[str] = []
    hop_trace = []
    for hop in range(1, CHAIN_MAX_HOPS + 1):
        current_frontier = frontier.copy()
        targets = [target for target in await graph_store.follow_references(current_frontier) if target not in visited]
        visited.update(targets)
        allowed = await vector_store.accessible_document_ids(targets, scope, filters) if targets else set()
        frontier = [target for target in targets if target in allowed]
        if trace is not None:
            # 권한 검증을 통과한 문서만 진단 응답에 기록한다.
            hop_trace.append({
                "hop": hop,
                "from_document_ids": current_frontier,
                "accessible_document_ids": frontier.copy(),
            })
        if not frontier:
            break
        reached.extend(frontier)
    present = {chunk["id"] for chunk in chunks}
    chunk_ids = [chunk_id for chunk_id in await graph_store.document_chunks(reached) if chunk_id not in present]
    if embedding is None:
        # 질문 임베딩이 실패해도 전문 검색으로 찾은 문서의 참조는 쓸 수 있다. 이때는 가까운 순서를 모른다
        selected = (await vector_store.fetch_accessible(chunk_ids, scope, filters))[:CHAIN_CONTEXT_LIMIT]
    else:
        # 질문과 표현이 다른 문서를 찾는 경로라 거리로 잘라내지 않고, 가까운 순서로 몇 개만 고른다
        selected = await vector_store.nearest_accessible(chunk_ids, scope, filters, embedding, CHAIN_CONTEXT_LIMIT)
    if trace is not None:
        trace["hops"] = hop_trace
        trace["hydrated_chunk_ids"] = [chunk["id"] for chunk in selected]
    return selected


async def filter_accessible_context(chunks: list[Chunk], scope: AccessScope | None = None) -> list[Chunk]:
    # 이전 근거를 재사용하기 전에 권한 변경, 문서 삭제, 개정을 다시 확인한다
    vector_store, _ = get_stores()
    accessible = await vector_store.filter_accessible([chunk["id"] for chunk in chunks], scope or config.access_scope())
    return [chunk for chunk in chunks if chunk["id"] in accessible]


async def search(
    query: str,
    filters: SearchFilters | None = None,
    request_scope: AccessScope | None = None,
    include_trace: bool = False,
    expected_document_ids: list[str] | None = None,
) -> SearchResult:
    # 한 DB가 실패해도 나머지 결과로 답변하고, 모두 실패하면 이전 컨텍스트를 덮어쓰지 않도록 중단한다
    vector_store, graph_store = get_stores()
    # 운영의 config.access_scope()는 항상 None이므로 요청에 권한 범위가 없으면 문서를 내보내지 않는다
    scope = request_scope or config.access_scope()
    if scope is None and not config.use_sample_data():
        # 권한 범위를 모르면 어떤 문서도 근거로 내보내지 않는다. 임베딩 비용도 쓰지 않는다
        logger.warning("Search skipped: no access scope")
        skipped: SearchResult = {"chunks": [], "failed_sources": []}
        if include_trace:
            skipped["retrieval_trace"] = {
                "status": "retrieval_skipped", "query": query, "search_modes": [],
                "skip_reason": "no_access_scope",
                "graph": {"status": "not_run", "seed_document_ids": [], "reference_chunk_ids": [],
                          "entity_chunk_ids": [], "neo4j_candidate_order": [], "hydrated_chunk_ids": [],
                          "chain": {"status": "not_run"}},
                "stages": {}, "rrf": [], "final": [], "timings_ms": {"total": 0.0},
            }
        return skipped
    started = time.perf_counter()
    trace: dict = {
        "status": "completed",
        "query": query,
        "search_modes": ["vector", "fulltext", "graph"],
        "filters": filters.model_dump(mode="json", by_alias=True, exclude_none=True) if filters else {},
        "graph": {"status": "pending", "seed_document_ids": [],
                  "reference_chunk_ids": [], "entity_chunk_ids": [], "neo4j_candidate_order": [],
                  "hydrated_chunk_ids": [],
                  "chain": {"status": "not_run", "hops": [], "hydrated_chunk_ids": []}},
        "stages": {},
        "timings_ms": {},
    }
    # 벡터 검색과 그래프 후보 관련도 검증이 같은 질문 벡터를 공유한다.
    query_embedding = asyncio.create_task(embed_query(query))

    async def search_vector() -> list[Chunk]:
        started_source = time.perf_counter()
        try:
            return await vector_store.search(await query_embedding, SOURCE_LIMIT, scope, filters)
        finally:
            trace["timings_ms"]["vector"] = round((time.perf_counter() - started_source) * 1000, 2)

    async def search_fulltext() -> list[Chunk]:
        started_source = time.perf_counter()
        try:
            return await vector_store.fulltext(query, SOURCE_LIMIT, scope, filters)
        finally:
            trace["timings_ms"]["fulltext"] = round((time.perf_counter() - started_source) * 1000, 2)

    async def search_graph() -> list[Chunk]:
        started_graph = time.perf_counter()
        if isinstance(graph_store, NoGraphStore):
            trace["graph"]["status"] = "disabled"
            return []
        if isinstance(graph_store, Neo4jGraphStore) and not config.graph_search_enabled():
            trace["graph"]["status"] = "disabled"
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
        trace["graph"]["seed_document_ids"] = [
            document_id for document_id in seed_ids if document_id in allowed_seed_ids
        ]
        reference_ids = []
        if allowed_seed_ids:
            reference_ids = await graph_store.search_from_documents(
                [document_id for document_id in seed_ids if document_id in allowed_seed_ids], GRAPH_CANDIDATE_LIMIT
            )
        # 기존 Entity 경로도 유지하되, 반환 청크는 PostgreSQL에서 다시 권한 확인한다.
        entity_chunk_ids = await graph_store.search_from_entities(query, GRAPH_CANDIDATE_LIMIT)
        candidate_ids = list(dict.fromkeys([*reference_ids, *entity_chunk_ids]))
        if not candidate_ids:
            trace["graph"]["status"] = "no_candidates"
            trace["timings_ms"]["graph"] = round((time.perf_counter() - started_graph) * 1000, 2)
            return []
        try:
            embedding = await query_embedding
        except Exception:
            # 질문 임베딩 장애에도 키워드 검색으로 찾은 명시 관계 경로는 계속 제공한다.
            embedding = None
        hydrated = await vector_store.fetch_accessible(
            candidate_ids[:GRAPH_CANDIDATE_LIMIT], scope, filters, embedding
        )
        safe_ids = {chunk["id"] for chunk in hydrated}
        trace["graph"]["reference_chunk_ids"] = [chunk_id for chunk_id in reference_ids if chunk_id in safe_ids]
        trace["graph"]["entity_chunk_ids"] = [chunk_id for chunk_id in entity_chunk_ids if chunk_id in safe_ids]
        trace["graph"]["neo4j_candidate_order"] = [chunk_id for chunk_id in candidate_ids if chunk_id in safe_ids]
        trace["graph"]["hydrated_chunk_ids"] = [chunk["id"] for chunk in hydrated]
        trace["graph"]["status"] = "ok"
        trace["timings_ms"]["graph"] = round((time.perf_counter() - started_graph) * 1000, 2)
        return hydrated

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
            if name == "graph":
                trace["graph"]["status"] = "error"
        else:
            rankings.append(result)
            if name == "graph" and trace["graph"]["status"] == "pending":
                trace["graph"]["status"] = "ok"
    if not rankings:
        raise AiUnavailableError("벡터, 그래프, 전문 검색이 모두 실패했습니다.")
    if include_trace:
        expected_ids = set(expected_document_ids) if expected_document_ids is not None else None
        trace["stages"] = {
            name: [
                {"rank": rank, "chunk_id": chunk["id"], "document_id": chunk.get("document_id"),
                 "version_id": chunk.get("version_id"),
                 **({"matches_expected": chunk.get("document_id") in expected_ids} if expected_ids is not None else {})}
                for rank, chunk in enumerate(result, start=1)
            ]
            for name, result in zip(names, results)
            if not isinstance(result, Exception)
        }
    chunks, rrf_trace = rrf_fuse_with_trace(rankings, CONTEXT_LIMIT)
    if include_trace:
        chunk_by_id = {chunk["id"]: chunk for ranking in rankings for chunk in ranking}
        successful_names = [name for name, item in zip(names, results) if not isinstance(item, Exception)]
        trace["rrf_sources"] = successful_names
        trace["rrf"] = [
            {**item,
             "source_ranks": {successful_names[int(source)]: rank for source, rank in item["source_ranks"].items()},
             "document_id": chunk_by_id[item["chunk_id"]].get("document_id"),
             "version_id": chunk_by_id[item["chunk_id"]].get("version_id"),
             **({"matches_expected": chunk_by_id[item["chunk_id"]].get("document_id") in expected_ids}
                if expected_ids is not None else {})}
            for item in rrf_trace
        ]
        trace["graph"]["status"] = trace["graph"].get("status", "ok")
        trace["timings_ms"]["retrieval_before_chain"] = round((time.perf_counter() - started) * 1000, 2)
    rrf_chunk_ids = {chunk["id"] for chunk in chunks}
    chain: list[Chunk] = []
    if chunks and isinstance(graph_store, Neo4jGraphStore) and config.graph_search_enabled():
        if include_trace:
            trace["graph"]["chain"]["status"] = "pending"
            chain_started = time.perf_counter()
        try:
            embedding = None if query_embedding.exception() else query_embedding.result()
            chain = await _follow_chain(
                vector_store, graph_store, chunks, scope, filters, embedding,
                trace["graph"]["chain"] if include_trace else None,
            )
        except Exception as exc:
            logger.error("chain search failed: %s", describe_exception(exc))
            failed.append("chain")
            if include_trace:
                trace["graph"]["chain"]["status"] = "error"
        else:
            # 기존 결과를 밀어내지 않도록 뒤에 더한다
            chunks = [*chunks, *chain]
            if include_trace:
                trace["graph"]["chain"]["status"] = "ok"
        if include_trace:
            trace["timings_ms"]["chain"] = round((time.perf_counter() - chain_started) * 1000, 2)
    if include_trace:
        trace["final"] = [
            {"rank": rank, "chunk_id": chunk["id"], "document_id": chunk.get("document_id"),
             "version_id": chunk.get("version_id"), "origin": "rrf" if chunk["id"] in rrf_chunk_ids else "chain",
             **({"matches_expected": chunk.get("document_id") in expected_ids}
                if expected_document_ids is not None else {})}
            for rank, chunk in enumerate(chunks, start=1)
        ]
        if expected_document_ids is not None:
            trace["evaluation"] = {
                "expected_document_ids": list(dict.fromkeys(expected_document_ids)),
                "unexpected_candidates": [
                    {"stage": name, "rank": candidate["rank"], "chunk_id": candidate["chunk_id"],
                     "document_id": candidate["document_id"]}
                    for name, candidates in trace["stages"].items()
                    for candidate in candidates
                    if candidate.get("document_id") not in expected_ids
                ],
                "unexpected_final": [
                    candidate for candidate in trace["final"] if not candidate["matches_expected"]
                ],
            }
        trace["timings_ms"]["total"] = round((time.perf_counter() - started) * 1000, 2)
    result: SearchResult = {"chunks": chunks, "failed_sources": failed}
    if include_trace:
        result["retrieval_trace"] = trace
    return result
