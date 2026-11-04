import asyncio
import logging
from functools import cache

from typing_extensions import TypedDict

from app.core import config
from app.core.errors import AiUnavailableError, describe_exception
from app.retrieval.embedding import embed_query
from app.retrieval.fusion import rrf_fuse
from app.retrieval.neo4j_store import Neo4jGraphStore
from app.retrieval.qdrant_store import QdrantVectorStore
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
def _real_stores(
    kind: str, database_url: str | None, qdrant_url: str | None, collection: str, neo4j_settings: tuple
) -> tuple[VectorStore, GraphStore]:
    if kind == "qdrant":
        vector_store = QdrantVectorStore.from_url(qdrant_url, collection)
    else:
        vector_store = PostgresVectorStore(database_url)
    return vector_store, Neo4jGraphStore.from_uri(*neo4j_settings)


def get_real_stores() -> tuple[VectorStore, GraphStore]:
    kind = config.vector_store()
    settings_by_kind = {"qdrant": config.qdrant_url(), "pgvector": config.database_url()}
    # 벡터 저장소는 비교 후 결정하므로 기본값을 두지 않는다
    if not settings_by_kind.get(kind) or not config.neo4j_uri() or not config.neo4j_password():
        raise AiUnavailableError("운영 저장소 설정이 없습니다.")
    return _real_stores(
        kind,
        config.database_url(),
        config.qdrant_url(),
        config.qdrant_collection(),
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

    embedding_lock = asyncio.Lock()
    embedding_attempted = False
    embedded = None
    embedding_error = None

    async def query_embedding() -> list[float]:
        nonlocal embedding_attempted, embedded, embedding_error
        async with embedding_lock:
            if not embedding_attempted:
                try:
                    embedded = await embed_query(query)
                except Exception as exc:
                    embedding_error = exc
                embedding_attempted = True
        if embedding_error is not None:
            raise embedding_error
        return embedded

    async def search_vector() -> list[Chunk]:
        return await vector_store.search(await query_embedding(), SOURCE_LIMIT, allowed_groups)

    async def search_graph() -> list[Chunk]:
        if isinstance(graph_store, Neo4jGraphStore):
            documents = await graph_store.search_documents(query, SOURCE_LIMIT, allowed_groups)
            if not documents:
                return []
            if not isinstance(vector_store, QdrantVectorStore):
                raise AiUnavailableError("문서 중심 그래프의 근거 회수에는 Qdrant가 필요합니다.")
            return await vector_store.search_in_documents(
                await query_embedding(), documents, SOURCE_LIMIT, allowed_groups
            )
        return await graph_store.search(query, SOURCE_LIMIT, allowed_groups)

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
    chunks = rrf_fuse(rankings, SOURCE_LIMIT * 2)
    if isinstance(graph_store, Neo4jGraphStore):
        # 이전 버전·철회된 권한·부분 적재 결과는 LLM에 전달하지 않는다.
        try:
            chunks = await graph_store.filter_current_chunks(chunks, allowed_groups)
        except Exception as exc:
            logger.error("Current document check failed: %s", describe_exception(exc))
            raise AiUnavailableError("문서의 현재 버전과 접근 범위를 확인할 수 없습니다.") from exc
    return {"chunks": chunks[:CONTEXT_LIMIT], "failed_sources": failed}
