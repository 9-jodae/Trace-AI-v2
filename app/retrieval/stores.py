import math
import re
from typing import Protocol

import psycopg
from psycopg.rows import dict_row

from app.core.schemas import AccessScope, SearchFilters
from app.retrieval.embedding import embed_documents
from app.retrieval.schemas import Chunk

MAX_FULLTEXT_TERMS = 12
# chunks.embedding 컬럼의 차원 (docs199-Server V11)
CHUNK_EMBEDDING_DIMENSIONS = 1024
# 권한과 필터 판단은 서버 함수에 맡긴다. 삭제된 문서와 이전 버전도 함수가 제외한다 (docs199-Server V12, V13)
VECTOR_SEARCH_SQL = """
SELECT c.id::text AS id, c.content AS text, d.title AS source, dv.document_id::text AS document_id,
       c.document_version_id::text AS version_id, c.char_start, c.char_end
FROM accessible_chunks(%s::uuid, %s::text, %s::text, %s::boolean, %s::text[], %s::text[], %s::text[], %s::date, %s::date) c
JOIN document_versions dv ON dv.id = c.document_version_id
JOIN documents d ON d.id = dv.document_id
WHERE c.embedding IS NOT NULL
ORDER BY c.embedding <=> %s::vector
LIMIT %s
"""
FULLTEXT_SEARCH_SQL = """
SELECT c.id::text AS id, c.content AS text, d.title AS source, dv.document_id::text AS document_id,
       c.document_version_id::text AS version_id, c.char_start, c.char_end
FROM accessible_chunks(%s::uuid, %s::text, %s::text, %s::boolean, %s::text[], %s::text[], %s::text[], %s::date, %s::date) c
JOIN document_versions dv ON dv.id = c.document_version_id
JOIN documents d ON d.id = dv.document_id
WHERE to_tsvector('simple', c.content) @@ to_tsquery('simple', %s)
ORDER BY ts_rank_cd(to_tsvector('simple', c.content), to_tsquery('simple', %s)) DESC, c.id
LIMIT %s
"""
FILTER_ACCESSIBLE_SQL = """
SELECT chunk_id::text AS id
FROM filter_accessible_chunk_ids(%s::uuid, %s::text, %s::text, %s::boolean, %s::uuid[]) AS chunk_id
"""
# filter_accessible_chunk_ids에는 필터 인자가 없어 필터가 있으면 accessible_chunks로 대조한다
FILTER_ACCESSIBLE_WITH_FILTERS_SQL = """
SELECT c.id::text AS id
FROM accessible_chunks(%s::uuid, %s::text, %s::text, %s::boolean, %s::text[], %s::text[], %s::text[], %s::date, %s::date) c
WHERE c.id = ANY(%s::uuid[])
"""
CONNECT_TIMEOUT_SECONDS = 5
SEARCH_STATEMENT_TIMEOUT_MS = 5_000
WRITE_STATEMENT_TIMEOUT_MS = 60_000
DELETE_VERSION_SQL = "DELETE FROM chunks WHERE document_version_id = %s"
INSERT_CHUNK_SQL = """
INSERT INTO chunks (id, document_version_id, chunk_index, content, char_start, char_end, content_hash, embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
"""

# USE_SAMPLE_DATA=true인 로컬 데모에서만 쓰는 샘플 데이터
SAMPLE_CHUNKS: list[Chunk] = [
    {"id": "remote-1", "source": "재택근무 규정", "text": "재택근무 규정은 인사팀이 소유하며 보안팀이 승인했다."},
    {"id": "remote-2", "source": "재택근무 규정", "text": "재택근무 시 사내 자료는 VPN에 접속한 회사 노트북에서만 열람한다."},
    {"id": "remote-3", "source": "재택근무 규정", "text": "재택근무 규정은 2026년에 원격 접속 조항을 추가하며 개정되었다."},
    {"id": "payment-1", "source": "결제 서비스 문서", "text": "결제 서비스는 플랫폼팀이 소유하며 인증 서비스에 의존한다."},
    {"id": "payment-2", "source": "결제 서비스 문서", "text": "E4012 오류는 결제 서비스의 인증 토큰이 만료되었을 때 발생한다."},
]
SAMPLE_ENTITIES = {
    "재택근무 규정": ["remote-1", "remote-3"],
    "결제 서비스": ["payment-1", "payment-2"],
}


def build_fulltext_query(text: str) -> str:
    # 사용자 입력은 단어만 뽑아 tsquery 구문으로 해석되지 않게 한다
    terms = re.findall(r"\w+", text, flags=re.UNICODE)[:MAX_FULLTEXT_TERMS]
    return " | ".join(f"'{term}':*" for term in terms)


class VectorStore(Protocol):
    async def search(
        self, embedding: list[float], limit: int, scope: AccessScope | None, filters: SearchFilters | None = None
    ) -> list[Chunk]: ...

    async def fulltext(
        self, query: str, limit: int, scope: AccessScope | None, filters: SearchFilters | None = None
    ) -> list[Chunk]: ...

    async def filter_accessible(
        self, chunk_ids: list[str], scope: AccessScope | None, filters: SearchFilters | None = None
    ) -> set[str]: ...


class GraphStore(Protocol):
    # 그래프에는 권한 정보가 없으므로 결과는 VectorStore.filter_accessible을 거쳐야 내보낼 수 있다
    async def search(self, query: str, limit: int) -> list[Chunk]: ...


def _to_vector(embedding: list[float]) -> str:
    return "[" + ",".join(map(str, embedding)) + "]"


def _filter_params(filters: SearchFilters | None) -> tuple:
    # 필터를 끄려면 서버 함수에 명시적으로 NULL을 넘긴다 (docs199-Server ai-server-contract)
    if filters is None:
        return (None, None, None, None, None)
    return (
        filters.extensions,
        filters.document_types,
        filters.source_kinds,
        filters.authored_from,
        filters.authored_to,
    )


def _cosine(a: list[float], b: list[float]) -> float:
    norm = math.hypot(*a) * math.hypot(*b)
    return sum(x * y for x, y in zip(a, b)) / norm if norm else 0.0


class PostgresVectorStore:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    async def close(self) -> None:
        pass

    async def _read(self, sql: str, params: tuple) -> list[dict]:
        async with await psycopg.AsyncConnection.connect(
            self._database_url,
            row_factory=dict_row,
            connect_timeout=CONNECT_TIMEOUT_SECONDS,
            options=f"-c statement_timeout={SEARCH_STATEMENT_TIMEOUT_MS}",
        ) as connection:
            cursor = await connection.execute(sql, params)
            return await cursor.fetchall()

    async def search(
        self, embedding: list[float], limit: int, scope: AccessScope | None, filters: SearchFilters | None = None
    ) -> list[Chunk]:
        if scope is None:
            return []
        return await self._read(VECTOR_SEARCH_SQL, (*scope, *_filter_params(filters), _to_vector(embedding), limit))

    async def fulltext(
        self, query: str, limit: int, scope: AccessScope | None, filters: SearchFilters | None = None
    ) -> list[Chunk]:
        expression = build_fulltext_query(query)
        if scope is None or not expression:
            return []
        return await self._read(FULLTEXT_SEARCH_SQL, (*scope, *_filter_params(filters), expression, expression, limit))

    async def filter_accessible(
        self, chunk_ids: list[str], scope: AccessScope | None, filters: SearchFilters | None = None
    ) -> set[str]:
        if scope is None or not chunk_ids:
            return set()
        if filters is None:
            rows = await self._read(FILTER_ACCESSIBLE_SQL, (*scope, chunk_ids))
        else:
            rows = await self._read(FILTER_ACCESSIBLE_WITH_FILTERS_SQL, (*scope, *_filter_params(filters), chunk_ids))
        return {row["id"] for row in rows}

    async def replace_version(self, version_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        # 재색인 시 줄어든 청크가 남지 않도록 같은 버전의 청크를 한 트랜잭션에서 교체한다
        pairs = list(zip(chunks, embeddings, strict=True))
        async with await psycopg.AsyncConnection.connect(
            self._database_url,
            connect_timeout=CONNECT_TIMEOUT_SECONDS,
            options=f"-c statement_timeout={WRITE_STATEMENT_TIMEOUT_MS}",
        ) as connection:
            await connection.execute(DELETE_VERSION_SQL, (version_id,))
            async with connection.cursor() as cursor:
                await cursor.executemany(
                    INSERT_CHUNK_SQL,
                    [
                        (
                            chunk["id"],
                            chunk["version_id"],
                            chunk["chunk_index"],
                            chunk["text"],
                            chunk["char_start"],
                            chunk["char_end"],
                            chunk["content_hash"],
                            _to_vector(embedding),
                        )
                        for chunk, embedding in pairs
                    ],
                )


class NoGraphStore:
    """NEO4J_URI를 비워 벡터 검색만 쓸 때의 그래프 저장소."""

    async def close(self) -> None:
        pass

    async def search(self, query: str, limit: int) -> list[Chunk]:
        return []

    async def replace_version(self, version_id: str, document_id: str, document: dict, chunks: list[Chunk]) -> None:
        pass


class SampleVectorStore:
    def __init__(self) -> None:
        self._embeddings: list[list[float]] | None = None

    async def search(
        self, embedding: list[float], limit: int, scope: AccessScope | None, filters: SearchFilters | None = None
    ) -> list[Chunk]:
        # 샘플 문서에는 필터 대상 정보가 없어 필터를 적용하지 않는다
        if self._embeddings is None:
            self._embeddings = await embed_documents([chunk["text"] for chunk in SAMPLE_CHUNKS])
        scored = zip(SAMPLE_CHUNKS, self._embeddings)
        ranked = sorted(scored, key=lambda item: _cosine(embedding, item[1]), reverse=True)
        return [chunk for chunk, _ in ranked[:limit]]

    async def fulltext(
        self, query: str, limit: int, scope: AccessScope | None, filters: SearchFilters | None = None
    ) -> list[Chunk]:
        return []

    async def filter_accessible(
        self, chunk_ids: list[str], scope: AccessScope | None, filters: SearchFilters | None = None
    ) -> set[str]:
        return set(chunk_ids)


class SampleGraphStore:
    async def search(self, query: str, limit: int) -> list[Chunk]:
        normalized = "".join(query.split()).casefold()
        linked = [
            chunk_id
            for name, chunk_ids in SAMPLE_ENTITIES.items()
            if "".join(name.split()).casefold() in normalized
            for chunk_id in chunk_ids
        ]
        return [chunk for chunk in SAMPLE_CHUNKS if chunk["id"] in linked][:limit]
