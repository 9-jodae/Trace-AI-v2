import math
from typing import Protocol

import psycopg
from psycopg.rows import dict_row

from app.retrieval.embedding import embed_documents
from app.retrieval.schemas import Chunk

# TODO: 벡터 저장소 확정(#9)과 docs199-Server#15 스키마에 맞게 수정. 아래 컬럼은 가정이다
VECTOR_SEARCH_SQL = """
SELECT id::text AS id, text, source, document_id::text AS document_id, version_id::text AS version_id,
       char_start, char_end
FROM chunks
ORDER BY embedding <=> %s::vector
LIMIT %s
"""
CONNECT_TIMEOUT_SECONDS = 5
DELETE_VERSION_SQL = "DELETE FROM chunks WHERE version_id = %s"
INSERT_CHUNK_SQL = """
INSERT INTO chunks (id, text, source, document_id, version_id, char_start, char_end, embedding)
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


class VectorStore(Protocol):
    async def search(self, embedding: list[float], limit: int) -> list[Chunk]: ...


class GraphStore(Protocol):
    async def search(self, query: str, limit: int) -> list[Chunk]: ...


def _to_vector(embedding: list[float]) -> str:
    return "[" + ",".join(map(str, embedding)) + "]"


def _cosine(a: list[float], b: list[float]) -> float:
    norm = math.hypot(*a) * math.hypot(*b)
    return sum(x * y for x, y in zip(a, b)) / norm if norm else 0.0


class PostgresVectorStore:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    async def search(self, embedding: list[float], limit: int) -> list[Chunk]:
        async with await psycopg.AsyncConnection.connect(
            self._database_url, row_factory=dict_row, connect_timeout=CONNECT_TIMEOUT_SECONDS
        ) as connection:
            cursor = await connection.execute(VECTOR_SEARCH_SQL, (_to_vector(embedding), limit))
            return await cursor.fetchall()

    async def replace_version(self, version_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        # 재색인 시 줄어든 청크가 남지 않도록 같은 버전의 청크를 한 트랜잭션에서 교체한다
        pairs = list(zip(chunks, embeddings, strict=True))
        async with await psycopg.AsyncConnection.connect(
            self._database_url, connect_timeout=CONNECT_TIMEOUT_SECONDS
        ) as connection:
            await connection.execute(DELETE_VERSION_SQL, (version_id,))
            for chunk, embedding in pairs:
                await connection.execute(
                    INSERT_CHUNK_SQL,
                    (
                        chunk["id"],
                        chunk["text"],
                        chunk["source"],
                        chunk["document_id"],
                        chunk["version_id"],
                        chunk["char_start"],
                        chunk["char_end"],
                        _to_vector(embedding),
                    ),
                )


class SampleVectorStore:
    def __init__(self) -> None:
        self._embeddings: list[list[float]] | None = None

    async def search(self, embedding: list[float], limit: int) -> list[Chunk]:
        if self._embeddings is None:
            self._embeddings = await embed_documents([chunk["text"] for chunk in SAMPLE_CHUNKS])
        scored = zip(SAMPLE_CHUNKS, self._embeddings)
        ranked = sorted(scored, key=lambda item: _cosine(embedding, item[1]), reverse=True)
        return [chunk for chunk, _ in ranked[:limit]]


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
