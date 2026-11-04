import math
from typing import Protocol

import psycopg
from psycopg.rows import dict_row

from app.retrieval.embedding import embed_documents
from app.retrieval.schemas import Chunk

# TODO: 실제 스키마에 맞게 수정 (pgvector 확장과 chunks(id, text, source, embedding vector(1536)) 테이블 가정)
VECTOR_SEARCH_SQL = """
SELECT id::text AS id, text, source
FROM chunks
ORDER BY embedding <=> %s::vector
LIMIT %s
"""
DELETE_DOCUMENT_SQL = "DELETE FROM chunks WHERE source = %s"
INSERT_CHUNK_SQL = "INSERT INTO chunks (id, text, source, embedding) VALUES (%s, %s, %s, %s::vector)"

# TODO: 그래프 DB 구현으로 교체하고, DATABASE_URL이 없을 때만 쓰는 샘플 데이터는 삭제
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
        async with await psycopg.AsyncConnection.connect(self._database_url, row_factory=dict_row) as connection:
            cursor = await connection.execute(VECTOR_SEARCH_SQL, (_to_vector(embedding), limit))
            return await cursor.fetchall()

    async def replace_document(self, source: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        # 재색인 시 줄어든 청크가 남지 않도록 같은 문서의 청크를 한 트랜잭션에서 교체한다
        async with await psycopg.AsyncConnection.connect(self._database_url) as connection:
            await connection.execute(DELETE_DOCUMENT_SQL, (source,))
            for chunk, embedding in zip(chunks, embeddings):
                await connection.execute(
                    INSERT_CHUNK_SQL, (chunk["id"], chunk["text"], chunk["source"], _to_vector(embedding))
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
