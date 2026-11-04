from qdrant_client import AsyncQdrantClient, models

from app.retrieval.schemas import ACTIVE_STATUS, Chunk

TIMEOUT_SECONDS = 5
UPSERT_BATCH_SIZE = 64

PAYLOAD_FIELDS = ("id", "text", "source", "document_id", "version_id", "char_start", "char_end")


class QdrantVectorStore:
    def __init__(self, client: AsyncQdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection

    @classmethod
    def from_url(cls, url: str, collection: str) -> "QdrantVectorStore":
        return cls(AsyncQdrantClient(url=url, timeout=TIMEOUT_SECONDS), collection)

    async def close(self) -> None:
        await self._client.close()

    async def search(self, embedding: list[float], limit: int, allowed_groups: list[str]) -> list[Chunk]:
        if not allowed_groups:
            return []
        response = await self._client.query_points(
            collection_name=self._collection,
            query=embedding,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(key="status", match=models.MatchValue(value=ACTIVE_STATUS)),
                    models.FieldCondition(key="allowed_group_ids", match=models.MatchAny(any=allowed_groups)),
                ]
            ),
            limit=limit,
            with_payload=True,
        )
        return [{field: point.payload.get(field) for field in PAYLOAD_FIELDS} for point in response.points]

    async def _ensure_collection(self, dimensions: int) -> None:
        # PR #17과 달리 차원이 달라도 기존 컬렉션을 지우지 않고 오류로 멈춘다
        if await self._client.collection_exists(self._collection):
            info = await self._client.get_collection(self._collection)
            if info.config.params.vectors.size != dimensions:
                raise ValueError("Qdrant 컬렉션의 차원이 임베딩 차원과 다릅니다.")
            return
        await self._client.create_collection(
            self._collection,
            vectors_config=models.VectorParams(size=dimensions, distance=models.Distance.COSINE),
        )

    async def replace_version(self, version_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        pairs = list(zip(chunks, embeddings, strict=True))
        await self._ensure_collection(len(embeddings[0]))
        # 저장이 끝난 뒤 남은 청크만 지워, 중간에 실패해도 기존 청크가 사라지지 않게 한다
        for start in range(0, len(pairs), UPSERT_BATCH_SIZE):
            await self._client.upsert(
                self._collection,
                points=[
                    models.PointStruct(id=chunk["id"], vector=vector, payload=dict(chunk))
                    for chunk, vector in pairs[start : start + UPSERT_BATCH_SIZE]
                ],
                wait=True,
            )
        await self._client.delete(
            self._collection,
            points_selector=models.Filter(
                must=[models.FieldCondition(key="version_id", match=models.MatchValue(value=version_id))],
                must_not=[models.HasIdCondition(has_id=[chunk["id"] for chunk in chunks])],
            ),
            wait=True,
        )
