import asyncio

from qdrant_client import AsyncQdrantClient, models

from app.retrieval.schemas import ACTIVE_STATUS, Chunk

TIMEOUT_SECONDS = 5
UPSERT_BATCH_SIZE = 64

PAYLOAD_FIELDS = ("id", "text", "source", "document_id", "version_id", "char_start", "char_end")


def chunk_payload(payload: dict) -> Chunk:
    chunk = {field: payload.get(field) for field in PAYLOAD_FIELDS}
    if "content_hash" in payload:
        chunk["content_hash"] = payload["content_hash"]
    return chunk


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
                    models.FieldCondition(key="index_status", match=models.MatchValue(value="READY")),
                ]
            ),
            limit=limit,
            with_payload=True,
        )
        return [chunk_payload(point.payload) for point in response.points]

    async def search_in_documents(
        self, embedding: list[float], documents: list[dict], limit: int, allowed_groups: list[str]
    ) -> list[Chunk]:
        """그래프의 문서 순서를 유지하고 각 문서에서 가장 관련 있는 청크 하나를 회수한다."""
        if not allowed_groups:
            return []
        semaphore = asyncio.Semaphore(4)

        async def retrieve(document: dict) -> list[Chunk]:
            async with semaphore:
                response = await self._client.query_points(
                    collection_name=self._collection, query=embedding, limit=1, with_payload=True,
                    query_filter=models.Filter(must=[
                        models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=document["document_id"])
                        ),
                        models.FieldCondition(key="version_id", match=models.MatchValue(value=document["version_id"])),
                        models.FieldCondition(key="status", match=models.MatchValue(value=ACTIVE_STATUS)),
                        models.FieldCondition(key="allowed_group_ids", match=models.MatchAny(any=allowed_groups)),
                        models.FieldCondition(key="index_status", match=models.MatchValue(value="READY")),
                    ]),
                )
            return [chunk_payload(point.payload) for point in response.points]

        results = await asyncio.gather(*(retrieve(document) for document in documents[:limit]))
        return [chunk for result in results for chunk in result]

    async def publish_version(self, document_id: str, version_id: str) -> None:
        await self._client.set_payload(
            self._collection, payload={"index_status": "READY"}, wait=True,
            points=models.Filter(must=[
                models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
                models.FieldCondition(key="version_id", match=models.MatchValue(value=version_id)),
            ]),
        )

    async def retire_version(self, document_id: str, version_id: str) -> None:
        # 다른 작업이 등록한 최신 버전을 지우지 않도록 이전 버전 ID 하나만 대상으로 한다.
        await self._client.delete(
            self._collection, wait=True,
            points_selector=models.Filter(must=[
                models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
                models.FieldCondition(key="version_id", match=models.MatchValue(value=version_id)),
            ]),
        )

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
        if not pairs or not embeddings[0]:
            raise ValueError("저장할 청크와 임베딩이 필요합니다.")
        document_ids = {chunk.get("document_id") for chunk in chunks}
        if (len(document_ids) != 1 or None in document_ids
                or any(chunk.get("version_id") != version_id for chunk in chunks)):
            raise ValueError("문서와 버전이 일치하지 않는 청크가 있습니다.")
        document_id = next(iter(document_ids))
        await self._ensure_collection(len(embeddings[0]))
        # 저장이 끝난 뒤 남은 청크만 지워, 중간에 실패해도 기존 청크가 사라지지 않게 한다
        for start in range(0, len(pairs), UPSERT_BATCH_SIZE):
            await self._client.upsert(
                self._collection,
                points=[
                    models.PointStruct(id=chunk["id"], vector=vector, payload={"index_status": "READY", **dict(chunk)})
                    for chunk, vector in pairs[start : start + UPSERT_BATCH_SIZE]
                ],
                wait=True,
            )
        await self._client.delete(
            self._collection,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(key="version_id", match=models.MatchValue(value=version_id)),
                    models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
                ],
                must_not=[models.HasIdCondition(has_id=[chunk["id"] for chunk in chunks])],
            ),
            wait=True,
        )
