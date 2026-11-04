import json
import re

from neo4j import AsyncDriver, AsyncGraphDatabase, Query

from app.retrieval.schemas import ACTIVE_STATUS, Chunk

CONNECT_TIMEOUT_SECONDS = 5
QUERY_TIMEOUT_SECONDS = 5
MAX_SEARCH_TERMS = 12

# 업무 그래프는 문서 노드만 사용한다. 세부 근거 청크는 Qdrant에서 회수한다.
SCHEMA_STATEMENTS = (
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE",
    "CREATE CONSTRAINT document_version_id IF NOT EXISTS FOR (d:Document) REQUIRE d.version_id IS UNIQUE",
    "CREATE FULLTEXT INDEX document_text IF NOT EXISTS FOR (d:Document) ON EACH [d.title, d.text]",
    "CALL db.awaitIndex('document_text', 300)",
)
FULLTEXT_SEARCH_CYPHER = """
CALL db.index.fulltext.queryNodes('document_text', $query) YIELD node AS seed, score
WHERE seed.status = $status AND seed.index_status = 'READY'
  AND any(g IN coalesce(seed.allowed_group_ids, []) WHERE g IN $allowed_groups)
WITH seed, score ORDER BY score DESC LIMIT $limit
CALL {
    WITH seed, score
    RETURN seed AS d, score AS relevance
    UNION ALL
    WITH seed, score
    MATCH (seed)-[r:REFERENCES]-(d:Document)
    WHERE r.source_version_id = startNode(r).version_id AND r.target_version_id = endNode(r).version_id
    RETURN d, score * 0.5 AS relevance
}
WITH d, max(relevance) AS relevance
WHERE d.status = $status AND d.index_status = 'READY'
  AND any(g IN coalesce(d.allowed_group_ids, []) WHERE g IN $allowed_groups)
RETURN d.document_id AS document_id, d.version_id AS version_id, d.title AS title
ORDER BY relevance DESC, document_id
LIMIT $limit
"""
MERGE_DOCUMENT_CYPHER = """
MERGE (d:Document {document_id: $document_id})
SET d.ingestion_revision = coalesce(d.ingestion_revision, 0) + 1
WITH d, d.version_id AS previous_version
WHERE (d.version_number IS NULL OR d.version_number <= $properties.version_number)
  AND (d.version_number IS NULL OR d.version_number <> $properties.version_number
       OR d.version_id = $properties.version_id)
  AND (d.version_id IS NULL OR d.version_id <> $properties.version_id
       OR d.content_hash IS NULL OR d.content_hash = $properties.content_hash)
SET d += $properties
RETURN previous_version
"""
DELETE_REFERENCES_CYPHER = "MATCH (d:Document {document_id: $document_id})-[r:REFERENCES]->() DELETE r"
# 기존 데모를 재적재할 때 이 문서의 기술적 Chunk 노드만 정리한다.
DELETE_LEGACY_CHUNKS_CYPHER = "MATCH (:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk) DETACH DELETE c"
CREATE_REFERENCES_CYPHER = """
MATCH (d:Document {document_id: $document_id})
UNWIND $references AS item
MATCH (target:Document {document_id: item.target_document_id, version_id: item.target_version_id})
WHERE target.status = $status AND target.index_status = 'READY'
  AND any(g IN coalesce(target.allowed_group_ids, []) WHERE g IN $allowed_groups)
MERGE (d)-[r:REFERENCES]->(target)
SET r.source_version_id = d.version_id, r.target_version_id = target.version_id,
    r.evidence_quote = item.quote, r.char_start = item.char_start, r.char_end = item.char_end,
    r.method = 'llm_grounded_title_match'
RETURN count(r) AS created
"""
FIND_TITLE_CYPHER = """
MATCH (d:Document)
WHERE toLower(d.title) = toLower($title) AND d.document_id <> $source_document_id
  AND d.status = $status AND d.index_status = 'READY'
  AND any(g IN coalesce(d.allowed_group_ids, []) WHERE g IN $allowed_groups)
RETURN d.document_id AS document_id, d.version_id AS version_id, d.title AS title
LIMIT 2
"""
FILTER_CHUNKS_CYPHER = """
UNWIND $chunks AS item
MATCH (d:Document {document_id: item.document_id, version_id: item.version_id})
WHERE d.status = $status AND d.index_status = 'READY'
  AND d.content_hash = item.content_hash
  AND any(g IN coalesce(d.allowed_group_ids, []) WHERE g IN $allowed_groups)
RETURN item.id AS id
"""


class DocumentVersionConflict(ValueError):
    """구버전 또는 동일 버전의 다른 본문을 덮어쓰려는 요청."""


def build_fulltext_query(text: str) -> str:
    terms = re.findall(r"[\w가-힣]+", text, flags=re.UNICODE)[:MAX_SEARCH_TERMS]
    return " OR ".join(f'"{term}"' for term in terms)


def document_properties(document: dict) -> dict:
    # Neo4j property에는 중첩 객체를 넣을 수 없으므로 검증 기록만 JSON 문자열로 보관한다.
    plain = (
        "document_id", "version_id", "version_number", "title", "source", "source_uri", "text", "content_hash",
        "status", "allowed_group_ids", "author", "authored_at", "source_modified_at", "department",
        "document_type", "classification_status", "classification_reason", "schema_version", "review_reasons",
        "index_status",
    )
    result = {key: document.get(key) for key in plain}
    result["title"] = document.get("title") or document.get("source")
    result["version_number"] = document.get("version_number", 1)
    result["index_status"] = document.get("index_status", "READY")
    result["topics"] = [item["value"] for item in document.get("topics", [])]
    for key in ("classification_evidence", "topics", "references", "field_sources"):
        result[key + "_json"] = json.dumps(document.get(key), ensure_ascii=False)
    return result


class Neo4jGraphStore:
    def __init__(self, driver: AsyncDriver, database: str | None = None) -> None:
        self._driver = driver
        self._database = database

    @classmethod
    def from_uri(cls, uri: str, user: str, password: str, database: str | None) -> "Neo4jGraphStore":
        driver = AsyncGraphDatabase.driver(uri, auth=(user, password), connection_timeout=CONNECT_TIMEOUT_SECONDS)
        return cls(driver, database)

    async def close(self) -> None:
        await self._driver.close()

    async def search_documents(self, query: str, limit: int, allowed_groups: list[str]) -> list[dict]:
        fulltext_query = build_fulltext_query(query)
        if not allowed_groups or not fulltext_query:
            return []
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                Query(FULLTEXT_SEARCH_CYPHER, timeout=QUERY_TIMEOUT_SECONDS),
                {"query": fulltext_query, "status": ACTIVE_STATUS, "allowed_groups": allowed_groups, "limit": limit},
            )
            return [dict(record) async for record in result]

    async def find_reference_documents(
        self, title: str, source_document_id: str, allowed_groups: list[str]
    ) -> list[dict]:
        if not allowed_groups:
            return []
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                Query(FIND_TITLE_CYPHER, timeout=QUERY_TIMEOUT_SECONDS),
                title=title, source_document_id=source_document_id, allowed_groups=allowed_groups, status=ACTIVE_STATUS,
            )
            return [dict(record) async for record in result]

    async def filter_current_chunks(self, chunks: list[Chunk], allowed_groups: list[str]) -> list[Chunk]:
        if not chunks or not allowed_groups:
            return []
        identifiers = [
            {key: chunk.get(key) for key in ("id", "document_id", "version_id", "content_hash")} for chunk in chunks
        ]
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                Query(FILTER_CHUNKS_CYPHER, timeout=QUERY_TIMEOUT_SECONDS),
                chunks=identifiers, allowed_groups=allowed_groups, status=ACTIVE_STATUS,
            )
            allowed = {record["id"] async for record in result}
        return [chunk for chunk in chunks if chunk["id"] in allowed]

    async def ensure_schema(self) -> None:
        async with self._driver.session(database=self._database) as session:
            for statement in SCHEMA_STATEMENTS:
                await (await session.run(statement)).consume()

    async def mark_ready(self, document_id: str, version_id: str, content_hash: str) -> None:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                Query("""MATCH (d:Document {document_id: $document_id, version_id: $version_id,
                    content_hash: $content_hash})
                    SET d.index_status = 'READY' RETURN d.document_id AS document_id""", timeout=QUERY_TIMEOUT_SECONDS),
                document_id=document_id, version_id=version_id, content_hash=content_hash,
            )
            if await result.single() is None:
                raise DocumentVersionConflict("저장 도중 문서 버전이 변경되어 재시도가 필요합니다.")

    async def replace_version(
        self, version_id: str, document_id: str, document: dict, chunks: list[Chunk]
    ) -> str | None:
        properties = document_properties({**document, "document_id": document_id, "version_id": version_id})
        references = [
            {
                "target_document_id": item["target_document_id"], "target_version_id": item["target_version_id"],
                **item["evidence"],
            }
            for item in document.get("references", []) if item.get("status") == "VALIDATED"
        ]

        async def write(tx) -> str | None:
            result = await tx.run(MERGE_DOCUMENT_CYPHER, document_id=document_id, properties=properties)
            row = await result.single()
            if row is None:
                raise DocumentVersionConflict("문서 버전 또는 본문이 현재 저장된 정보와 충돌합니다.")
            await (await tx.run(DELETE_LEGACY_CHUNKS_CYPHER, document_id=document_id)).consume()
            await (await tx.run(DELETE_REFERENCES_CYPHER, document_id=document_id)).consume()
            if references:
                result = await tx.run(
                    CREATE_REFERENCES_CYPHER, document_id=document_id, references=references,
                    status=ACTIVE_STATUS, allowed_groups=document["allowed_group_ids"],
                )
                created = await result.single()
                if created is None or created["created"] != len(references):
                    raise DocumentVersionConflict("참조 대상의 버전 또는 권한이 변경되어 재검증이 필요합니다.")
            return row["previous_version"]

        await self.ensure_schema()
        async with self._driver.session(database=self._database) as session:
            return await session.execute_write(write)
