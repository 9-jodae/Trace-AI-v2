import re

from neo4j import AsyncDriver, AsyncGraphDatabase, Query

from app.retrieval.relations import DocumentReference, document_names, normalize_name
from app.retrieval.schemas import Chunk

CONNECT_TIMEOUT_SECONDS = 5
QUERY_TIMEOUT_SECONDS = 5
MIN_NAME_LENGTH = 2
CHUNKS_PER_DOCUMENT_LIMIT = 5

# Neo4j에는 본문과 권한을 두지 않고 문서·청크 위치와 참조 관계만 저장한다.
SCHEMA_STATEMENTS = (
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
)
SEEDS_CYPHER = "MATCH (d:Document) RETURN d.document_id AS id, d.source AS source"
DOCUMENT_CATALOG_CYPHER = "MATCH (d:Document) RETURN d.document_id AS document_id, d.source AS source"
REFERENCE_SEARCH_CYPHER = """
CALL {
    UNWIND $documents AS document_id
    MATCH (d:Document {document_id: document_id})-[:HAS_CHUNK]->(c:Chunk)
    RETURN c.id AS id, d.document_id AS document_id, c.chunk_index AS chunk_index, 2 AS priority
    UNION ALL
    UNWIND $documents AS document_id
    MATCH (d:Document {document_id: document_id})-[r:REFERENCES]-(related:Document)
    MATCH (related)-[:HAS_CHUNK]->(c:Chunk)
    RETURN c.id AS id, related.document_id AS document_id, c.chunk_index AS chunk_index, 1 AS priority
    UNION ALL
    UNWIND $documents AS document_id
    MATCH (d:Document {document_id: document_id})-[r:REFERENCES]-(related:Document)
    MATCH (source:Document)-[:HAS_CHUNK]->(reference_chunk:Chunk {id: r.chunk_id})
    RETURN reference_chunk.id AS id, source.document_id AS document_id,
           reference_chunk.chunk_index AS chunk_index, 0 AS priority
}
WITH id, document_id, min(priority) AS priority, min(chunk_index) AS chunk_index
ORDER BY priority, document_id, chunk_index, id
WITH document_id, priority, collect(id)[0..$per_document_limit] AS ids
UNWIND ids AS id
RETURN id
ORDER BY priority, document_id, id
LIMIT $limit
"""
MERGE_DOCUMENT_CYPHER = """
MERGE (d:Document {document_id: $document_id})
SET d.source = $source, d.names = $names
"""
DELETE_DOCUMENT_CHUNKS_CYPHER = """
MATCH (:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
DETACH DELETE c
"""
CREATE_CHUNKS_CYPHER = """
MATCH (d:Document {document_id: $document_id})
UNWIND $chunks AS item
CREATE (c:Chunk {
    id: item.id, version_id: item.version_id, chunk_index: item.chunk_index,
    char_start: item.char_start, char_end: item.char_end
})
CREATE (d)-[:HAS_CHUNK]->(c)
"""
DELETE_REFERENCES_CYPHER = "MATCH (:Document {document_id: $document_id})-[r:REFERENCES]->() DELETE r"
CREATE_REFERENCES_CYPHER = """
UNWIND $links AS item
MATCH (source:Document {document_id: item.source_document_id})
MATCH (target:Document {document_id: item.target_document_id})
CREATE (source)-[:REFERENCES {chunk_id: item.chunk_id}]->(target)
"""


def name_in_query(name: str, query: str) -> bool:
    normalized = normalize_name(name)
    if len(normalized) < MIN_NAME_LENGTH:
        return False
    compact_name = "".join(normalized.split())
    compact_query = "".join(query.split())
    left = r"(?<![a-z0-9])" if compact_name[0].isascii() and compact_name[0].isalnum() else ""
    right = r"(?![a-z0-9])" if compact_name[-1].isascii() and compact_name[-1].isalnum() else ""
    return re.search(left + re.escape(compact_name) + right, compact_query) is not None


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

    async def _read(self, statement: str, parameters: dict) -> list[dict]:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(Query(statement, timeout=QUERY_TIMEOUT_SECONDS), parameters)
            return [dict(record) async for record in result]

    async def seed_documents(self, query: str) -> list[str]:
        normalized_query = normalize_name(query)
        if not normalized_query:
            return []
        seeds = await self._read(SEEDS_CYPHER, {})

        def matches_title(seed: dict) -> bool:
            document = {
                "document_id": seed["id"],
                "version_id": "",
                "text": "",
                "source": seed["source"] or "",
            }
            return any(name_in_query(name, normalized_query) for name in document_names(document))

        return [
            seed["id"]
            for seed in seeds
            if matches_title(seed)
        ]

    async def search_from_documents(self, document_ids: list[str], limit: int) -> list[str]:
        if not document_ids:
            return []
        rows = await self._read(
            REFERENCE_SEARCH_CYPHER,
            {"documents": document_ids, "limit": limit, "per_document_limit": CHUNKS_PER_DOCUMENT_LIMIT},
        )
        return [row["id"] for row in rows]

    async def list_document_catalog(self) -> list[dict[str, str]]:
        return await self._read(DOCUMENT_CATALOG_CYPHER, {})

    async def ensure_schema(self) -> None:
        async with self._driver.session(database=self._database) as session:
            for statement in SCHEMA_STATEMENTS:
                await (await session.run(statement)).consume()

    async def replace_document(self, document_id: str, source: str, names: list[str], chunks: list[Chunk]) -> None:
        async def write(tx) -> None:
            await (await tx.run(MERGE_DOCUMENT_CYPHER, document_id=document_id, source=source, names=names)).consume()
            await (await tx.run(DELETE_DOCUMENT_CHUNKS_CYPHER, document_id=document_id)).consume()
            items = [
                {key: chunk[key] for key in ("id", "version_id", "chunk_index", "char_start", "char_end")}
                for chunk in chunks
            ]
            if items:
                await (await tx.run(CREATE_CHUNKS_CYPHER, document_id=document_id, chunks=items)).consume()

        await self.ensure_schema()
        async with self._driver.session(database=self._database) as session:
            await session.execute_write(write)

    async def replace_references(self, document_id: str, links: list[DocumentReference]) -> None:
        async def write(tx) -> None:
            await (await tx.run(DELETE_REFERENCES_CYPHER, document_id=document_id)).consume()
            if links:
                await (await tx.run(CREATE_REFERENCES_CYPHER, links=links)).consume()

        async with self._driver.session(database=self._database) as session:
            await session.execute_write(write)
