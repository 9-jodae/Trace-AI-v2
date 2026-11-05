from neo4j import AsyncDriver, AsyncGraphDatabase, Query

from app.retrieval.entities import Mention
from app.retrieval.relations import DocumentLink, hop_depth, name_in_query, normalize_name
from app.retrieval.schemas import Chunk

CONNECT_TIMEOUT_SECONDS = 5
QUERY_TIMEOUT_SECONDS = 5

# 그래프에는 권한과 본문을 복사하지 않고 위치만 둔다. 찾은 청크 ID는 PostgreSQL 권한 함수를 거쳐 본문을 읽는다
SCHEMA_STATEMENTS = (
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
)
SEEDS_CYPHER = """
MATCH (d:Document)
RETURN 'document' AS kind, d.document_id AS id, d.names AS names
UNION ALL
MATCH (e:Entity)
RETURN 'entity' AS kind, e.entity_id AS id, [e.normalized_name] AS names
"""
GRAPH_SEARCH_CYPHER = """
CALL {
    MATCH (d:Document)-[:REFERENCES]-(:Document)-[:HAS_CHUNK]->(c:Chunk)
    WHERE d.document_id IN $documents
    RETURN c.id AS id, 0 AS priority
    UNION ALL
    MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
    WHERE d.document_id IN $documents
    RETURN c.id AS id, 1 AS priority
    UNION ALL
    MATCH (e:Entity)<-[:MENTIONS]-(c:Chunk)
    WHERE e.entity_id IN $entities
    RETURN c.id AS id, 1 AS priority
    UNION ALL
    MATCH (e:Entity)<-[:MENTIONS]-(:Chunk)<-[:HAS_CHUNK]-(:Document)-[:REFERENCES]-(:Document)-[:HAS_CHUNK]->(c:Chunk)
    WHERE e.entity_id IN $entities
    RETURN c.id AS id, 2 AS priority
    UNION ALL
    MATCH (e:Entity)<-[:MENTIONS]-(:Chunk)<-[:HAS_CHUNK]-(:Document)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(other:Entity)
          <-[:MENTIONS]-(c:Chunk)
    WHERE $depth = 2 AND e.entity_id IN $entities AND other <> e
    RETURN c.id AS id, 3 AS priority
}
WITH id, min(priority) AS priority
RETURN id
ORDER BY priority, id
LIMIT $limit
"""
MERGE_DOCUMENT_CYPHER = """
MERGE (d:Document {document_id: $document_id})
SET d.source = $source, d.names = $names
"""
# 문서당 현재 버전만 그래프에 둔다. 이전 버전 청크는 새 버전을 쓸 때 지운다
DELETE_DOCUMENT_CHUNKS_CYPHER = "MATCH (:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk) DETACH DELETE c"
CREATE_CHUNKS_CYPHER = """
MATCH (d:Document {document_id: $document_id})
UNWIND $chunks AS item
CREATE (c:Chunk {id: item.id, version_id: item.version_id, char_start: item.char_start, char_end: item.char_end})
CREATE (d)-[:HAS_CHUNK]->(c)
"""
CREATE_MENTIONS_CYPHER = """
UNWIND $mentions AS item
MATCH (c:Chunk {id: item.chunk_id})
MERGE (e:Entity {entity_id: item.entity_id})
SET e.entity_type = item.entity_type, e.name = item.name, e.normalized_name = item.normalized_name
CREATE (c)-[:MENTIONS {char_start: item.char_start, char_end: item.char_end}]->(e)
"""
DELETE_REFERENCES_CYPHER = "MATCH (:Document {document_id: $document_id})-[r:REFERENCES]->() DELETE r"
CREATE_REFERENCES_CYPHER = """
UNWIND $links AS item
MATCH (source:Document {document_id: item.source_document_id})
MATCH (target:Document {document_id: item.target_document_id})
CREATE (source)-[:REFERENCES {chunk_id: item.chunk_id, char_start: item.char_start, char_end: item.char_end}]->(target)
"""


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
            # run()의 첫 인자 이름이 query라서 키워드 인자로 넘기면 충돌한다
            result = await session.run(Query(statement, timeout=QUERY_TIMEOUT_SECONDS), parameters)
            return [dict(record) async for record in result]

    async def search(self, query: str, limit: int) -> list[dict]:
        normalized = normalize_name(query)
        seeds = await self._read(SEEDS_CYPHER, {})
        matched = {"document": [], "entity": []}
        for seed in seeds:
            if any(name_in_query(name, normalized) for name in seed["names"] or []):
                matched[seed["kind"]].append(seed["id"])
        if not matched["document"] and not matched["entity"]:
            return []
        return await self._read(
            GRAPH_SEARCH_CYPHER,
            {
                "documents": matched["document"],
                "entities": matched["entity"],
                "depth": hop_depth(normalized),
                "limit": limit,
            },
        )

    async def ensure_schema(self) -> None:
        async with self._driver.session(database=self._database) as session:
            for statement in SCHEMA_STATEMENTS:
                await (await session.run(statement)).consume()

    async def replace_version(
        self, document_id: str, source: str, names: list[str], chunks: list[Chunk], mentions: list[Mention]
    ) -> None:
        async def write(tx) -> None:
            await (await tx.run(MERGE_DOCUMENT_CYPHER, document_id=document_id, source=source, names=names)).consume()
            await (await tx.run(DELETE_DOCUMENT_CHUNKS_CYPHER, document_id=document_id)).consume()
            items = [{key: chunk[key] for key in ("id", "version_id", "char_start", "char_end")} for chunk in chunks]
            await (await tx.run(CREATE_CHUNKS_CYPHER, document_id=document_id, chunks=items)).consume()
            if mentions:
                await (await tx.run(CREATE_MENTIONS_CYPHER, mentions=mentions)).consume()

        await self.ensure_schema()
        async with self._driver.session(database=self._database) as session:
            await session.execute_write(write)

    async def replace_references(self, document_id: str, links: list[DocumentLink]) -> None:
        async def write(tx) -> None:
            await (await tx.run(DELETE_REFERENCES_CYPHER, document_id=document_id)).consume()
            if links:
                await (await tx.run(CREATE_REFERENCES_CYPHER, links=links)).consume()

        async with self._driver.session(database=self._database) as session:
            await session.execute_write(write)
