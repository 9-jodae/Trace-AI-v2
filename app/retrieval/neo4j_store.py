import re

from neo4j import AsyncDriver, AsyncGraphDatabase, Query

from app.retrieval.schemas import Chunk

CONNECT_TIMEOUT_SECONDS = 5
QUERY_TIMEOUT_SECONDS = 5
MAX_SEARCH_TERMS = 12

# 노드·관계·인덱스는 AI 팀이 관리한다 (docs199-Server ai-server-contract)
SCHEMA_STATEMENTS = (
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]",
    "CALL db.awaitIndex('chunk_text', 300)",
)
FULLTEXT_SEARCH_CYPHER = """
CALL db.index.fulltext.queryNodes('chunk_text', $query) YIELD node AS c, score
MATCH (d:Document)-[:HAS_CHUNK]->(c)
RETURN c.id AS id, c.text AS text, d.source AS source, d.document_id AS document_id,
       c.version_id AS version_id, c.char_start AS char_start, c.char_end AS char_end
ORDER BY score DESC
LIMIT $limit
"""
DELETE_VERSION_CYPHER = "MATCH (c:Chunk {version_id: $version_id}) DETACH DELETE c"
MERGE_DOCUMENT_CYPHER = """
MERGE (d:Document {document_id: $document.document_id})
SET d.source = $document.source
"""
CREATE_CHUNKS_CYPHER = """
MATCH (d:Document {document_id: $document_id})
UNWIND $chunks AS item
CREATE (c:Chunk {id: item.id, text: item.text, version_id: item.version_id,
                 char_start: item.char_start, char_end: item.char_end})
CREATE (d)-[:HAS_CHUNK]->(c)
"""


def build_fulltext_query(text: str) -> str:
    # Lucene 특수문자를 빼고 단어별 OR 검색으로 만든다
    terms = re.findall(r"[\w가-힣]+", text, flags=re.UNICODE)[:MAX_SEARCH_TERMS]
    return " OR ".join(f'"{term}"' for term in terms)


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

    async def search(self, query: str, limit: int) -> list[Chunk]:
        fulltext_query = build_fulltext_query(query)
        if not fulltext_query:
            return []
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                Query(FULLTEXT_SEARCH_CYPHER, timeout=QUERY_TIMEOUT_SECONDS),
                # run()의 첫 인자 이름이 query라서 키워드 인자로 넘기면 충돌한다
                {"query": fulltext_query, "limit": limit},
            )
            return [dict(record) async for record in result]

    async def ensure_schema(self) -> None:
        async with self._driver.session(database=self._database) as session:
            for statement in SCHEMA_STATEMENTS:
                await (await session.run(statement)).consume()

    async def replace_version(self, version_id: str, document_id: str, document: dict, chunks: list[Chunk]) -> None:
        async def write(tx) -> None:
            await (await tx.run(DELETE_VERSION_CYPHER, version_id=version_id)).consume()
            await (await tx.run(MERGE_DOCUMENT_CYPHER, document=document)).consume()
            await (await tx.run(CREATE_CHUNKS_CYPHER, document_id=document_id, chunks=chunks)).consume()

        await self.ensure_schema()
        async with self._driver.session(database=self._database) as session:
            await session.execute_write(write)
