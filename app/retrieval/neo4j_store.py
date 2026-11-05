import re

from neo4j import AsyncDriver, AsyncGraphDatabase, Query

from app.retrieval.relations import DocumentReference, document_names, normalize_name
from app.retrieval.schemas import Chunk

CONNECT_TIMEOUT_SECONDS = 5
QUERY_TIMEOUT_SECONDS = 5
MIN_NAME_LENGTH = 2
CHUNKS_PER_DOCUMENT_LIMIT = 5

# Neo4j에는 권한과 문서 전체를 두지 않는다. 엔티티 관계의 근거 인용은 위치와 함께 저장한다.
SCHEMA_STATEMENTS = (
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
    "CREATE CONSTRAINT relation_assertion_id IF NOT EXISTS FOR (r:RelationAssertion) REQUIRE r.assertion_id IS UNIQUE",
    "CREATE CONSTRAINT entity_alias_id IF NOT EXISTS FOR (a:EntityAlias) REQUIRE a.alias_id IS UNIQUE",
    "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
)
SEEDS_CYPHER = "MATCH (d:Document) RETURN d.document_id AS id, d.source AS source"
DOCUMENT_CATALOG_CYPHER = "MATCH (d:Document) RETURN d.document_id AS document_id, d.source AS source"
ENTITY_CATALOG_CYPHER = "MATCH (e:Entity) RETURN e.entity_id AS entity_id, e.canonical_name AS name"
ENTITY_MENTION_SEARCH_CYPHER = """
MATCH (:Document)-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity)
WHERE e.entity_id IN $entity_ids
RETURN c.id AS id, c.chunk_index AS chunk_index
ORDER BY chunk_index, id
LIMIT $limit
"""
ASSERTION_EVIDENCE_SEARCH_CYPHER = """
MATCH (d:Document)-[:HAS_ASSERTION]->(r:RelationAssertion)-[:SUPPORTED_BY]->(proof:Chunk)
MATCH (r)-[:SUBJECT|OBJECT]->(e:Entity)
WHERE e.entity_id IN $entity_ids AND r.predicate IN $predicates
RETURN proof.id AS id, proof.chunk_index AS chunk_index
ORDER BY chunk_index, id
LIMIT $limit
"""
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
REPLACE_ENTITY_GRAPH_CYPHER = """
MATCH (d:Document {document_id: $document_id})
OPTIONAL MATCH (d)-[:HAS_CHUNK]->(old_chunk:Chunk {version_id: $version_id})-[m:MENTIONS]->()
DELETE m
WITH d
OPTIONAL MATCH (d)-[:HAS_ASSERTION]->(old:RelationAssertion)
DETACH DELETE old
WITH d
UNWIND $mentions AS mention
MATCH (d)-[:HAS_CHUNK]->(c:Chunk {id: mention.chunk_id, version_id: $version_id})
MERGE (e:Entity {entity_id: mention.entity_id})
ON CREATE SET e.entity_type = mention.entity_type, e.canonical_name = mention.canonical_name
MERGE (c)-[m:MENTIONS {mention_id: mention.mention_id}]->(e)
SET m.surface_form = mention.surface_form, m.char_start = mention.char_start,
    m.char_end = mention.char_end, m.evidence_start = mention.evidence_start,
    m.evidence_end = mention.evidence_end, m.evidence_quote = mention.evidence_quote
WITH DISTINCT d
UNWIND $assertions AS assertion
MATCH (subject:Entity {entity_id: assertion.subject_entity_id})
MATCH (object:Entity {entity_id: assertion.object_entity_id})
MATCH (d)-[:HAS_CHUNK]->(c:Chunk {id: assertion.evidence_chunk_id, version_id: $version_id})
CREATE (r:RelationAssertion {
    assertion_id: assertion.assertion_id, predicate: assertion.predicate,
    source_document_id: assertion.source_document_id, source_version_id: assertion.source_version_id,
    evidence_chunk_id: assertion.evidence_chunk_id, evidence_start: assertion.evidence_start,
    evidence_end: assertion.evidence_end, evidence_quote: assertion.evidence_quote,
    polarity: assertion.polarity, modality: assertion.modality,
    condition_quote: assertion.condition_quote
})
MERGE (d)-[:HAS_ASSERTION]->(r)
MERGE (r)-[:SUBJECT]->(subject)
MERGE (r)-[:OBJECT]->(object)
MERGE (r)-[:SUPPORTED_BY]->(c)
"""


TOPIC_CATALOG_CYPHER = "MATCH (t:Topic) RETURN t.name AS name ORDER BY name"
REPLACE_TOPICS_CYPHER = """
MATCH (d:Document {document_id: $document_id})
OPTIONAL MATCH (d)-[old:HAS_TOPIC]->(:Topic)
DELETE old
WITH DISTINCT d
UNWIND $topics AS topic
MERGE (t:Topic {name: topic})
MERGE (d)-[:HAS_TOPIC]->(t)
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

    async def search_from_entities(self, query: str, limit: int) -> list[str]:
        """질문에서 실제 Entity 이름을 찾고 해당 이름이 언급된 청크를 시작점으로 반환한다."""
        catalog = await self._read(ENTITY_CATALOG_CYPHER, {})
        entity_ids = [row["entity_id"] for row in catalog if name_in_query(row["name"] or "", query)]
        if not entity_ids:
            return []
        assertion_terms = {
            "RESPONSIBLE_FOR": ("담당", "관리", "누가", "소유"),
            "USES": ("사용", "이용", "어디서"),
            "GOVERNED_BY": ("규정", "적용", "따르"),
        }
        predicates = [
            predicate for predicate, terms in assertion_terms.items() if any(term in query for term in terms)
        ]
        evidence_ids = []
        if predicates:
            rows = await self._read(
                ASSERTION_EVIDENCE_SEARCH_CYPHER,
                {"entity_ids": entity_ids, "predicates": predicates, "limit": limit},
            )
            evidence_ids = [row["id"] for row in rows]
        rows = await self._read(ENTITY_MENTION_SEARCH_CYPHER, {"entity_ids": entity_ids, "limit": limit})
        mention_ids = [row["id"] for row in rows]
        return list(dict.fromkeys([*evidence_ids, *mention_ids]))[:limit]

    async def list_topics(self) -> list[str]:
        return [row["name"] for row in await self._read(TOPIC_CATALOG_CYPHER, {})]

    async def replace_document_topics(self, document_id: str, topics: list[str]) -> None:
        async def write(tx) -> None:
            await (await tx.run(REPLACE_TOPICS_CYPHER, document_id=document_id, topics=topics)).consume()

        await self.ensure_schema()
        async with self._driver.session(database=self._database) as session:
            await session.execute_write(write)

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

    async def replace_entity_graph(
        self, document_id: str, version_id: str, mentions: list[dict], assertions: list[dict]
    ) -> None:
        """현재 적재 버전의 언급·업무 관계 주장을 원자적으로 교체한다."""
        async def write(tx) -> None:
            await (await tx.run(
                REPLACE_ENTITY_GRAPH_CYPHER,
                document_id=document_id,
                version_id=version_id,
                mentions=mentions,
                assertions=assertions,
            )).consume()

        await self.ensure_schema()
        async with self._driver.session(database=self._database) as session:
            await session.execute_write(write)
