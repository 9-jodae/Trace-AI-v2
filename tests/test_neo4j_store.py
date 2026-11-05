"""Neo4j stores only graph identifiers; driver calls are checked with a test double."""

import asyncio
import inspect

from neo4j import AsyncManagedTransaction, AsyncSession

from app.retrieval import neo4j_store
from app.retrieval.neo4j_store import Neo4jGraphStore


class FakeResult:
    def __init__(self, rows=()):
        self.rows = rows

    def __aiter__(self):
        async def generate():
            for row in self.rows:
                yield row

        return generate()

    async def consume(self):
        pass


class FakeSession:
    def __init__(self, driver):
        self.driver = driver

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def run(self, query, parameters=None, **params):
        call_params = {**(parameters or {}), **params}
        self.driver.calls.append((query, call_params))
        query_text = getattr(query, "text", query)
        if query_text == neo4j_store.REFERENCE_SEARCH_CYPHER:
            return FakeResult(self.driver.results)
        return FakeResult()

    async def execute_write(self, work):
        self.driver.transactions += 1
        await work(self)


class FakeDriver:
    def __init__(self, seeds=(), results=()):
        self.seeds = seeds
        self.results = results
        self.calls = []
        self.transactions = 0
        self.closed = False

    def session(self, database=None):
        self.database = database
        return FakeSession(self)

    async def close(self):
        self.closed = True


def test_search_expands_supplied_hybrid_seed_documents_to_chunk_ids():
    driver = FakeDriver(results=[{"id": "chunk-1"}, {"id": "chunk-2"}])
    store = Neo4jGraphStore(driver, "db")
    result = asyncio.run(store.search_from_documents(["doc-keyword", "doc-vector"], 50))
    assert result == ["chunk-1", "chunk-2"]
    assert len(driver.calls) == 1
    query, params = driver.calls[0]
    assert query.text == neo4j_store.REFERENCE_SEARCH_CYPHER
    assert query.timeout == neo4j_store.QUERY_TIMEOUT_SECONDS
    assert params == {
        "documents": ["doc-keyword", "doc-vector"],
        "limit": 50,
        "per_document_limit": neo4j_store.CHUNKS_PER_DOCUMENT_LIMIT,
    }
    assert driver.database == "db"


def test_query_text_is_not_interpolated_into_cypher():
    driver = FakeDriver(results=[])
    store = Neo4jGraphStore(driver)
    asyncio.run(store.search_from_documents(["doc-a\") DETACH DELETE"], 10))
    assert all("DETACH DELETE" not in str(query) for query, _ in driver.calls)
    assert driver.calls[-1][1]["documents"] == ["doc-a\") DETACH DELETE"]


def test_reference_traversal_is_one_hop_bounded_and_returns_no_content():
    query = neo4j_store.REFERENCE_SEARCH_CYPHER
    assert query.strip().startswith("CALL () {")
    assert "REFERENCES" in query and "LIMIT $limit" in query
    assert "2.." not in query and "c.text" not in query and "quote" not in query
    assert "collect(id)[0..$per_document_limit]" in query
    assert "ORDER BY priority, document_id, chunk_index" in query
    assert "0 AS priority" in query and "1 AS priority" in query and "2 AS priority" in query


def test_replace_document_persists_only_identifiers_and_offsets():
    driver = FakeDriver()
    chunks = [{
        "id": "c1", "text": "PRIVATE BODY", "version_id": "v1", "chunk_index": 0,
        "char_start": 0, "char_end": 12,
    }]
    asyncio.run(Neo4jGraphStore(driver).replace_document("d1", "Title", ["title"], chunks))
    assert driver.transactions == 1
    writes = [
        (getattr(query, "text", query), params)
        for query, params in driver.calls
        if getattr(query, "text", query) not in neo4j_store.SCHEMA_STATEMENTS
    ]
    created = next(params for query, params in writes if "UNWIND $chunks" in query)
    expected = {
        "document_id": "d1",
        "chunks": [{"id": "c1", "version_id": "v1", "chunk_index": 0, "char_start": 0, "char_end": 12}],
    }
    assert created == expected
    assert all("PRIVATE BODY" not in repr(params) for _, params in driver.calls)


def test_replace_references_stores_only_document_and_chunk_ids():
    driver = FakeDriver()
    links = [{"source_document_id": "a", "target_document_id": "b", "chunk_id": "c1"}]
    asyncio.run(Neo4jGraphStore(driver).replace_references("a", links))
    assert driver.transactions == 1
    assert driver.calls[-2][1] == {"document_id": "a"}
    assert driver.calls[-1][1] == {"links": links}
    assert "quote" not in repr(driver.calls[-1][1])


def test_schema_does_not_create_content_fulltext_index():
    assert all("FULLTEXT" not in statement for statement in neo4j_store.SCHEMA_STATEMENTS)
    assert all("text" not in statement for statement in neo4j_store.SCHEMA_STATEMENTS)


def test_close_closes_driver():
    driver = FakeDriver()
    asyncio.run(Neo4jGraphStore(driver).close())
    assert driver.closed


def test_fake_session_matches_real_driver_signature():
    for real in (AsyncSession.run, AsyncManagedTransaction.run):
        names = list(inspect.signature(real).parameters)
        assert names[:3] == list(inspect.signature(FakeSession.run).parameters)[:3]
