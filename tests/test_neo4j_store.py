"""Neo4j는 이 환경에서 실행할 수 없어 전달하는 쿼리와 인자만 대역으로 확인한다."""

import asyncio
import inspect

import pytest
from neo4j import AsyncManagedTransaction, AsyncSession

from app.retrieval import neo4j_store
from app.retrieval.neo4j_store import Neo4jGraphStore, build_fulltext_query
from app.retrieval.schemas import ACTIVE_STATUS


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

    async def single(self):
        return self.rows[0] if self.rows else None


class FakeSession:
    def __init__(self, driver):
        self.driver = driver

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def run(self, query, parameters=None, **params):
        # 실제 드라이버와 같은 시그니처여야 키워드 충돌을 잡을 수 있다
        self.driver.calls.append((query, {**(parameters or {}), **params}))
        text = query.text if isinstance(query, neo4j_store.Query) else query
        if text == neo4j_store.MERGE_DOCUMENT_CYPHER:
            return FakeResult([self.driver.write_row] if self.driver.write_row is not None else [])
        if text == neo4j_store.CREATE_REFERENCES_CYPHER:
            return FakeResult([{"created": len(params["references"])}])
        if "SET d.index_status" in text:
            return FakeResult([{"document_id": "doc"}])
        return FakeResult(self.driver.rows)

    async def execute_write(self, work):
        self.driver.transactions += 1
        return await work(self)


class FakeDriver:
    def __init__(self, rows=()):
        self.rows = rows
        self.write_row = {"previous_version": None}
        self.calls = []
        self.transactions = 0
        self.closed = False

    def session(self, database=None):
        self.database = database
        return FakeSession(self)

    async def close(self):
        self.closed = True


class TestFulltextQuery:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("E4012 오류", '"E4012" OR "오류"'),
            ('a" OR b:*', '"a" OR "OR" OR "b"'),
            ("!!! ???", ""),
        ],
    )
    def test_builds_escaped_or_query(self, text, expected):
        assert build_fulltext_query(text) == expected

    def test_limits_terms(self):
        assert build_fulltext_query(" ".join(str(i) for i in range(50))).count(" OR ") == 11


class TestSearch:
    def test_passes_permission_and_status(self):
        driver = FakeDriver(rows=[{"id": "a", "text": "내용"}])
        result = asyncio.run(Neo4jGraphStore(driver, "db").search_documents("재택근무", 3, ["hr"]))
        assert result == [{"id": "a", "text": "내용"}]
        query, params = driver.calls[0]
        assert params == {"query": '"재택근무"', "status": ACTIVE_STATUS, "allowed_groups": ["hr"], "limit": 3}
        assert query.timeout == neo4j_store.QUERY_TIMEOUT_SECONDS
        assert "d.status = $status" in query.text and "$allowed_groups" in query.text
        assert driver.database == "db"

    def test_no_groups_makes_no_query(self):
        driver = FakeDriver()
        assert asyncio.run(Neo4jGraphStore(driver).search_documents("질문", 3, [])) == []
        assert driver.calls == []

    def test_query_without_terms_makes_no_query(self):
        driver = FakeDriver()
        assert asyncio.run(Neo4jGraphStore(driver).search_documents("!!!", 3, ["hr"])) == []
        assert driver.calls == []

    def test_user_text_is_never_in_cypher(self):
        driver = FakeDriver()
        asyncio.run(Neo4jGraphStore(driver).search_documents("DETACH DELETE", 3, ["hr"]))
        assert "DETACH" not in driver.calls[0][0].text


class TestReplaceVersion:
    def test_stores_only_document_and_cleans_legacy_nodes_in_one_transaction(self):
        driver = FakeDriver()
        document = {"document_id": "doc", "text": "내용", "status": "유효", "allowed_group_ids": ["hr"]}
        asyncio.run(Neo4jGraphStore(driver).replace_version("ver", "doc", document, []))
        assert driver.transactions == 1
        written = driver.calls[4:]
        assert written[0][0] == neo4j_store.MERGE_DOCUMENT_CYPHER
        assert written[1][0] == neo4j_store.DELETE_LEGACY_CHUNKS_CYPHER
        assert written[2][0] == neo4j_store.DELETE_REFERENCES_CYPHER
        assert written[0][1]["properties"]["text"] == "내용"
        assert all("CREATE (c:Chunk" not in str(query) for query, _ in driver.calls)

    def test_legacy_cleanup_is_scoped_to_document(self):
        assert "document_id: $document_id" in neo4j_store.DELETE_LEGACY_CHUNKS_CYPHER

    def test_schema_is_created_first(self):
        driver = FakeDriver()
        asyncio.run(Neo4jGraphStore(driver).replace_version("v", "d", {}, []))
        assert [call[0] for call in driver.calls[:4]] == list(neo4j_store.SCHEMA_STATEMENTS)
        assert "d.title, d.text" in neo4j_store.SCHEMA_STATEMENTS[2]

    def test_conflicting_version_stops_transaction(self):
        driver = FakeDriver()
        driver.write_row = None
        with pytest.raises(neo4j_store.DocumentVersionConflict):
            asyncio.run(Neo4jGraphStore(driver).replace_version("v", "d", {}, []))
        assert len(driver.calls) == 5

    def test_nested_provenance_is_serialized(self):
        props = neo4j_store.document_properties({"topics": [{"value": "주제"}], "field_sources": {"title": "metadata"}})
        assert props["topics"] == ["주제"]
        assert isinstance(props["field_sources_json"], str)
        assert "field_sources" not in props


def test_close_closes_driver():
    driver = FakeDriver()
    asyncio.run(Neo4jGraphStore(driver).close())
    assert driver.closed


def test_fake_session_matches_real_driver_signature():
    for real in (AsyncSession.run, AsyncManagedTransaction.run):
        names = list(inspect.signature(real).parameters)
        assert names[:3] == list(inspect.signature(FakeSession.run).parameters)[:3]



def test_current_chunk_filter_never_sends_document_text():
    driver = FakeDriver(rows=[{"id": "ok"}])
    chunks = [{"id": "ok", "document_id": "d", "version_id": "v", "content_hash": "h", "text": "비공개"},
              {"id": "old", "document_id": "d", "version_id": "old", "text": "이전"}]
    result = asyncio.run(Neo4jGraphStore(driver).filter_current_chunks(chunks, ["hr"]))
    assert [chunk["id"] for chunk in result] == ["ok"]
    params = driver.calls[0][1]
    assert all("text" not in chunk for chunk in params["chunks"])
    assert "d.content_hash = item.content_hash" in driver.calls[0][0].text


def test_reference_query_checks_both_versions():
    assert "r.source_version_id = startNode(r).version_id" in neo4j_store.FULLTEXT_SEARCH_CYPHER
    assert "r.target_version_id = endNode(r).version_id" in neo4j_store.FULLTEXT_SEARCH_CYPHER
    assert "target.index_status = 'READY'" in neo4j_store.CREATE_REFERENCES_CYPHER
