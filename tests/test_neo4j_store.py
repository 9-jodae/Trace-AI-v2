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
        return FakeResult(self.driver.rows)

    async def execute_write(self, work):
        self.driver.transactions += 1
        await work(self)


class FakeDriver:
    def __init__(self, rows=()):
        self.rows = rows
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
        result = asyncio.run(Neo4jGraphStore(driver, "db").search("재택근무", 3, ["hr"]))
        assert result == [{"id": "a", "text": "내용"}]
        query, params = driver.calls[0]
        assert params == {"query": '"재택근무"', "status": ACTIVE_STATUS, "allowed_groups": ["hr"], "limit": 3}
        assert query.timeout == neo4j_store.QUERY_TIMEOUT_SECONDS
        assert "d.status = $status" in query.text and "$allowed_groups" in query.text
        assert driver.database == "db"

    def test_no_groups_makes_no_query(self):
        driver = FakeDriver()
        assert asyncio.run(Neo4jGraphStore(driver).search("질문", 3, [])) == []
        assert driver.calls == []

    def test_query_without_terms_makes_no_query(self):
        driver = FakeDriver()
        assert asyncio.run(Neo4jGraphStore(driver).search("!!!", 3, ["hr"])) == []
        assert driver.calls == []

    def test_user_text_is_never_in_cypher(self):
        driver = FakeDriver()
        asyncio.run(Neo4jGraphStore(driver).search("DETACH DELETE", 3, ["hr"]))
        assert "DETACH" not in driver.calls[0][0].text


class TestReplaceVersion:
    def test_runs_delete_merge_create_in_one_transaction(self):
        driver = FakeDriver()
        chunks = [{"id": "a", "text": "내용"}]
        document = {"document_id": "doc", "status": "유효"}
        asyncio.run(Neo4jGraphStore(driver).replace_version("ver", "doc", document, chunks))
        assert driver.transactions == 1
        queries = [call[0] for call in driver.calls if isinstance(call[0], str)]
        written = queries[-3:]
        assert "DETACH DELETE" in written[0] and "MERGE (d:Document" in written[1] and "UNWIND $chunks" in written[2]
        assert driver.calls[-3][1] == {"version_id": "ver"}
        assert driver.calls[-1][1] == {"document_id": "doc", "chunks": chunks}

    def test_delete_is_scoped_to_version(self):
        assert "{version_id: $version_id}" in neo4j_store.DELETE_VERSION_CYPHER

    def test_schema_is_created_first(self):
        driver = FakeDriver()
        asyncio.run(Neo4jGraphStore(driver).replace_version("v", "d", {}, []))
        assert driver.calls[0][0] == neo4j_store.SCHEMA_STATEMENTS[0]


def test_close_closes_driver():
    driver = FakeDriver()
    asyncio.run(Neo4jGraphStore(driver).close())
    assert driver.closed


def test_fake_session_matches_real_driver_signature():
    for real in (AsyncSession.run, AsyncManagedTransaction.run):
        names = list(inspect.signature(real).parameters)
        assert names[:3] == list(inspect.signature(FakeSession.run).parameters)[:3]

