"""Neo4j는 이 환경에서 실행할 수 없어 전달하는 쿼리와 인자만 대역으로 확인한다."""

import asyncio
import inspect

from neo4j import AsyncManagedTransaction, AsyncSession

from app.retrieval import neo4j_store
from app.retrieval.neo4j_store import Neo4jGraphStore

DOCUMENT_SEED = {"kind": "document", "id": "doc", "names": ["출장비 규정", "travel-policy"]}
ENTITY_SEED = {"kind": "entity", "id": "ent", "names": ["hr"]}


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
        text = getattr(query, "text", query)
        if text == neo4j_store.SEEDS_CYPHER:
            return FakeResult(self.driver.seeds)
        return FakeResult(self.driver.rows)

    async def execute_write(self, work):
        self.driver.transactions += 1
        await work(self)


class FakeDriver:
    def __init__(self, seeds=(), rows=()):
        self.seeds = seeds
        self.rows = rows
        self.calls = []
        self.transactions = 0
        self.closed = False

    def session(self, database=None):
        self.database = database
        return FakeSession(self)

    async def close(self):
        self.closed = True


def search(driver, query="출장비 규정 알려줘", limit=3):
    return asyncio.run(Neo4jGraphStore(driver, "db").search(query, limit))


class TestSearch:
    def test_matched_document_starts_graph_search(self):
        driver = FakeDriver(seeds=[DOCUMENT_SEED], rows=[{"id": "a"}])
        assert search(driver) == [{"id": "a"}]
        query, params = driver.calls[-1]
        assert query.text == neo4j_store.GRAPH_SEARCH_CYPHER
        assert params == {"documents": ["doc"], "entities": [], "depth": 1, "limit": 3}
        assert query.timeout == neo4j_store.QUERY_TIMEOUT_SECONDS and driver.database == "db"

    def test_no_match_skips_graph_search(self):
        driver = FakeDriver(seeds=[DOCUMENT_SEED])
        assert search(driver, "관계없는 질문") == []
        assert len(driver.calls) == 1

    def test_entity_needs_word_boundary_for_english_names(self):
        driver = FakeDriver(seeds=[ENTITY_SEED])
        assert search(driver, "chrome 설치") == []
        search(driver, "HR 담당")
        assert driver.calls[-1][1]["entities"] == ["ent"]

    def test_korean_particle_is_allowed(self):
        driver = FakeDriver(seeds=[DOCUMENT_SEED])
        search(driver, "출장비 규정은 뭐야")
        assert driver.calls[-1][1]["documents"] == ["doc"]

    def test_relation_question_uses_two_hops(self):
        driver = FakeDriver(seeds=[ENTITY_SEED])
        search(driver, "hr 관련 문서")
        assert driver.calls[-1][1]["depth"] == 2

    def test_user_text_is_never_in_cypher(self):
        driver = FakeDriver(seeds=[DOCUMENT_SEED])
        search(driver, "출장비 규정 DETACH DELETE")
        assert all("DETACH" not in getattr(call[0], "text", call[0]) for call in driver.calls)

    def test_results_are_ids_not_text(self):
        assert "text" not in neo4j_store.GRAPH_SEARCH_CYPHER
        assert "RETURN id" in neo4j_store.GRAPH_SEARCH_CYPHER

    def test_null_names_do_not_crash(self):
        assert search(FakeDriver(seeds=[{"kind": "document", "id": "d", "names": None}])) == []


CHUNK = {"id": "a", "version_id": "ver", "char_start": 0, "char_end": 2, "text": "본문", "source": "규정"}
MENTION = {"chunk_id": "a", "entity_id": "e"}


class TestReplaceVersion:
    def replace(self, driver, mentions=()):
        asyncio.run(Neo4jGraphStore(driver).replace_version("doc", "규정", ["규정"], [CHUNK], list(mentions)))

    def written(self, driver):
        return driver.calls[len(neo4j_store.SCHEMA_STATEMENTS) :]

    def test_runs_in_one_transaction(self):
        driver = FakeDriver()
        self.replace(driver, [MENTION])
        statements = [call[0] for call in self.written(driver)]
        assert driver.transactions == 1
        assert statements == [
            neo4j_store.MERGE_DOCUMENT_CYPHER,
            neo4j_store.DELETE_DOCUMENT_CHUNKS_CYPHER,
            neo4j_store.CREATE_CHUNKS_CYPHER,
            neo4j_store.CREATE_MENTIONS_CYPHER,
        ]

    def test_chunk_text_is_not_stored(self):
        driver = FakeDriver()
        self.replace(driver)
        items = self.written(driver)[2][1]["chunks"]
        assert items == [{"id": "a", "version_id": "ver", "char_start": 0, "char_end": 2}]

    def test_no_mentions_skips_statement(self):
        driver = FakeDriver()
        self.replace(driver)
        assert neo4j_store.CREATE_MENTIONS_CYPHER not in [call[0] for call in driver.calls]

    def test_schema_is_created_first_without_permission_copy(self):
        driver = FakeDriver()
        self.replace(driver)
        assert [call[0] for call in driver.calls[:3]] == list(neo4j_store.SCHEMA_STATEMENTS)
        for statement in (neo4j_store.MERGE_DOCUMENT_CYPHER, neo4j_store.CREATE_CHUNKS_CYPHER):
            assert "allowed_group" not in statement and "status" not in statement


class TestReplaceReferences:
    def test_replaces_only_outgoing_links(self):
        driver = FakeDriver()
        links = [{"source_document_id": "a", "target_document_id": "b"}]
        asyncio.run(Neo4jGraphStore(driver).replace_references("a", links))
        assert driver.transactions == 1
        assert driver.calls[0] == (neo4j_store.DELETE_REFERENCES_CYPHER, {"document_id": "a"})
        assert driver.calls[1] == (neo4j_store.CREATE_REFERENCES_CYPHER, {"links": links})

    def test_empty_links_only_delete(self):
        driver = FakeDriver()
        asyncio.run(Neo4jGraphStore(driver).replace_references("a", []))
        assert [call[0] for call in driver.calls] == [neo4j_store.DELETE_REFERENCES_CYPHER]


def test_close_closes_driver():
    driver = FakeDriver()
    asyncio.run(Neo4jGraphStore(driver).close())
    assert driver.closed


def test_fake_session_matches_real_driver_signature():
    for real in (AsyncSession.run, AsyncManagedTransaction.run):
        names = list(inspect.signature(real).parameters)
        assert names[:3] == list(inspect.signature(FakeSession.run).parameters)[:3]
