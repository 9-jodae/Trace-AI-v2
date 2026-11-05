import asyncio

import pytest
from conftest import chunk
from fastapi.testclient import TestClient

import main
from app.core import config
from app.core.errors import AiUnavailableError
from app.core.schemas import AccessScope, SearchFilters
from app.retrieval import service

SCOPE = AccessScope("dept", "STAFF", "INTERNAL", False)
SCOPE_FROM_ENV = AccessScope(None, None, "INTERNAL", False)


class FakeStore:
    def __init__(self, result=None, error=None, accessible=None, filter_error=None, fulltext=None, fulltext_error=None):
        self.result = result or []
        self.error = error
        self.accessible = accessible
        self.filter_error = filter_error
        self.fulltext_result = fulltext or []
        self.fulltext_error = fulltext_error
        self.calls = []
        self.filter_calls = []
        self.document_calls = []
        self.expansion_calls = []
        self.fulltext_calls = []

    async def search(self, query_or_embedding, limit, *scope):
        self.calls.append((query_or_embedding, limit, *scope))
        if self.error:
            raise self.error
        if isinstance(query_or_embedding, str):
            return [item["id"] if isinstance(item, dict) else item for item in self.result]
        return self.result

    async def fulltext(self, query, limit, scope, filters=None):
        self.fulltext_calls.append((query, limit, scope, filters))
        if self.fulltext_error:
            raise self.fulltext_error
        return self.fulltext_result

    async def seed_documents(self, query):
        self.calls.append((query, service.GRAPH_CANDIDATE_LIMIT))
        if self.error:
            raise self.error
        return ["doc"] if self.result else []

    async def search_from_entities(self, query, limit):
        return getattr(self, "entity_chunks", [])

    async def follow_references(self, document_ids):
        self.followed = getattr(self, "followed", []) + [list(document_ids)]
        links = getattr(self, "links", {})
        return sorted({target for document_id in document_ids for target in links.get(document_id, [])})

    async def document_chunks(self, document_ids):
        return [f"{document_id}-chunk" for document_id in document_ids]

    async def search_from_documents(self, document_ids, limit):
        self.expansion_calls.append((document_ids, limit))
        return [item["id"] if isinstance(item, dict) else item for item in self.result[:limit]]

    async def list_document_catalog(self):
        return []

    async def accessible_document_ids(self, document_ids, scope, filters=None):
        self.document_calls.append((document_ids, scope, filters))
        if scope is None or self.accessible == set():
            return set()
        return set(document_ids)

    async def filter_accessible(self, chunk_ids, scope, filters=None):
        self.filter_calls.append((chunk_ids, scope, filters))
        if self.filter_error:
            raise self.filter_error
        return set(chunk_ids) if self.accessible is None else self.accessible

    async def fetch_accessible(self, chunk_ids, scope, filters=None, embedding=None):
        self.filter_calls.append((chunk_ids, scope, filters))
        if self.filter_error:
            raise self.filter_error
        if scope is None:
            return []
        allowed = set(chunk_ids) if self.accessible is None else self.accessible
        rows = {item["id"]: item for item in [*self.result, *self.fulltext_result] if isinstance(item, dict)}
        return [rows.get(chunk_id, chunk(chunk_id)) for chunk_id in chunk_ids if chunk_id in allowed]


@pytest.fixture
def stores(monkeypatch):
    async def fake_embed(query):
        return [0.1, 0.2]

    def install(vector, graph):
        monkeypatch.setattr(service.config, "access_scope", lambda: SCOPE)
        monkeypatch.setattr(service, "get_stores", lambda: (vector, graph))
        monkeypatch.setattr(service, "embed_query", fake_embed)

    return install


def test_queries_both_stores(stores):
    vector, graph = FakeStore([chunk("a")]), FakeStore([chunk("b")])
    stores(vector, graph)
    asyncio.run(service.search("질문"))
    assert vector.calls == [([0.1, 0.2], service.SOURCE_LIMIT, SCOPE, None)]
    assert graph.calls == [("질문", service.GRAPH_CANDIDATE_LIMIT)]


def test_stores_run_in_parallel(stores):
    class Waiting(FakeStore):
        event = None
        order = []

        async def seed_documents(self, value):
            Waiting.order.append("start")
            await asyncio.sleep(0)
            Waiting.order.append("end")
            return []

        async def search(self, query_or_embedding, limit, *scope):
            Waiting.order.append("start")
            await asyncio.sleep(0)
            Waiting.order.append("end")
            return []

    stores(Waiting(), Waiting())
    asyncio.run(service.search("질문"))
    assert Waiting.order[:2] == ["start", "start"]


def test_results_are_fused(stores):
    stores(FakeStore([chunk("a"), chunk("b")]), FakeStore([chunk("b"), chunk("c")]))
    result = asyncio.run(service.search("질문"))
    assert [c["id"] for c in result["chunks"]] == ["b", "a", "c"]
    assert result["failed_sources"] == []


def test_context_limit(stores):
    stores(FakeStore([chunk(str(i)) for i in range(10)]), FakeStore())
    assert len(asyncio.run(service.search("질문"))["chunks"]) == service.CONTEXT_LIMIT


def test_vector_failure_keeps_graph(stores):
    stores(FakeStore(error=RuntimeError("x")), FakeStore([chunk("b")]))
    result = asyncio.run(service.search("질문"))
    assert result == {"chunks": [chunk("b")], "failed_sources": ["vector"]}


def test_graph_failure_keeps_vector(stores):
    stores(FakeStore([chunk("a")]), FakeStore(error=RuntimeError("x")))
    result = asyncio.run(service.search("질문"))
    assert result == {"chunks": [chunk("a")], "failed_sources": ["graph"]}


def test_embedding_failure_counts_as_vector_failure(monkeypatch):
    async def fail(query):
        raise RuntimeError("embedding")

    graph = FakeStore([chunk("b")])
    monkeypatch.setenv("ACCESS_CLEARANCE", "INTERNAL")
    monkeypatch.setattr(service, "embed_query", fail)
    monkeypatch.setattr(service, "get_stores", lambda: (FakeStore(), graph))
    assert asyncio.run(service.search("질문"))["failed_sources"] == ["vector"]


def test_both_failures_raise(stores):
    stores(FakeStore(error=RuntimeError("x"), fulltext_error=RuntimeError("z")), FakeStore(error=RuntimeError("y")))
    with pytest.raises(AiUnavailableError):
        asyncio.run(service.search("질문"))


def test_postgres_failure_with_empty_graph_is_not_no_result(stores):
    # 운영에서 ai_app 권한이 사라졌을 때 "찾지 못했습니다"로 보인 경우다
    stores(FakeStore(error=RuntimeError("x"), fulltext_error=RuntimeError("z")), FakeStore())
    with pytest.raises(AiUnavailableError):
        asyncio.run(service.search("질문"))


def test_postgres_failure_keeps_graph_chunks(stores):
    stores(FakeStore(error=RuntimeError("x"), fulltext_error=RuntimeError("z")), FakeStore([chunk("b")]))
    result = asyncio.run(service.search("질문"))
    assert result["chunks"] and set(result["failed_sources"]) >= {"vector", "fulltext"}


def test_empty_results_are_not_failures(stores):
    stores(FakeStore(), FakeStore())
    assert asyncio.run(service.search("질문")) == {"chunks": [], "failed_sources": []}


def test_sample_stores_require_explicit_setting(monkeypatch):
    monkeypatch.setenv("USE_SAMPLE_DATA", "")
    with pytest.raises(AiUnavailableError):
        service.get_stores()
    monkeypatch.setenv("USE_SAMPLE_DATA", "true")
    assert type(service.get_stores()[0]).__name__ == "SampleVectorStore"


def test_database_url_alone_does_not_enable_samples(monkeypatch):
    monkeypatch.setenv("USE_SAMPLE_DATA", "")
    monkeypatch.setenv("DATABASE_URL", "postgresql://db")
    assert type(service.get_stores()[0]).__name__ == "PostgresVectorStore"
    service._real_stores.cache_clear()


def test_missing_stores_fail_before_paying_for_embedding(monkeypatch):
    async def forbidden(query):
        raise AssertionError("임베딩을 호출하면 안 됩니다")

    monkeypatch.setenv("USE_SAMPLE_DATA", "")
    monkeypatch.setattr(service, "embed_query", forbidden)
    with pytest.raises(AiUnavailableError):
        asyncio.run(service.search("질문"))


class TestRealStoreSelection:
    @pytest.fixture(autouse=True)
    def clean_cache(self):
        service._real_stores.cache_clear()
        yield
        service._real_stores.cache_clear()

    @pytest.fixture
    def neo4j(self, monkeypatch):
        monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
        monkeypatch.setenv("NEO4J_PASSWORD", "fake-password")

    def test_real_stores_selected(self, monkeypatch, neo4j):
        monkeypatch.setenv("DATABASE_URL", "postgresql://db")
        vector, graph = service.get_stores()
        assert type(vector).__name__ == "PostgresVectorStore" and type(graph).__name__ == "Neo4jGraphStore"

    def test_stores_are_reused(self, monkeypatch, neo4j):
        monkeypatch.setenv("DATABASE_URL", "postgresql://db")
        assert service.get_stores()[0] is service.get_stores()[0]

    def test_missing_database_url_fails(self, neo4j):
        with pytest.raises(AiUnavailableError):
            service.get_stores()

    def test_dimensions_must_match_chunks_table(self, monkeypatch, neo4j):
        monkeypatch.setenv("DATABASE_URL", "postgresql://db")
        monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1536")
        with pytest.raises(AiUnavailableError):
            service.get_stores()

    def test_missing_neo4j_uses_vector_only(self, monkeypatch, caplog):
        monkeypatch.setenv("DATABASE_URL", "postgresql://db")
        vector, graph = service.get_stores()
        assert type(vector).__name__ == "PostgresVectorStore" and type(graph).__name__ == "NoGraphStore"
        assert asyncio.run(graph.search("질문", 3)) == []
        assert "Graph search disabled" in caplog.text

    def test_neo4j_uri_without_password_fails(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://db")
        monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
        with pytest.raises(AiUnavailableError):
            service.get_stores()

    def test_vector_only_search_reports_no_failure(self, monkeypatch):
        async def embed(query):
            return [0.1]

        monkeypatch.setenv("ACCESS_CLEARANCE", "INTERNAL")
        vector = FakeStore([chunk("a")])
        monkeypatch.setattr(service, "get_stores", lambda: (vector, service.NoGraphStore()))
        monkeypatch.setattr(service, "embed_query", embed)
        result = asyncio.run(service.search("질문"))
        assert result == {"chunks": [chunk("a")], "failed_sources": []}
        assert vector.document_calls == [] and vector.filter_calls == []

    def test_close_stores_closes_both(self, monkeypatch, neo4j):
        closed = []

        class Closable:
            def __init__(self, name):
                self.name = name

            async def close(self):
                closed.append(self.name)

        monkeypatch.setenv("DATABASE_URL", "postgresql://db")
        monkeypatch.setattr(service, "PostgresVectorStore", lambda url: Closable("vector"))
        monkeypatch.setattr(service.Neo4jGraphStore, "from_uri", lambda *args: Closable("graph"))
        service.get_stores()
        asyncio.run(service.close_stores())
        assert sorted(closed) == ["graph", "vector"]

    def test_close_without_stores_is_safe(self):
        asyncio.run(service.close_stores())


class TestFulltextFallback:
    def test_runs_when_graph_has_few_results(self, stores):
        vector = FakeStore([chunk("a")], fulltext=[chunk("t")])
        stores(vector, FakeStore([chunk("b")]))
        filters = SearchFilters(extensions=["pdf"])
        result = asyncio.run(service.search("질문", filters))
        assert vector.fulltext_calls == [("질문", service.SOURCE_LIMIT, SCOPE, filters)]
        assert {c["id"] for c in result["chunks"]} == {"a", "b", "t"}

    def test_skipped_when_graph_has_enough_results(self, stores):
        vector = FakeStore(fulltext=[chunk("t")])
        stores(vector, FakeStore([chunk(str(i)) for i in range(service.FULLTEXT_FALLBACK_BELOW)]))
        asyncio.run(service.search("질문"))
        assert vector.fulltext_calls == []

    def test_results_in_both_rank_higher(self, stores):
        stores(FakeStore([chunk("a"), chunk("b")], fulltext=[chunk("b"), chunk("c")]), FakeStore())
        assert asyncio.run(service.search("질문"))["chunks"][0]["id"] == "b"

    def test_failure_is_reported_and_keeps_others(self, stores):
        stores(FakeStore([chunk("a")], fulltext_error=RuntimeError("x")), FakeStore())
        assert asyncio.run(service.search("질문")) == {"chunks": [chunk("a")], "failed_sources": ["fulltext"]}

    def test_keeps_answering_when_vector_and_graph_fail(self, stores):
        vector = FakeStore(error=RuntimeError("x"), fulltext=[chunk("t")])
        stores(vector, FakeStore(error=RuntimeError("y")))
        assert asyncio.run(service.search("질문")) == {"chunks": [chunk("t")], "failed_sources": ["vector", "graph"]}

    def test_no_scope_skips_fulltext(self, monkeypatch):
        store = FakeStore(fulltext=[chunk("secret")])
        monkeypatch.setattr(service, "get_stores", lambda: (store, store))
        assert asyncio.run(service.search("질문"))["chunks"] == []
        assert store.fulltext_calls == []


class TestGraphPermissionFilter:
    def test_inaccessible_graph_results_are_removed(self, stores):
        vector = FakeStore(accessible={"b"})
        graph = FakeStore([chunk("secret"), chunk("b")])
        stores(vector, graph)
        result = asyncio.run(service.search("질문"))
        assert [c["id"] for c in result["chunks"]] == ["b"]
        assert vector.filter_calls == [(["secret", "b"], SCOPE, None)]

    def test_filters_reach_vector_and_graph_check(self, stores):
        vector = FakeStore(accessible={"b"})
        stores(vector, FakeStore([chunk("x"), chunk("b")]))
        filters = SearchFilters(extensions=["pdf"])
        asyncio.run(service.search("질문", filters))
        assert vector.calls[0][3] is filters
        assert vector.filter_calls == [(["x", "b"], SCOPE, filters)]

    def test_filter_failure_drops_graph_results(self, stores):
        stores(FakeStore([chunk("a")], filter_error=RuntimeError("db")), FakeStore([chunk("secret")]))
        result = asyncio.run(service.search("질문"))
        assert result == {"chunks": [chunk("a")], "failed_sources": ["graph"]}

    def test_inaccessible_seed_is_rejected_before_graph_expansion(self, stores):
        vector, graph = FakeStore(accessible=set()), FakeStore([chunk("secret")])
        stores(vector, graph)
        result = asyncio.run(service.search("질문"))
        assert graph.expansion_calls == []
        assert result["chunks"] == []
        assert vector.document_calls == [(["doc"], SCOPE, None)]

    def test_postgres_down_exposes_nothing(self, stores):
        down = RuntimeError("db")
        stores(FakeStore(error=down, filter_error=down, fulltext_error=down), FakeStore([chunk("secret")]))
        with pytest.raises(AiUnavailableError):
            asyncio.run(service.search("질문"))

    def test_filtered_results_keep_limit(self, stores):
        stores(FakeStore(), FakeStore([chunk(str(i)) for i in range(service.GRAPH_CANDIDATE_LIMIT)]))
        assert len(asyncio.run(service.search("질문"))["chunks"]) == service.CONTEXT_LIMIT


class TestFilterAccessibleContext:
    def test_keeps_order_of_accessible_chunks(self, monkeypatch):
        vector = FakeStore(accessible={"c", "a"})
        monkeypatch.setattr(service, "get_stores", lambda: (vector, FakeStore()))
        monkeypatch.setattr(service.config, "access_scope", lambda: SCOPE)
        chunks = [chunk("a"), chunk("b"), chunk("c")]
        assert [c["id"] for c in asyncio.run(service.filter_accessible_context(chunks))] == ["a", "c"]
        assert vector.filter_calls == [(["a", "b", "c"], SCOPE, None)]


class TestRequestScope:
    def test_request_scope_reaches_vector_and_graph_check(self, stores):
        vector, graph = FakeStore(), FakeStore([chunk("b")])
        stores(vector, graph)
        scope = AccessScope("요청", "STAFF", "PUBLIC", False)
        asyncio.run(service.search("질문", None, scope))
        assert vector.calls[0][2] == scope
        assert vector.filter_calls == [(["b"], scope, None)]

    def test_production_uses_request_scope(self, monkeypatch):
        async def embed(query):
            return [0.1]

        monkeypatch.setenv("APP_ENV", "production")
        vector = FakeStore([chunk("a")])
        monkeypatch.setattr(service, "get_stores", lambda: (vector, FakeStore()))
        monkeypatch.setattr(service, "embed_query", embed)
        scope = AccessScope("요청", "STAFF", "PUBLIC", False)
        assert [c["id"] for c in asyncio.run(service.search("질문", None, scope))["chunks"]] == ["a"]
        assert vector.calls[0][2] == scope

    def test_production_without_request_scope_exposes_nothing(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "production")
        vector = FakeStore([chunk("secret")])
        monkeypatch.setattr(service, "get_stores", lambda: (vector, vector))
        assert asyncio.run(service.search("질문")) == {"chunks": [], "failed_sources": []}
        assert vector.calls == []

    def test_context_revalidation_uses_request_scope(self, monkeypatch):
        vector = FakeStore(accessible={"a"})
        monkeypatch.setattr(service, "get_stores", lambda: (vector, FakeStore()))
        scope = AccessScope("요청", "STAFF", "PUBLIC", False)
        asyncio.run(service.filter_accessible_context([chunk("a")], scope))
        assert vector.filter_calls == [(["a"], scope, None)]

    def test_production_revalidation_without_scope_keeps_nothing(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "production")
        vector = FakeStore(accessible=set())
        monkeypatch.setattr(service, "get_stores", lambda: (vector, FakeStore()))
        assert asyncio.run(service.filter_accessible_context([chunk("a")])) == []
        assert vector.filter_calls[0][1] is None


class TestPermissionFailsClosed:
    def test_no_scope_returns_nothing_and_skips_embedding(self, monkeypatch):
        async def forbidden(query):
            raise AssertionError("임베딩을 호출하면 안 됩니다")

        store = FakeStore([chunk("secret")])
        monkeypatch.setattr(service, "get_stores", lambda: (store, store))
        monkeypatch.setattr(service, "embed_query", forbidden)
        assert asyncio.run(service.search("질문")) == {"chunks": [], "failed_sources": []}
        assert store.calls == []

    def test_production_ignores_scope_setting(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("ACCESS_CLEARANCE", "CONFIDENTIAL")
        monkeypatch.setenv("ACCESS_IS_ADMIN", "true")
        assert config.access_scope() is None

    def test_local_scope_is_parsed(self, monkeypatch):
        monkeypatch.setenv("ACCESS_DEPARTMENT_ID", "dept")
        monkeypatch.setenv("ACCESS_POSITION", "STAFF")
        monkeypatch.setenv("ACCESS_CLEARANCE", "INTERNAL")
        assert config.access_scope() == AccessScope("dept", "STAFF", "INTERNAL", False)

    def test_local_admin_needs_no_clearance(self, monkeypatch):
        monkeypatch.setenv("ACCESS_IS_ADMIN", "true")
        assert config.access_scope() == AccessScope(None, None, None, True)

    def test_local_scope_without_clearance_is_none(self, monkeypatch):
        monkeypatch.setenv("ACCESS_DEPARTMENT_ID", "dept")
        assert config.access_scope() is None


def test_app_shutdown_closes_stores(monkeypatch):
    closed = []

    async def fake_close():
        closed.append(True)

    monkeypatch.setattr(main, "close_stores", fake_close)
    with TestClient(main.create_app()):
        assert closed == []
    assert closed == [True]


class TestEntityChunks:
    def test_entity_chunks_are_fetched_without_document_seed(self, stores):
        vector = FakeStore([])
        graph = FakeStore([])
        graph.entity_chunks = ["chunk-a"]
        vector.result = []
        stores(vector, graph)
        asyncio.run(service.search("재무팀 담당", None, SCOPE))
        assert vector.filter_calls[-1][0] == ["chunk-a"] and vector.document_calls == []


class TestGraphSearchSwitch:
    @pytest.fixture
    def neo4j(self, stores, monkeypatch):
        vector = FakeStore([])
        graph = FakeStore([chunk("g")])
        stores(vector, graph)
        monkeypatch.setattr(service, "Neo4jGraphStore", FakeStore)
        return graph

    def test_neo4j_is_searched_by_default(self, neo4j):
        asyncio.run(service.search("질문", None, SCOPE))
        assert len(neo4j.calls) == 1

    def test_neo4j_can_be_turned_off(self, neo4j, monkeypatch):
        monkeypatch.setenv("GRAPH_SEARCH_ENABLED", "false")
        asyncio.run(service.search("질문", None, SCOPE))
        assert neo4j.calls == []


class TestChainSearch:
    @pytest.fixture
    def chain(self, stores, monkeypatch):
        vector = FakeStore([{**chunk(f"v{i}"), "document_id": f"d{i}"} for i in range(5)])
        graph = FakeStore([])
        vector.fetch_chain = {}

        async def nearest(chunk_ids, scope, filters, embedding, limit):
            vector.nearest_calls.append((chunk_ids, limit))
            nearest_first = sorted(chunk_ids, reverse=True)[:limit]
            return [{**chunk(chunk_id), "document_id": chunk_id.removesuffix("-chunk")} for chunk_id in nearest_first]

        vector.nearest_calls = []
        vector.nearest_accessible = nearest
        monkeypatch.setattr(service, "Neo4jGraphStore", FakeStore)
        stores(vector, graph)
        return vector, graph

    def ids(self, query="질문"):
        return [item["id"] for item in asyncio.run(service.search(query, None, SCOPE))["chunks"]]

    def test_follows_two_hops_and_keeps_context_limit(self, chain):
        _, graph = chain
        graph.links = {"d0": ["x1"], "x1": ["x2"], "x2": ["x3"]}
        result = self.ids()
        assert result == [*[f"v{i}" for i in range(5)], "x2-chunk", "x1-chunk"]

    def test_keeps_only_nearest_chunks(self, chain):
        vector, graph = chain
        graph.links = {"d0": ["x1", "x2", "x3"]}
        result = self.ids()
        assert result[-2:] == ["x3-chunk", "x2-chunk"]
        assert vector.nearest_calls == [(["x1-chunk", "x2-chunk", "x3-chunk"], service.CHAIN_CONTEXT_LIMIT)]

    def test_does_not_pass_through_inaccessible_document(self, chain):
        vector, graph = chain
        graph.links = {"d0": ["secret"], "secret": ["after"]}
        vector.accessible = {"d0", "d1", "d2"}

        async def allowed(document_ids, scope, filters=None):
            return {document_id for document_id in document_ids if document_id != "secret"}

        vector.accessible_document_ids = allowed
        result = self.ids()
        assert "secret-chunk" not in result and "after-chunk" not in result
        assert len(graph.followed) == 1

    def test_seeds_are_limited(self, chain):
        _, graph = chain
        self.ids()
        assert graph.followed[0] == ["d0", "d1", "d2"]

    def test_no_references_keeps_results(self, chain):
        assert self.ids() == [f"v{i}" for i in range(5)]

    def test_failure_is_reported_and_keeps_results(self, chain):
        _, graph = chain

        async def fail(document_ids):
            raise RuntimeError("bolt")

        graph.follow_references = fail
        result = asyncio.run(service.search("질문", None, SCOPE))
        assert [item["id"] for item in result["chunks"]] == [f"v{i}" for i in range(5)]
        assert result["failed_sources"] == ["chain"]

    def test_embedding_failure_still_follows_references(self, chain, monkeypatch):
        vector, graph = chain
        graph.links = {"d0": ["x1"]}

        async def fail(query):
            raise RuntimeError("embedding down")

        async def fetch(chunk_ids, scope, filters=None, embedding=None):
            return [{**chunk(chunk_id), "document_id": chunk_id.removesuffix("-chunk")} for chunk_id in chunk_ids]

        async def fulltext(query, limit, scope, filters=None):
            return [{**chunk("v0"), "document_id": "d0"}]

        monkeypatch.setattr(service, "embed_query", fail)
        vector.fetch_accessible = fetch
        vector.fulltext = fulltext
        result = asyncio.run(service.search("질문", None, SCOPE))
        assert [item["id"] for item in result["chunks"]] == ["v0", "x1-chunk"]
        assert result["failed_sources"] == ["vector"] and vector.nearest_calls == []

    def test_disabled_does_not_follow(self, chain, monkeypatch):
        _, graph = chain
        monkeypatch.setenv("GRAPH_SEARCH_ENABLED", "false")
        self.ids()
        assert not hasattr(graph, "followed")

