import asyncio

import pytest
from conftest import chunk
from fastapi.testclient import TestClient

import main
from app.core import config
from app.core.errors import AiUnavailableError
from app.retrieval import service


class FakeStore:
    def __init__(self, result=None, error=None):
        self.result = result or []
        self.error = error
        self.calls = []

    async def search(self, query_or_embedding, limit, allowed_groups):
        self.calls.append((query_or_embedding, limit, allowed_groups))
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def stores(monkeypatch):
    async def fake_embed(query):
        return [0.1, 0.2]

    def install(vector, graph):
        monkeypatch.setenv("ALLOWED_GROUP_IDS", "all-staff, hr")
        monkeypatch.setattr(service, "get_stores", lambda: (vector, graph))
        monkeypatch.setattr(service, "embed_query", fake_embed)

    return install


def test_queries_both_stores(stores):
    vector, graph = FakeStore([chunk("a")]), FakeStore([chunk("b")])
    stores(vector, graph)
    asyncio.run(service.search("질문"))
    assert vector.calls == [([0.1, 0.2], service.SOURCE_LIMIT, ["all-staff", "hr"])]
    assert graph.calls == [("질문", service.SOURCE_LIMIT, ["all-staff", "hr"])]


def test_stores_run_in_parallel(stores):
    class Waiting(FakeStore):
        event = None
        order = []

        async def search(self, value, limit, allowed_groups):
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
    monkeypatch.setenv("ALLOWED_GROUP_IDS", "all-staff")
    monkeypatch.setattr(service, "embed_query", fail)
    monkeypatch.setattr(service, "get_stores", lambda: (FakeStore(), graph))
    assert asyncio.run(service.search("질문"))["failed_sources"] == ["vector"]


def test_both_failures_raise(stores):
    stores(FakeStore(error=RuntimeError("x")), FakeStore(error=RuntimeError("y")))
    with pytest.raises(AiUnavailableError):
        asyncio.run(service.search("질문"))


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
    with pytest.raises(AiUnavailableError):
        service.get_stores()


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

    def test_missing_neo4j_fails(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://db")
        with pytest.raises(AiUnavailableError):
            service.get_stores()

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


class TestPermissionFailsClosed:
    def test_no_groups_returns_nothing_and_skips_embedding(self, monkeypatch):
        async def forbidden(query):
            raise AssertionError("임베딩을 호출하면 안 됩니다")

        store = FakeStore([chunk("secret")])
        monkeypatch.setattr(service, "get_stores", lambda: (store, store))
        monkeypatch.setattr(service, "embed_query", forbidden)
        assert asyncio.run(service.search("질문")) == {"chunks": [], "failed_sources": []}
        assert store.calls == []

    def test_production_ignores_group_setting(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("ALLOWED_GROUP_IDS", "all-staff")
        assert config.allowed_group_ids() == []

    def test_local_group_setting_is_parsed(self, monkeypatch):
        monkeypatch.setenv("ALLOWED_GROUP_IDS", " a , ,b ")
        assert config.allowed_group_ids() == ["a", "b"]


def test_app_shutdown_closes_stores(monkeypatch):
    closed = []

    async def fake_close():
        closed.append(True)

    monkeypatch.setattr(main, "close_stores", fake_close)
    with TestClient(main.create_app()):
        assert closed == []
    assert closed == [True]

