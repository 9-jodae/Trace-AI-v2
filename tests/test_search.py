import asyncio

import pytest
from conftest import chunk

from app.core.errors import AiUnavailableError
from app.retrieval import service


class FakeStore:
    def __init__(self, result=None, error=None):
        self.result = result or []
        self.error = error
        self.calls = []

    async def search(self, query_or_embedding, limit):
        self.calls.append((query_or_embedding, limit))
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def stores(monkeypatch):
    async def fake_embed(query):
        return [0.1, 0.2]

    def install(vector, graph):
        monkeypatch.setattr(service, "get_stores", lambda: (vector, graph))
        monkeypatch.setattr(service, "embed_query", fake_embed)

    return install


def test_queries_both_stores(stores):
    vector, graph = FakeStore([chunk("a")]), FakeStore([chunk("b")])
    stores(vector, graph)
    asyncio.run(service.search("질문"))
    assert vector.calls == [([0.1, 0.2], service.SOURCE_LIMIT)]
    assert graph.calls == [("질문", service.SOURCE_LIMIT)]


def test_stores_run_in_parallel(stores):
    class Waiting(FakeStore):
        event = None
        order = []

        async def search(self, value, limit):
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
