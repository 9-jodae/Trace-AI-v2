import asyncio

import pytest

from app.core.errors import AiUnavailableError
from app.retrieval import embedding


class FakeEmbedder:
    async def aembed_query(self, text):
        return [0.1, 0.2]

    async def aembed_documents(self, texts):
        return [[0.1, 0.2] for _ in texts]


@pytest.fixture
def init(monkeypatch):
    calls = []

    def factory(model, **kwargs):
        calls.append((model, kwargs))
        return FakeEmbedder()

    monkeypatch.setattr(embedding, "init_embeddings", factory)
    return calls


def test_passes_model_key_and_1536_dimensions(init):
    asyncio.run(embedding.embed_query("질문"))
    assert init == [("openai:fake-embedding", {"api_key": "fake-embedding-key", "dimensions": 1536})]


def test_embed_query(init):
    assert asyncio.run(embedding.embed_query("질문")) == [0.1, 0.2]


def test_embed_documents_keeps_count(init):
    assert len(asyncio.run(embedding.embed_documents(["a", "b", "c"]))) == 3


@pytest.mark.parametrize("name", ["EMBEDDING_API_KEY", "EMBEDDING_MODEL"])
def test_missing_setting_raises_without_creating_client(monkeypatch, init, name):
    monkeypatch.setenv(name, "")
    with pytest.raises(AiUnavailableError):
        asyncio.run(embedding.embed_query("질문"))
    assert init == []


def test_hanging_call_times_out(monkeypatch):
    class Hanging:
        async def aembed_query(self, text):
            await asyncio.sleep(10)

    monkeypatch.setattr(embedding, "init_embeddings", lambda model, **kwargs: Hanging())
    monkeypatch.setattr(embedding, "TIMEOUT_SECONDS", 0.01)
    with pytest.raises(TimeoutError):
        asyncio.run(embedding.embed_query("질문"))
