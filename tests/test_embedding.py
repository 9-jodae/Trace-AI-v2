import asyncio

import pytest

from app.core.errors import AiUnavailableError
from app.retrieval import embedding


class FakeEmbedder:
    size = 1024

    async def aembed_query(self, text):
        return [0.1] * self.size

    async def aembed_documents(self, texts):
        return [[0.1] * self.size for _ in texts]


@pytest.fixture
def init(monkeypatch):
    calls = []

    def factory(model, **kwargs):
        calls.append((model, kwargs))
        return FakeEmbedder()

    monkeypatch.setattr(embedding, "init_embeddings", factory)
    return calls


def test_passes_model_key_and_dimensions(init):
    asyncio.run(embedding.embed_query("질문"))
    assert init == [("openai:fake-embedding", {"api_key": "fake-embedding-key", "dimensions": 1024})]


def test_embed_query(init):
    assert asyncio.run(embedding.embed_query("질문")) == [0.1] * 1024


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


def test_bedrock_uses_aws_credentials_and_model_kwargs(monkeypatch, init):
    monkeypatch.setenv("EMBEDDING_MODEL", "bedrock:amazon.titan-embed-text-v2:0")
    monkeypatch.setenv("EMBEDDING_API_KEY", "")
    asyncio.run(embedding.embed_query("질문"))
    assert init == [("bedrock:amazon.titan-embed-text-v2:0", {"model_kwargs": {"dimensions": 1024}})]


def test_query_with_wrong_dimensions_is_rejected(init, monkeypatch):
    monkeypatch.setattr(FakeEmbedder, "size", 2)
    with pytest.raises(AiUnavailableError, match="1024"):
        asyncio.run(embedding.embed_query("질문"))


def test_documents_with_wrong_dimensions_are_rejected(init, monkeypatch):
    monkeypatch.setattr(FakeEmbedder, "size", 2)
    with pytest.raises(AiUnavailableError, match="1024"):
        asyncio.run(embedding.embed_documents(["a", "b"]))


def test_one_wrong_vector_rejects_the_batch(monkeypatch):
    class Mixed:
        async def aembed_documents(self, texts):
            return [[0.1] * 1024, [0.1]]

    monkeypatch.setattr(embedding, "init_embeddings", lambda model, **kwargs: Mixed())
    with pytest.raises(AiUnavailableError):
        asyncio.run(embedding.embed_documents(["a", "b"]))
