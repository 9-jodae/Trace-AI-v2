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


class TestDocumentBatches:
    @pytest.fixture
    def embedder(self, monkeypatch):
        class Recording:
            def __init__(self):
                self.batches = []
                self.failures = []

            async def aembed_documents(self, texts):
                self.batches.append(list(texts))
                if self.failures:
                    failure = self.failures.pop(0)
                    if failure == "hang":
                        await asyncio.sleep(10)
                    raise failure
                return [[float(text)] + [0.0] * 1023 for text in texts]

        recording = Recording()
        monkeypatch.setattr(embedding, "init_embeddings", lambda model, **kwargs: recording)
        return recording

    @staticmethod
    def texts(count):
        return [str(index) for index in range(count)]

    def test_splits_into_batches_and_keeps_order(self, embedder):
        count = embedding.BATCH_SIZE * 2 + 3
        vectors = asyncio.run(embedding.embed_documents(self.texts(count)))
        assert [len(batch) for batch in embedder.batches] == [embedding.BATCH_SIZE, embedding.BATCH_SIZE, 3]
        assert [vector[0] for vector in vectors] == [float(index) for index in range(count)]

    def test_empty_input_makes_no_call(self, embedder):
        assert asyncio.run(embedding.embed_documents([])) == []
        assert embedder.batches == []

    def test_failed_batch_is_retried_without_repeating_others(self, embedder):
        embedder.failures = [RuntimeError("일시 오류")]
        vectors = asyncio.run(embedding.embed_documents(self.texts(embedding.BATCH_SIZE + 1)))
        assert len(vectors) == embedding.BATCH_SIZE + 1
        assert [len(batch) for batch in embedder.batches] == [embedding.BATCH_SIZE, embedding.BATCH_SIZE, 1]

    def test_repeated_failure_raises(self, embedder):
        embedder.failures = [RuntimeError("a"), RuntimeError("b")]
        with pytest.raises(RuntimeError, match="b"):
            asyncio.run(embedding.embed_documents(self.texts(3)))
        assert len(embedder.batches) == embedding.BATCH_ATTEMPTS

    def test_timeout_applies_per_batch_and_is_retried(self, embedder, monkeypatch):
        monkeypatch.setattr(embedding, "TIMEOUT_SECONDS", 0.01)
        embedder.failures = ["hang"]
        assert len(asyncio.run(embedding.embed_documents(self.texts(2)))) == 2
        assert len(embedder.batches) == 2

    def test_wrong_dimensions_are_not_retried(self, monkeypatch):
        calls = []

        class Wrong:
            async def aembed_documents(self, texts):
                calls.append(texts)
                return [[0.1] for _ in texts]

        monkeypatch.setattr(embedding, "init_embeddings", lambda model, **kwargs: Wrong())
        with pytest.raises(AiUnavailableError):
            asyncio.run(embedding.embed_documents(["a"]))
        assert len(calls) == 1


class TestClientReuse:
    def test_embedder_is_created_once(self, init):
        async def scenario():
            await embedding.embed_query("a")
            await embedding.embed_documents(["b"])

        asyncio.run(scenario())
        assert len(init) == 1

    def test_changed_settings_create_new_embedder(self, init, monkeypatch):
        asyncio.run(embedding.embed_query("a"))
        monkeypatch.setenv("EMBEDDING_API_KEY", "other-key")
        asyncio.run(embedding.embed_query("b"))
        assert len(init) == 2


class TestQueryCache:
    @pytest.fixture
    def counting(self, monkeypatch):
        calls = []

        class Counting:
            async def aembed_query(self, text):
                calls.append(text)
                return [float(len(calls))] + [0.0] * 1023

        monkeypatch.setattr(embedding, "init_embeddings", lambda model, **kwargs: Counting())
        return calls

    def test_same_question_calls_api_once(self, counting):
        first = asyncio.run(embedding.embed_query("질문"))
        second = asyncio.run(embedding.embed_query("질문"))
        assert counting == ["질문"] and first == second

    def test_different_questions_are_separate(self, counting):
        asyncio.run(embedding.embed_query("a"))
        asyncio.run(embedding.embed_query("b"))
        assert counting == ["a", "b"]

    def test_changed_model_or_dimensions_miss_the_cache(self, counting, monkeypatch):
        asyncio.run(embedding.embed_query("a"))
        monkeypatch.setenv("EMBEDDING_MODEL", "openai:other")
        asyncio.run(embedding.embed_query("a"))
        assert len(counting) == 2

    def test_cache_size_is_limited_and_evicts_oldest(self, counting, monkeypatch):
        monkeypatch.setattr(embedding, "QUERY_CACHE_SIZE", 2)
        for text in ("a", "b", "a", "c"):
            asyncio.run(embedding.embed_query(text))
        assert len(embedding._query_cache) == 2
        asyncio.run(embedding.embed_query("a"))
        assert counting == ["a", "b", "c"]
        asyncio.run(embedding.embed_query("b"))
        assert counting == ["a", "b", "c", "b"]

    def test_failure_is_not_cached(self, monkeypatch):
        attempts = []

        class FailsOnce:
            async def aembed_query(self, text):
                attempts.append(text)
                if len(attempts) == 1:
                    raise RuntimeError("x")
                return [0.1] * 1024

        monkeypatch.setattr(embedding, "init_embeddings", lambda model, **kwargs: FailsOnce())
        with pytest.raises(RuntimeError):
            asyncio.run(embedding.embed_query("q"))
        assert asyncio.run(embedding.embed_query("q")) == [0.1] * 1024

    def test_wrong_dimensions_are_not_cached(self, init, monkeypatch):
        monkeypatch.setattr(FakeEmbedder, "size", 2)
        with pytest.raises(AiUnavailableError):
            asyncio.run(embedding.embed_query("q"))
        assert not embedding._query_cache

