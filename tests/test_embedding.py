import asyncio
import json

import httpx
import pytest
from langchain_aws import BedrockEmbeddings

from app.core.errors import AiUnavailableError
from app.retrieval import embedding


class FakeEmbedder:
    size = 1024

    async def aembed_query(self, text):
        return [0.1] * self.size

    async def aembed_documents(self, texts):
        return [[0.1] * self.size for _ in texts]


def test_upstage_test_mode_uses_matching_query_and_passage_models(monkeypatch):
    calls = []

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, headers, json):
            calls.append((url, json["model"], json["input"]))
            return httpx.Response(200, json={"data": [
                {"index": index, "embedding": [0.1] * 1024} for index in range(len(json["input"]))
            ]}, request=httpx.Request("POST", url))

    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("USE_SAMPLE_DATA", "true")
    monkeypatch.setenv("UPSTAGE_API_KEY", "test-key")
    monkeypatch.setattr(embedding.httpx, "AsyncClient", Client)
    asyncio.run(embedding.embed_query("질문"))
    asyncio.run(embedding.embed_documents(["문서"]))
    assert [item[1] for item in calls] == ["solar-embedding-2-query", "solar-embedding-2-passage"]


def test_upstage_test_key_refuses_real_database_mode(monkeypatch):
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("USE_SAMPLE_DATA", "false")
    monkeypatch.setenv("UPSTAGE_API_KEY", "test-key")
    with pytest.raises(AiUnavailableError, match="샘플 데이터"):
        embedding._embedder()


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
    assert init == [("bedrock:amazon.titan-embed-text-v2:0", {"model_kwargs": {"dimensions": 1024, "normalize": True}})]


def test_bedrock_region_overrides_aws_default(monkeypatch, init):
    monkeypatch.setenv("EMBEDDING_MODEL", "bedrock:amazon.titan-embed-text-v2:0")
    monkeypatch.setenv("AWS_REGION", "ap-southeast-1")
    monkeypatch.setenv("BEDROCK_REGION", "ap-northeast-1")
    embedding._build_embedder.cache_clear()
    asyncio.run(embedding.embed_query("질문"))
    assert init[0][1]["region_name"] == "ap-northeast-1"
    embedding._build_embedder.cache_clear()


def test_real_bedrock_client_uses_bedrock_region(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "ap-southeast-1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-southeast-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake")
    client = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", region_name="ap-northeast-1")
    assert client.client.meta.region_name == "ap-northeast-1"


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
        monkeypatch.setattr(embedding, "BATCH_TIMEOUT_SECONDS", 0.01)
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


class TestProxy:
    URL = "http://localhost:18880/api/internal/embeddings"

    @pytest.fixture
    def proxy(self, monkeypatch):
        state = {"requests": [], "status": 200, "extra": []}
        real_client = httpx.AsyncClient

        def handler(request):
            body = json.loads(request.content)
            state["requests"].append((str(request.url), request.headers.get("X-Internal-Secret"), body["texts"]))
            vectors = [[0.5] * 1024 for _ in body["texts"]] + state["extra"]
            return httpx.Response(state["status"], json={"dimensions": 1024, "embeddings": vectors})

        monkeypatch.setenv("EMBEDDING_PROXY_URL", self.URL)
        monkeypatch.setenv("INTERNAL_SECRET", "fake-secret")
        monkeypatch.setattr(
            embedding.httpx,
            "AsyncClient",
            lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
        )
        embedding._query_cache.clear()
        return state

    def test_query_sends_secret_header_and_text(self, proxy):
        assert len(asyncio.run(embedding.embed_query("질문"))) == 1024
        assert proxy["requests"] == [(self.URL, "fake-secret", ["질문"])]

    def test_documents_are_split_to_proxy_limit(self, proxy):
        vectors = asyncio.run(embedding.embed_documents([f"t{i}" for i in range(33)]))
        assert len(vectors) == 33
        assert [len(request[2]) for request in proxy["requests"]] == [32, 1]

    def test_batch_size_fits_proxy_limit(self):
        assert embedding.BATCH_SIZE <= 32

    def test_error_status_raises_without_secret_in_message(self, proxy):
        proxy["status"] = 503
        with pytest.raises(httpx.HTTPStatusError) as error:
            asyncio.run(embedding.embed_query("질문"))
        assert "fake-secret" not in str(error.value)

    def test_count_mismatch_is_rejected(self, proxy):
        proxy["extra"] = [[0.5] * 1024]
        with pytest.raises(AiUnavailableError):
            asyncio.run(embedding.embed_documents(["a"]))

    def test_wrong_dimensions_are_rejected(self, monkeypatch):
        async def short(self, texts):
            return [[0.5]]

        monkeypatch.setenv("EMBEDDING_PROXY_URL", self.URL)
        monkeypatch.setenv("INTERNAL_SECRET", "fake-secret")
        monkeypatch.setattr(embedding.ProxyEmbedder, "aembed_documents", short)
        with pytest.raises(AiUnavailableError):
            asyncio.run(embedding.embed_documents(["a"]))

    def test_missing_secret_fails_before_sending(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROXY_URL", self.URL)
        monkeypatch.setenv("INTERNAL_SECRET", "")
        with pytest.raises(AiUnavailableError):
            asyncio.run(embedding.embed_query("질문"))

    def test_production_ignores_proxy(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("EMBEDDING_PROXY_URL", self.URL)
        assert embedding.config.embedding_proxy_url() is None
