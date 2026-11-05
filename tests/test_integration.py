"""저장소, RRF, LangGraph, JEV·LLM 클라이언트를 실제 코드로 묶고 외부 SDK만 가짜로 바꾼다."""

import pytest
from langchain_core.messages import AIMessage

from app.jev import client as jev_client
from app.llm import client as llm_client
from app.retrieval import embedding, service, stores


class FakeEmbedder:
    def __init__(self, counter):
        self.counter = counter

    async def aembed_query(self, text):
        self.counter["query"] += 1
        return [1.0, 0.0]

    async def aembed_documents(self, texts):
        self.counter["documents"] += 1
        return [[1.0, 0.0] if "승인" in text else [0.0, 1.0] for text in texts]


class FakeChatModel:
    def __init__(self, counter):
        self.counter = counter

    async def ainvoke(self, messages):
        self.counter["llm"] += 1
        self.counter["prompt"] = messages[-1].content
        return AIMessage(content="정리된 답변")


class FakeJevClient:
    probability = 0.9
    counter = None

    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def aclose(self):
        pass

    async def system_one(self, state, questions):
        small_talk = "small_talk" in questions
        if not small_talk:
            FakeJevClient.counter["jev"] += 1
            FakeJevClient.counter["state"] = state
        probability = 0.0 if small_talk else FakeJevClient.probability

        class Noul:
            noul = probability

        class Response:
            nouls = {"small_talk" if small_talk else "can_answer": Noul}

        return Response


@pytest.fixture
def counter(monkeypatch):
    calls = {"query": 0, "documents": 0, "llm": 0, "jev": 0}
    FakeJevClient.counter = calls
    FakeJevClient.probability = 0.9
    monkeypatch.setattr(embedding, "init_embeddings", lambda model, **kwargs: FakeEmbedder(calls))
    monkeypatch.setattr(llm_client, "init_chat_model", lambda model, **kwargs: FakeChatModel(calls))
    monkeypatch.setattr(jev_client, "AsyncTypeSafeClient", FakeJevClient)
    monkeypatch.setenv("USE_SAMPLE_DATA", "true")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "2")
    service._sample_stores.cache_clear()
    return calls


def post(client, message, thread="t"):
    return client.post("/api/v1/agent/chat", json={"message": message, "threadId": thread})


def test_first_question_runs_full_search_path(client, counter):
    body = post(client, "재택근무 규정을 누가 승인했어?").json()
    assert body["route"] == "SEARCH" and body["answer"] == "정리된 답변" and body["failedSources"] == []
    assert body["sources"][0]["id"] == "remote-1"
    assert len(body["sources"]) == service.CONTEXT_LIMIT
    assert counter == {**counter, "query": 1, "documents": 1, "llm": 1, "jev": 0}
    assert "재택근무 규정은 인사팀이 소유" in counter["prompt"]


def test_follow_up_saves_search_cost(client, counter):
    post(client, "재택근무 규정을 누가 승인했어?")
    body = post(client, "그 규정의 소유 부서는?").json()
    assert body["route"] == "CONTEXT" and body["contextProbability"] == 0.9
    assert counter["query"] == 1 and counter["jev"] == 1 and counter["llm"] == 2
    assert "재택근무 규정은 인사팀이 소유" in counter["state"] and "그 규정의 소유 부서는?" in counter["state"]


def test_follow_up_searches_again_when_context_is_insufficient(client, counter):
    post(client, "재택근무 규정을 누가 승인했어?")
    FakeJevClient.probability = 0.1
    body = post(client, "E4012 오류는 왜 나?").json()
    assert body["route"] == "SEARCH" and counter["query"] == 2 and counter["llm"] == 2


def test_missing_llm_model_is_503_but_keeps_format(client, counter, monkeypatch):
    # 현재 .env처럼 LLM_MODEL이 비어 있는 상태
    monkeypatch.setenv("LLM_MODEL", "")
    response = post(client, "재택근무 규정")
    assert response.status_code == 503 and response.json()["code"] == "AI_UNAVAILABLE"
    assert counter["llm"] == 0


def test_missing_embedding_uses_graph_only(client, counter, monkeypatch):
    monkeypatch.setenv("EMBEDDING_API_KEY", "")
    body = post(client, "결제 서비스 담당은?").json()
    assert body["failedSources"] == ["vector"]
    assert [s["id"] for s in body["sources"]] == ["payment-1", "payment-2"]


def test_missing_embedding_and_no_graph_match_answers_no_result(client, counter, monkeypatch):
    monkeypatch.setenv("EMBEDDING_API_KEY", "")
    body = post(client, "관계없는 질문").json()
    # 다른 표기를 만들고 응대 모델로 답한다. 문서 내용은 넘기지 않는다
    assert body["sources"] == [] and body["failedSources"] == ["vector"] and counter["llm"] == 2
    assert counter["prompt"] == "관계없는 질문"


def test_invalid_dimensions_only_disable_vector_search(client, counter, monkeypatch):
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "abc")
    assert post(client, "결제 서비스").json()["failedSources"] == ["vector"]


def test_database_connection_failure_exposes_nothing(client, counter, monkeypatch):
    # 그래프 결과의 권한도 PostgreSQL로 확인하므로 DB가 멈추면 그래프 결과도 내보내지 않는다
    class RefusingPool:
        check_connection = None

        def __init__(self, *args, **kwargs):
            pass

        async def open(self, wait, timeout):
            raise OSError("connection refused")

        async def close(self):
            pass

    monkeypatch.setattr(stores, "AsyncConnectionPool", RefusingPool)
    monkeypatch.setenv("ACCESS_CLEARANCE", "INTERNAL")
    monkeypatch.setattr(service, "get_stores", lambda: (stores.PostgresVectorStore("db"), stores.SampleGraphStore()))
    assert post(client, "결제 서비스").status_code == 503


def test_wrong_embedding_dimensions_only_disable_vector_search(client, counter, monkeypatch):
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1024")
    body = post(client, "결제 서비스").json()
    assert body["failedSources"] == ["vector"] and body["sources"]
