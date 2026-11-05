import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import safety

safety.install()

import pytest
from fastapi.testclient import TestClient

import main
from app.agent import graph
from app.jev import client as jev_client
from app.llm import client as llm_client
from app.retrieval import embedding


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _forbidden(*args, **kwargs):
    raise AssertionError("테스트에서 실제 외부 클라이언트를 만들려고 했습니다.")


@pytest.fixture(autouse=True)
def forbid_real_clients(monkeypatch):
    # 각 테스트가 명시적으로 가짜로 바꾸지 않는 한 실제 SDK 클라이언트는 생성할 수 없다
    monkeypatch.setattr(jev_client, "AsyncTypeSafeClient", _forbidden)
    monkeypatch.setattr(llm_client, "init_chat_model", _forbidden)
    monkeypatch.setattr(embedding, "init_embeddings", _forbidden)
    # 재사용하는 클라이언트가 테스트 사이에 남지 않게 한다
    embedding._build_embedder.cache_clear()
    embedding._query_cache.clear()
    llm_client._build_model.cache_clear()
    monkeypatch.setattr(jev_client, "_client", None)
    monkeypatch.setattr(jev_client, "_client_settings", None)


def pytest_sessionfinish(session):
    if safety.blocked_attempts:
        session.exitstatus = 1


def pytest_terminal_summary(terminalreporter):
    terminalreporter.write_line(f"차단된 외부 접속 시도: {len(safety.blocked_attempts)}건 {safety.blocked_attempts}")


def chunk(chunk_id: str, text: str | None = None, source: str = "문서") -> dict:
    return {"id": chunk_id, "text": text or f"{chunk_id} 내용", "source": source}


class FakeAi:
    """JEV, 검색, LLM을 대신하며 호출 횟수와 인자를 기록한다."""

    def __init__(self):
        self.probability = 0.9
        self.search_result = {"chunks": [chunk("a"), chunk("b")], "failed_sources": []}
        self.llm_error = None
        self.jev_calls = []
        self.search_calls = []
        self.trace_calls = []
        self.llm_calls = []
        self.small_talk = 0.0
        self.small_talk_calls = []
        self.reply = graph.NO_EVIDENCE_ANSWER
        self.reply_clarification = False
        self.reply_calls = []
        self.reply_allowed = []
        self.rewrite = None
        self.rewrite_calls = []

    async def small_talk_probability(self, question):
        self.small_talk_calls.append(question)
        if isinstance(self.small_talk, Exception):
            raise self.small_talk
        return self.small_talk

    async def rewrite_query(self, question):
        self.rewrite_calls.append(question)
        if isinstance(self.rewrite, Exception):
            raise self.rewrite
        return self.rewrite

    async def generate_reply(self, question, allow_clarification=True):
        self.reply_calls.append(question)
        self.reply_allowed.append(allow_clarification)
        if isinstance(self.reply, Exception):
            raise self.reply
        return llm_client.NoDocumentReply(self.reply, self.reply_clarification)

    async def can_answer(self, question, context, history):
        self.jev_calls.append({"question": question, "context": context, "history": history})
        if isinstance(self.probability, Exception):
            raise self.probability
        return self.probability

    async def filter_context(self, chunks, scope=None):
        self.filter_scope = scope
        return chunks

    async def search(self, query, filters=None, scope=None, include_trace=False, expected_document_ids=None):
        self.search_calls.append(query)
        self.search_filters = filters
        self.search_scope = scope
        if isinstance(self.search_result, Exception):
            raise self.search_result
        if include_trace and isinstance(self.search_result, dict):
            self.trace_calls.append((query, expected_document_ids))
            return {
                **self.search_result,
                "retrieval_trace": self.search_result.get(
                    "retrieval_trace", {"query": query, "stages": {}, "rrf": [], "final": [],
                                        "evaluation": {"expected_document_ids": expected_document_ids}}
                ),
            }
        return self.search_result

    async def generate(self, question, context, history):
        self.llm_calls.append({"question": question, "context": context, "history": [m.text for m in history]})
        if self.llm_error:
            raise self.llm_error
        return f"답변: {question}"


@pytest.fixture
def ai(monkeypatch):
    fake = FakeAi()
    monkeypatch.setattr(graph, "can_answer_probability", fake.can_answer)
    monkeypatch.setattr(graph, "small_talk_probability", fake.small_talk_probability)
    monkeypatch.setattr(graph, "generate_reply_without_documents", fake.generate_reply)
    monkeypatch.setattr(graph, "rewrite_search_query", fake.rewrite_query)
    monkeypatch.setattr(graph, "search", fake.search)
    monkeypatch.setattr(graph, "filter_accessible_context", fake.filter_context)
    monkeypatch.setattr(graph, "generate_answer", fake.generate)
    return fake


@pytest.fixture
def client():
    with TestClient(main.app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def ask(client, ai):
    def _ask(message, thread_id="thread-1"):
        return client.post("/api/v1/agent/chat", json={"message": message, "threadId": thread_id})

    return _ask
