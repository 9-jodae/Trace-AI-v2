"""#16: 요청 전체 시간 상한과 단계별 시간 로그."""

import asyncio
import logging

import pytest
from conftest import chunk
from langchain_core.messages import AIMessage

from app.agent import graph
from app.core import config
from app.llm import client as llm

URL = "/api/v1/agent/chat"


@pytest.fixture
def short_limit(monkeypatch):
    monkeypatch.setattr(config, "request_timeout_seconds", lambda: 0.05)


def hang_generate(ai, monkeypatch):
    async def hang(question, context, history):
        await asyncio.sleep(10)

    monkeypatch.setattr(graph, "generate_answer", hang)


class TestRequestTimeout:
    def test_default_and_setting(self, monkeypatch):
        assert config.request_timeout_seconds() == 60
        monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "90")
        assert config.request_timeout_seconds() == 90

    def test_hanging_request_returns_503(self, ask, ai, short_limit, monkeypatch):
        hang_generate(ai, monkeypatch)
        response = ask("q")
        assert response.status_code == 503 and response.json()["code"] == "AI_UNAVAILABLE"

    def test_timed_out_turn_is_not_saved(self, ask, ai, client, short_limit, monkeypatch):
        hang_generate(ai, monkeypatch)
        ask("q", "t")
        assert client.app.state.conversations.get("t") == {"messages": [], "context": []}

    def test_thread_works_after_timeout(self, ask, ai, short_limit, monkeypatch):
        with monkeypatch.context() as patch:
            hang_generate(ai, patch)
            assert ask("q1", "t").status_code == 503
        assert ask("q2", "t").status_code == 200

    def test_fast_request_is_unaffected(self, ask, ai, monkeypatch):
        monkeypatch.setattr(config, "request_timeout_seconds", lambda: 5)
        assert ask("q").status_code == 200

    def test_limit_stops_fallback_model(self, client, monkeypatch):
        calls = []

        class Hanging:
            async def ainvoke(self, messages):
                await asyncio.sleep(10)
                return AIMessage(content="늦은 답변")

        def factory(name, **kwargs):
            calls.append(name)
            return Hanging()

        async def fake_search(query, filters=None):
            return {"chunks": [chunk("a")], "failed_sources": []}

        monkeypatch.setenv("LLM_FALLBACK_MODEL", "fake:backup")
        monkeypatch.setattr(llm, "init_chat_model", factory)
        monkeypatch.setattr(graph, "search", fake_search)
        monkeypatch.setattr(config, "request_timeout_seconds", lambda: 0.05)
        assert client.post(URL, json={"message": "q"}).status_code == 503
        assert calls == ["fake:fake-llm"]


class TestStepLogs:
    @pytest.fixture(autouse=True)
    def capture(self, caplog):
        caplog.set_level(logging.INFO, logger="app")

    def test_each_step_and_total_are_logged(self, ask, ai, caplog):
        ask("q1", "t")
        ask("q2", "t")
        messages = [record.getMessage() for record in caplog.records]
        for step in ("judge_context", "retrieve", "generate"):
            assert any(f"step={step} elapsed_ms=" in message for message in messages)
        assert any("route=SEARCH total_ms=" in message for message in messages)
        assert any("route=CONTEXT total_ms=" in message for message in messages)

    def test_failed_step_is_still_logged(self, ask, ai, caplog):
        ai.search_result = RuntimeError("x")
        ask("q")
        assert any("step=retrieve elapsed_ms=" in record.getMessage() for record in caplog.records)

    def test_logs_have_no_question_or_document_text(self, ask, ai, caplog):
        secret = "비밀-질문"
        ai.search_result = {"chunks": [chunk("a", "비밀-본문")], "failed_sources": []}
        ask(secret)
        assert secret not in caplog.text and "비밀-본문" not in caplog.text
