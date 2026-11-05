"""외부 오류 원문과 요청 내용이 로그에 남지 않는지 확인한다."""

import asyncio
import logging

import pytest
from conftest import chunk

from app.core.errors import AiUnavailableError, describe_exception
from app.llm import client as llm
from app.retrieval import service

SECRET = "비밀-문서-본문"
URL = "/api/v1/agent/chat"


@pytest.fixture(autouse=True)
def capture(caplog):
    caplog.set_level(logging.DEBUG)


def assert_clean(caplog):
    assert SECRET not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


class TestDescribeException:
    def test_has_type_and_location_not_message(self):
        try:
            raise ValueError(SECRET)
        except ValueError as exc:
            text = describe_exception(exc)
        assert text.startswith("ValueError at test_logging.py:") and SECRET not in text

    @pytest.mark.parametrize("name", ["status_code", "code"])
    def test_includes_status_code(self, name):
        error = RuntimeError(SECRET)
        setattr(error, name, 503)
        assert describe_exception(error) == "RuntimeError (503)"

    def test_ignores_non_numeric_code(self):
        error = RuntimeError("x")
        error.code = SECRET
        assert describe_exception(error) == "RuntimeError"


def test_llm_failure_log_has_no_exception_text(monkeypatch, caplog):
    class Failing:
        async def ainvoke(self, messages):
            raise RuntimeError(SECRET)

    monkeypatch.setattr(llm, "init_chat_model", lambda model, **kwargs: Failing())
    with pytest.raises(AiUnavailableError):
        asyncio.run(llm.generate_answer("질문", [chunk("a")], []))
    assert "RuntimeError" in caplog.text
    assert_clean(caplog)


def test_jev_failure_log_has_no_exception_text(ask, ai, caplog):
    ask("q1")
    ai.probability = RuntimeError(SECRET)
    assert ask("q2").json()["route"] == "SEARCH"
    assert "RuntimeError" in caplog.text
    assert_clean(caplog)


def test_search_failure_log_has_no_exception_text(monkeypatch, caplog):
    class Failing:
        async def search(self, value, limit, scope, filters=None):
            raise RuntimeError(SECRET)

        async def filter_accessible(self, chunk_ids, scope, filters=None):
            return set()

    class Empty:
        async def search(self, value, limit):
            return []

    async def embed(query):
        return [0.1]

    monkeypatch.setenv("ACCESS_CLEARANCE", "INTERNAL")
    monkeypatch.setattr(service, "get_stores", lambda: (Failing(), Empty()))
    monkeypatch.setattr(service, "embed_query", embed)
    assert asyncio.run(service.search("질문"))["failed_sources"] == ["vector"]
    assert "RuntimeError" in caplog.text
    assert_clean(caplog)


def test_unhandled_error_log_has_no_exception_text(client, ai, caplog):
    ai.search_result = RuntimeError(SECRET)
    assert client.post(URL, json={"message": "q"}).status_code == 500
    assert "RuntimeError" in caplog.text
    assert_clean(caplog)
