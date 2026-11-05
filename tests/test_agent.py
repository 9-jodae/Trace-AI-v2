import pytest
from conftest import chunk

from app.agent import graph
from app.core.errors import AiUnavailableError


class TestFirstQuestion:
    def test_searches_without_jev(self, ask, ai):
        body = ask("첫 질문").json()
        assert ai.jev_calls == [] and ai.search_calls == ["첫 질문"] and len(ai.llm_calls) == 1
        assert body["route"] == "SEARCH" and body["contextProbability"] is None and body["fallbackReason"] is None

    def test_answer_and_sources(self, ask, ai):
        body = ask("첫 질문").json()
        assert body["answer"] == "답변: 첫 질문"
        assert [s["id"] for s in body["sources"]] == ["a", "b"]


class TestContextJudgment:
    def test_high_probability_reuses_context(self, ask, ai):
        ask("q1")
        body = ask("q2").json()
        assert body["route"] == "CONTEXT" and body["contextProbability"] == 0.9
        assert ai.search_calls == ["q1"]
        assert [c["id"] for c in ai.llm_calls[-1]["context"]] == ["a", "b"]

    @pytest.mark.parametrize(
        "probability, route",
        [(1.0, "CONTEXT"), (0.7, "CONTEXT"), (0.6999, "SEARCH"), (0.0, "SEARCH")],
    )
    def test_threshold(self, ask, ai, probability, route):
        ask("q1")
        ai.probability = probability
        assert ask("q2").json()["route"] == route

    def test_low_probability_replaces_context(self, ask, ai):
        ask("q1")
        ai.probability = 0.1
        ai.search_result = {"chunks": [chunk("new")], "failed_sources": []}
        body = ask("q2").json()
        assert [s["id"] for s in body["sources"]] == ["new"]
        ai.probability = 0.9
        assert [s["id"] for s in ask("q3").json()["sources"]] == ["new"]

    def test_jev_receives_question_context_history(self, ask, ai):
        ask("q1")
        ask("q2")
        call = ai.jev_calls[0]
        assert call["question"] == "q2"
        assert [c["id"] for c in call["context"]] == ["a", "b"]
        assert call["history"] == ["human: q1", "ai: 답변: q1"]

    def test_jev_failure_falls_back_to_search(self, ask, ai):
        ask("q1")
        ai.probability = RuntimeError("비밀")
        body = ask("q2").json()
        assert body["route"] == "SEARCH" and body["contextProbability"] is None
        assert body["fallbackReason"] == "JEV 호출 실패 (RuntimeError)"
        assert "비밀" not in str(body)

    def test_missing_jev_key_falls_back_to_search(self, ask, ai, monkeypatch):
        ask("q1")
        monkeypatch.setenv("JEV_API_KEY", "")
        body = ask("q2").json()
        assert body["route"] == "SEARCH" and "JEV_API_KEY" in body["fallbackReason"] and len(ai.jev_calls) == 0

    def test_fallback_reason_clears_on_next_turn(self, ask, ai):
        ask("q1")
        ai.probability = RuntimeError("x")
        ask("q2")
        ai.probability = 0.9
        assert ask("q3").json()["fallbackReason"] is None


class TestSearchFailures:
    def test_partial_failure_is_reported(self, ask, ai):
        ai.search_result = {"chunks": [chunk("a")], "failed_sources": ["vector"]}
        assert ask("q").json()["failedSources"] == ["vector"]

    def test_failed_sources_clear_on_reuse(self, ask, ai):
        ai.search_result = {"chunks": [chunk("a")], "failed_sources": ["vector"]}
        ask("q1")
        assert ask("q2").json()["failedSources"] == []

    def test_total_failure_is_503(self, ask, ai):
        ai.search_result = AiUnavailableError("x")
        response = ask("q")
        assert response.status_code == 503 and response.json()["code"] == "AI_UNAVAILABLE"
        assert ai.llm_calls == []

    def test_total_failure_keeps_old_context(self, ask, ai):
        ask("q1")
        ai.probability = 0.1
        ai.search_result = AiUnavailableError("x")
        assert ask("q2").status_code == 503
        ai.probability = 0.9
        assert [s["id"] for s in ask("q3").json()["sources"]] == ["a", "b"]

    def test_failed_turn_is_not_left_in_history(self, ask, ai):
        ai.search_result = AiUnavailableError("x")
        assert ask("실패한 질문").status_code == 503
        ai.search_result = {"chunks": [chunk("a")], "failed_sources": []}
        ask("다음 질문")
        assert "실패한 질문" not in ai.llm_calls[-1]["history"]


class TestNoResults:
    def test_fixed_answer_without_llm(self, ask, ai):
        ai.search_result = {"chunks": [], "failed_sources": []}
        body = ask("q").json()
        assert body["answer"] == graph.NO_EVIDENCE_ANSWER and body["sources"] == [] and ai.llm_calls == []

    def test_next_question_skips_jev(self, ask, ai):
        ai.search_result = {"chunks": [], "failed_sources": []}
        ask("q1")
        ask("q2")
        assert ai.jev_calls == []


class TestLlmFailure:
    def test_is_503(self, ask, ai):
        ai.llm_error = AiUnavailableError("x")
        assert ask("q").status_code == 503

    def test_recovers_on_next_request(self, ask, ai):
        ai.llm_error = AiUnavailableError("x")
        ask("q1")
        ai.llm_error = None
        assert ask("q2").status_code == 200


class TestHistoryAndThreads:
    def test_history_is_limited(self, ask, ai):
        for index in range(6):
            ask(f"q{index}")
        assert len(ai.llm_calls[-1]["history"]) == graph.HISTORY_LIMIT

    def test_history_excludes_current_question(self, ask, ai):
        ask("q1")
        assert ai.llm_calls[-1]["history"] == []

    def test_threads_do_not_share_context(self, ask, ai):
        ask("q1", "thread-A")
        ask("q1", "thread-B")
        assert ai.jev_calls == [] and len(ai.search_calls) == 2

    def test_threads_do_not_share_history(self, ask, ai):
        ask("비밀 질문", "thread-A")
        ask("q", "thread-B")
        assert ai.llm_calls[-1]["history"] == []

    def test_graph_structure(self):
        edges = {(edge.source, edge.target) for edge in graph.build_agent().get_graph().edges}
        assert ("judge_context", "generate") in edges and ("judge_context", "retrieve") in edges
        assert ("retrieve", "generate") in edges and ("generate", "__end__") in edges
