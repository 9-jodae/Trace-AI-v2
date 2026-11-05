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
    def test_replies_without_documents(self, ask, ai):
        ai.search_result = {"chunks": [], "failed_sources": []}
        ai.reply = "어떤 기간에 관한 질문인가요? 좀 더 자세히 말씀해 주세요."
        body = ask("기간이 얼마나 됨?").json()
        assert body["answer"] == ai.reply and body["sources"] == [] and body["evidence"] == []
        assert ai.llm_calls == [] and ai.reply_calls == ["기간이 얼마나 됨?"]

    def test_reply_failure_falls_back_to_fixed_answer(self, ask, ai):
        ai.search_result = {"chunks": [], "failed_sources": []}
        ai.reply = AiUnavailableError("LLM 호출 실패")
        response = ask("q")
        assert response.status_code == 200 and response.json()["answer"] == graph.NO_EVIDENCE_ANSWER

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


class TestSmallTalk:
    def test_high_probability_skips_search_and_answer_model(self, ask, ai):
        ai.small_talk = 0.9
        ai.reply = "안녕하세요!"
        body = ask("안녕?").json()
        assert body["answer"] == "안녕하세요!" and body["sources"] == [] and body["evidence"] == []
        assert ai.search_calls == [] and ai.llm_calls == [] and ai.jev_calls == []

    def test_below_threshold_searches(self, ask, ai):
        ai.small_talk = 0.89
        body = ask("q").json()
        assert body["answer"] == "답변: q" and ai.reply_calls == [] and len(ai.search_calls) == 1

    def test_judge_failure_searches(self, ask, ai):
        ai.small_talk = RuntimeError("JEV down")
        assert ask("q").json()["answer"] == "답변: q" and ai.reply_calls == []

    def test_no_jev_key_searches_without_judging(self, ask, ai, monkeypatch):
        monkeypatch.setenv("JEV_API_KEY", "")
        assert ask("q").status_code == 200 and ai.small_talk_calls == [] and len(ai.search_calls) == 1

    def test_small_talk_clears_previous_sources(self, ask, ai):
        ask("q1")
        ai.small_talk = 0.95
        ai.reply = "천만에요!"
        assert ask("고마워").json()["sources"] == []
        ai.small_talk = 0.0
        ai.jev_calls.clear()
        ask("q2")
        assert ai.jev_calls == [] and len(ai.search_calls) == 2


class TestClarificationWithoutDocuments:
    @pytest.fixture
    def asked(self, ask, ai):
        ai.search_result = {"chunks": [], "failed_sources": []}
        ai.reply = "어떤 기간에 관한 질문인가요?"
        ai.reply_clarification = True
        ask("기간이 얼마나 됨?")
        ai.small_talk_calls.clear()
        return ask

    def test_next_search_keeps_previous_question(self, asked, ai):
        asked("연차요")
        assert ai.search_calls[-1] == "기간이 얼마나 됨?\n연차요"

    def test_reply_to_clarification_skips_small_talk(self, asked, ai):
        ai.small_talk = 0.99
        asked("네")
        assert ai.small_talk_calls == [] and len(ai.search_calls) == 2

    def test_does_not_clarify_twice(self, asked, ai):
        asked("글쎄")
        assert ai.reply_allowed == [True, False]
        asked("연차요")
        # 표시를 다시 붙이지 않아 다음 말은 일반 질문으로 처리한다
        assert ai.search_calls[-1] == "연차요" and ai.reply_allowed[-1] is True


class TestAlternateSearch:
    NO_RESULT = {"chunks": [], "failed_sources": []}
    FOUND = {"chunks": [chunk("leave", source="leave-policy")], "failed_sources": []}

    def test_empty_search_retries_with_other_spelling(self, ask, ai, monkeypatch):
        results = [self.NO_RESULT, self.FOUND]

        async def search(query, filters=None, scope=None):
            ai.search_calls.append(query)
            return results.pop(0)

        monkeypatch.setattr(graph, "search", search)
        ai.rewrite = "leave policy"
        body = ask("리브 팔러시가 뭐야?").json()
        assert ai.search_calls == ["리브 팔러시가 뭐야?", "leave policy"]
        assert [source["id"] for source in body["sources"]] == ["leave"]
        assert "leave policy" in ai.llm_calls[0]["question"]

    def test_no_other_spelling_does_not_search_again(self, ask, ai):
        ai.search_result = self.NO_RESULT
        assert ask("사내 주차 등록은?").json()["answer"] == graph.NO_EVIDENCE_ANSWER
        assert len(ai.search_calls) == 1 and ai.reply_calls == ["사내 주차 등록은?"]

    def test_rewrite_failure_replies_without_documents(self, ask, ai):
        ai.search_result = self.NO_RESULT
        ai.rewrite = AiUnavailableError("x")
        assert ask("q").status_code == 200 and ai.reply_calls == ["q"]

    def test_refused_answer_retries_once(self, ask, ai, monkeypatch):
        answers = [f'{{"answer": "{graph.NO_EVIDENCE_ANSWER}"}}', "답변"]

        async def generate(question, context, history):
            ai.llm_calls.append({"question": question})
            return answers.pop(0)

        monkeypatch.setattr(graph, "generate_answer", generate)
        ai.rewrite = "leave policy"
        body = ask("리브 팔러시가 뭐야?").json()
        assert body["answer"] == "답변"
        assert len(ai.llm_calls) == 2 and len(ai.search_calls) == 2 and ai.rewrite_calls == ["리브 팔러시가 뭐야?"]

    def test_small_talk_refusal_falls_through_to_search(self, ask, ai):
        ai.small_talk = 0.99
        body = ask("리브 팔러시가 뭐야?").json()
        assert len(ai.search_calls) == 1 and body["answer"].startswith("답변")

    def test_small_talk_clarification_falls_through_to_search(self, ask, ai):
        # 문서에 나오는 이름을 잡담으로 보고 되물으면 검색으로 다시 찾는다
        ai.small_talk = 0.99
        ai.reply = "무엇을 말씀하시는지 명확하지 않습니다."
        ai.reply_clarification = True
        body = ask("새결이 뭘까요?").json()
        assert len(ai.search_calls) == 1 and body["answer"].startswith("답변")
