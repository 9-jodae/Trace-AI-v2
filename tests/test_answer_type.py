from app.agent import graph
from app.core.errors import AiUnavailableError


class TestAnswerType:
    NO_RESULT = {"chunks": [], "failed_sources": []}

    def test_answer(self, ask, ai):
        assert ask("q").json()["answerType"] == "ANSWER"

    def test_small_talk(self, ask, ai):
        ai.small_talk = 0.95
        ai.reply = "안녕하세요!"
        assert ask("안녕?").json()["answerType"] == "SMALL_TALK"

    def test_no_evidence(self, ask, ai):
        ai.search_result = self.NO_RESULT
        assert ask("q").json()["answerType"] == "NO_EVIDENCE"

    def test_no_evidence_when_reply_fails(self, ask, ai):
        ai.search_result = self.NO_RESULT
        ai.reply = AiUnavailableError("x")
        assert ask("q").json()["answerType"] == "NO_EVIDENCE"

    def test_clarification_without_documents(self, ask, ai):
        ai.search_result = self.NO_RESULT
        ai.reply = "어떤 기간인가요?"
        ai.reply_clarification = True
        assert ask("기간은?").json()["answerType"] == "CLARIFICATION"

    def test_repeated_clarification_without_documents_is_refused(self, ask, ai):
        ai.search_result = self.NO_RESULT
        ai.reply = "어떤 기간인가요?"
        ai.reply_clarification = True
        ask("기간은?")
        body = ask("글쎄").json()
        assert body["answerType"] == "NO_EVIDENCE" and body["answer"] == graph.NO_EVIDENCE_ANSWER
