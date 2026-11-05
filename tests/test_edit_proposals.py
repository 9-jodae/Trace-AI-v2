import json

import pytest
from conftest import chunk

from app.agent import graph
from app.agent.evidence import ground_edits, parse_edit_proposals
from app.core import config

BEFORE = "이월은 최대 7일이다."
TEXT = f"앞 문장이다. {BEFORE} 뒤 문장이다."
CONTEXT = [{**chunk("c1", TEXT), "source": "연차 지침", "document_id": "d1", "version_id": "v1", "char_start": 100}]


def edit(**values):
    return {"chunk": 1, "before": BEFORE, "after": "이월은 최대 5일이다.", "reason": "취업규칙에 맞춤", **values}


class TestGroundEdits:
    def test_locates_sentence_in_source(self):
        [proposal] = ground_edits([edit()], CONTEXT)
        start = TEXT.index(BEFORE)
        assert proposal["document_id"] == "d1" and proposal["version_id"] == "v1"
        assert proposal["char_start"] == 100 + start and proposal["char_end"] == 100 + start + len(BEFORE)
        assert proposal["after"] == "이월은 최대 5일이다."

    @pytest.mark.parametrize(
        "bad",
        [
            edit(before="문서에 없는 문장"),
            edit(after=BEFORE),
            edit(after="  "),
            edit(before=" "),
            edit(chunk=9),
            edit(chunk="1"),
            edit(after="가" * 1001),
            "문자열",
        ],
    )
    def test_invalid_edit_is_dropped(self, bad):
        assert ground_edits([bad], CONTEXT) == []

    @pytest.mark.parametrize("raw", [None, "edits", {"chunk": 1}])
    def test_invalid_shape_is_empty(self, raw):
        assert ground_edits(raw, CONTEXT) == []

    def test_ambiguous_sentence_is_dropped(self):
        context = [{**CONTEXT[0], "text": f"{BEFORE} {BEFORE}"}]
        assert ground_edits([edit()], context) == []

    def test_duplicates_and_limit(self):
        text = " ".join(f"문장{index}이다." for index in range(7))
        context = [{**CONTEXT[0], "text": text}]
        edits = [edit(before=f"문장{index}이다.", after="바꿈") for index in range(7)]
        assert len(ground_edits([edits[0], edits[0]], context)) == 1
        assert len(ground_edits(edits, context)) == 5


def test_parse_reads_edits_from_answer():
    response = json.dumps({"answer": "제안입니다.", "evidence": [], "edits": [edit()]}, ensure_ascii=False)
    assert len(parse_edit_proposals(response, CONTEXT)) == 1
    assert parse_edit_proposals("그냥 글", CONTEXT) == []


class TestGraph:
    @pytest.fixture
    def judged(self, monkeypatch):
        monkeypatch.setattr(config, "jev_api_key", lambda: "k")

        def set_values(small_talk=0.0, edit_request=0.0):
            async def small(_question):
                if isinstance(small_talk, Exception):
                    raise small_talk
                return small_talk

            async def editing(_question):
                if isinstance(edit_request, Exception):
                    raise edit_request
                return edit_request

            monkeypatch.setattr(graph, "small_talk_probability", small)
            monkeypatch.setattr(graph, "edit_request_probability", editing)

        return set_values

    @staticmethod
    def state(text="최대 5일로 수정해줘"):
        from langchain_core.messages import HumanMessage

        return {"messages": [HumanMessage(content=text)]}

    @pytest.mark.parametrize(
        "probability, expected",
        [(0.9, True), (0.8, True), (0.5, False), (RuntimeError("x"), False)],
    )
    @pytest.mark.anyio
    async def test_judge_threshold_and_failure(self, judged, probability, expected):
        judged(edit_request=probability)
        result = await graph.judge_small_talk(self.state())
        assert result["edit_request"] is expected and result["small_talk"] is False

    @pytest.mark.anyio
    async def test_no_key_means_no_edit_request(self, monkeypatch):
        monkeypatch.setattr(config, "jev_api_key", lambda: None)
        assert (await graph.judge_small_talk(self.state()))["edit_request"] is False

    def test_edit_request_searches_with_previous_question(self):
        from langchain_core.messages import AIMessage, HumanMessage

        messages = [
            HumanMessage(content="연차 이월 충돌이 뭐야?"),
            AIMessage(content="답"),
            HumanMessage(content="5일로 수정해줘"),
        ]
        searched = graph._search_query({"messages": messages, "edit_request": True})
        assert searched == "연차 이월 충돌이 뭐야?\n5일로 수정해줘"
        assert graph._search_query({"messages": messages, "edit_request": False}) == "5일로 수정해줘"

    @pytest.fixture
    def answer(self, monkeypatch):
        seen = {}

        def set_response(response):
            async def generate(question, context, history, **kwargs):
                seen.update(kwargs)
                return response

            monkeypatch.setattr(graph, "generate_answer", generate)
            return seen

        return set_response

    @pytest.mark.anyio
    async def test_edit_request_returns_proposals(self, answer):
        seen = answer(json.dumps({"answer": "제안입니다.", "edits": [edit()]}, ensure_ascii=False))
        result = await graph._answer_with_documents({**self.state(), "edit_request": True}, CONTEXT, None)
        assert seen == {"edit_request": True} and len(result["edit_proposals"]) == 1

    @pytest.mark.anyio
    async def test_ordinary_question_ignores_edits(self, answer):
        seen = answer(json.dumps({"answer": "답.", "edits": [edit()]}, ensure_ascii=False))
        result = await graph._answer_with_documents(self.state("연차 며칠?"), CONTEXT, None)
        assert seen == {} and result["edit_proposals"] == []

    @pytest.mark.anyio
    async def test_refusal_has_no_proposals(self, answer):
        answer(json.dumps({"answer": "관련 문서를 찾지 못했습니다.", "edits": [edit()]}, ensure_ascii=False))
        result = await graph._answer_with_documents({**self.state(), "edit_request": True}, CONTEXT, None)
        assert result["edit_proposals"] == []
