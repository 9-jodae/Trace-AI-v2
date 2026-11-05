import json

import pytest
from conftest import chunk

from app.agent import graph
from app.agent.evidence import ground_edits, parse_edit_proposals
from app.core import config
from app.core.errors import AiUnavailableError
from app.llm.client import EDIT_REQUEST_INSTRUCTION, SYSTEM_PROMPT

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
        text = " ".join(f"문장{index}이다." for index in range(9))
        context = [{**CONTEXT[0], "text": text}]
        edits = [edit(before=f"문장{index}이다.", after="바꿈") for index in range(9)]
        assert len(ground_edits([edits[0], edits[0]], context)) == 1
        assert len(ground_edits(edits, context)) == 8


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
        result = await graph.judge_small_talk(self.state("이 내용 좀 정리해 볼까"))
        assert result["edit_request"] is expected and result["small_talk"] is False

    @pytest.mark.parametrize(
        "text",
        [
            "최대 5일로 수정해줘",
            "값 변경해줘",
            "이거 고쳐줘",
            "내용 반영해줘",
            "7일을 5일로 바꿔줘",
            "둘이 맞춰줘",
            "충돌 해결해줘",
            "이월 일수 수정 부탁",
            "5일로 해줘",
            "수정해 주세요",
        ],
    )
    @pytest.mark.anyio
    async def test_direct_edit_words_are_recognized_without_jev(self, judged, text):
        judged(edit_request=RuntimeError("down"))
        result = await graph.judge_small_talk(self.state(text))
        assert result["edit_request"] is True

    @pytest.mark.parametrize(
        "text",
        [
            "연차 이월 며칠이야?",
            "수정 방법이 뭐야?",
            "연차 규정을 변경해야 하나요?",
            "해결해야 할 충돌이 뭐야?",
            "수정해야 할 문서 알려줘",
            "변경 좀 있었어?",
            "최근 변경 내용 알려줘",
            "문서 어떻게 바꿔?",
            "충돌이 뭐야?",
            "안녕",
        ],
    )
    @pytest.mark.anyio
    async def test_ordinary_questions_are_not_edit_requests(self, judged, text):
        judged(edit_request=0.1)
        assert (await graph.judge_small_talk(self.state(text)))["edit_request"] is False

    @pytest.mark.anyio
    async def test_edit_request_is_not_small_talk(self, judged):
        judged(small_talk=0.99, edit_request=0.99)
        result = await graph.judge_small_talk(self.state("최대 5일로 수정해줘"))
        assert result["small_talk"] is False and result["edit_request"] is True

    @pytest.mark.anyio
    async def test_no_key_means_no_edit_request(self, monkeypatch):
        monkeypatch.setattr(config, "jev_api_key", lambda: None)
        assert (await graph.judge_small_talk(self.state("연차 며칠이야?")))["edit_request"] is False
        assert (await graph.judge_small_talk(self.state()))["edit_request"] is True

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


class TestAnswerRetry:
    @pytest.mark.anyio
    async def test_malformed_answer_is_retried_once(self, monkeypatch):
        calls = []

        async def generate(question, context, history):
            calls.append(question)
            return '{"answer": "잘린' if len(calls) == 1 else '{"answer": "정상"}'

        monkeypatch.setattr(graph, "generate_answer", generate)
        result = await graph._answer_with_documents(TestGraph.state("질문"), CONTEXT, None)
        assert len(calls) == 2 and result["messages"][0].content == "정상"

    @pytest.mark.anyio
    async def test_second_failure_raises(self, monkeypatch):
        calls = []

        async def generate(question, context, history):
            calls.append(question)
            return '{"answer": "잘린'

        monkeypatch.setattr(graph, "generate_answer", generate)
        with pytest.raises(AiUnavailableError):
            await graph._answer_with_documents(TestGraph.state("질문"), CONTEXT, None)
        assert len(calls) == 2


def test_prompts_cover_counts_tables_and_quotes():
    assert "전체 수는 알 수 없고" in SYSTEM_PROMPT and '\\"' in SYSTEM_PROMPT
    assert "표는 그 행 전체" in EDIT_REQUEST_INSTRUCTION and "설명 문장" in EDIT_REQUEST_INSTRUCTION


class TestConflictScan:
    @pytest.mark.parametrize(
        "text",
        [
            "여기서 각 문서 간 같은 내용인데 다른 말로 기재된 충돌 발생 문서 있는지 찾아줘",
            "지금 발생한 문서 충돌 알려줘",
            "전체 문서에서 모순 찾아줘",
            "어떤 문서가 서로 다른 값인지 알려줘",
        ],
    )
    def test_overview_questions_match(self, text):
        assert graph._is_conflict_scan(text)

    @pytest.mark.parametrize("text", ["연차 이월 충돌이 뭐야?", "연차 며칠이야?", "문서 몇 개야?", "안녕"])
    def test_specific_questions_do_not_match(self, text):
        assert not graph._is_conflict_scan(text)

    def test_search_adds_terms_for_scan_only(self):
        scan = graph._search_query(TestGraph.state("지금 발생한 문서 충돌 알려줘"))
        assert "일치하지 않" in scan
        assert graph._search_query(TestGraph.state("연차 며칠이야?")) == "연차 며칠이야?"

    @pytest.mark.anyio
    async def test_answer_call_receives_scan_flag(self, monkeypatch):
        seen = {}

        async def generate(question, context, history, **kwargs):
            seen.update(kwargs)
            return json.dumps({"answer": "찾았습니다."}, ensure_ascii=False)

        monkeypatch.setattr(graph, "generate_answer", generate)
        await graph._answer_with_documents(TestGraph.state("지금 발생한 문서 충돌 알려줘"), CONTEXT, None)
        assert seen == {"conflict_scan": True}

    @pytest.mark.anyio
    async def test_scan_is_not_small_talk_and_does_not_reuse(self, monkeypatch):
        async def high(_question):
            return 0.99

        monkeypatch.setattr(config, "jev_api_key", lambda: "k")
        monkeypatch.setattr(graph, "small_talk_probability", high)
        monkeypatch.setattr(graph, "edit_request_probability", high)
        state = TestGraph.state("지금 발생한 문서 충돌 알려줘")
        assert (await graph.judge_small_talk(state))["small_talk"] is False
        assert (await graph.judge_context({**state, "context": CONTEXT}))["route"] == "SEARCH"
