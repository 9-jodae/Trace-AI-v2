import json

import pytest
from conftest import chunk
from langchain_core.messages import HumanMessage

from app.agent import graph
from app.agent.evidence import NO_GROUNDED_EDITS
from app.llm.client import EDIT_REQUEST_INSTRUCTION, SYSTEM_PROMPT

BEFORE = "이월은 최대 7일이다."
EDIT_CHUNK = {**chunk("c1", f"앞 문장. {BEFORE} 뒤 문장."), "source": "연차 지침", "document_id": "d1"}
OTHERS = [chunk(f"n{number}", f"무관한 문서 {number}") for number in range(1, 8)]


def state(text="5일로 수정해줘", **values):
    return {"messages": [HumanMessage(content=text)], **values}


@pytest.fixture
def respond(monkeypatch):
    def set_response(payload):
        async def generate(question, context, history, **kwargs):
            return json.dumps(payload, ensure_ascii=False)

        monkeypatch.setattr(graph, "generate_answer", generate)

    return set_response


class TestPreviousContext:
    @pytest.fixture
    def searched(self, monkeypatch):
        async def search(query, filters, scope):
            return {"chunks": [chunk("s1", "새 검색")], "failed_sources": []}

        monkeypatch.setattr(graph, "search", search)

    @pytest.fixture
    def accessible(self, monkeypatch):
        calls = []

        async def check(chunks, scope):
            calls.append(chunks)
            return [item for item in chunks if item["id"] != "gone"]

        monkeypatch.setattr(graph, "filter_accessible_context", check)
        return calls

    @pytest.mark.anyio
    async def test_edit_request_keeps_visible_previous_chunks_first(self, searched, accessible):
        previous = [EDIT_CHUNK, chunk("gone", "삭제됨"), chunk("s1", "새 검색")]
        result = await graph.retrieve(state(edit_request=True, context=previous, filters=None))
        assert [item["id"] for item in result["context"]] == ["c1", "s1"]

    @pytest.mark.anyio
    async def test_other_questions_and_filters_use_new_search_only(self, searched, accessible):
        ordinary = await graph.retrieve(state("연차 며칠?", context=[EDIT_CHUNK]))
        filtered = await graph.retrieve(state(edit_request=True, context=[EDIT_CHUNK], filters={"x": 1}))
        assert [item["id"] for item in ordinary["context"]] == ["s1"]
        assert [item["id"] for item in filtered["context"]] == ["s1"] and accessible == []

    @pytest.mark.anyio
    async def test_revalidation_failure_falls_back_to_search(self, searched, monkeypatch):
        async def broken(chunks, scope):
            raise RuntimeError("db down")

        monkeypatch.setattr(graph, "filter_accessible_context", broken)
        result = await graph.retrieve(state(edit_request=True, context=[EDIT_CHUNK]))
        assert [item["id"] for item in result["context"]] == ["s1"]


class TestSources:
    @pytest.mark.anyio
    async def test_limits_sources_and_keeps_cited_chunks_first(self, respond):
        respond({"answer": "답.", "evidence": [{"chunk": 8, "quote": "무관한 문서 7"}]})
        result = await graph._answer_with_documents(state("질문"), [EDIT_CHUNK, *OTHERS], None)
        ids = [item["id"] for item in result["context"]]
        assert len(ids) == graph.SOURCE_LIMIT and ids[0] == "n7"


class TestInlineProposals:
    @pytest.mark.anyio
    async def test_grounded_edits_are_appended_to_answer(self, respond):
        respond(
            {
                "answer": "취업규칙에 맞춥니다.",
                "edits": [{"chunk": 1, "before": BEFORE, "after": "이월은 최대 5일이다.", "reason": "맞춤"}],
            }
        )
        result = await graph._answer_with_documents(state(edit_request=True), [EDIT_CHUNK], None)
        answer = result["messages"][0].content
        assert answer.startswith("취업규칙에 맞춥니다.") and "아직 바뀌지 않았습니다" in answer
        assert f"현재: {BEFORE}" in answer and "수정안: 이월은 최대 5일이다." in answer
        assert result["context"][0]["id"] == "c1"

    @pytest.mark.anyio
    async def test_ungrounded_edits_say_so(self, respond):
        respond(
            {"answer": "제안합니다.", "edits": [{"chunk": 1, "before": "없는 문장", "after": "새 문장", "reason": "r"}]}
        )
        result = await graph._answer_with_documents(state(edit_request=True), [EDIT_CHUNK], None)
        assert result["messages"][0].content.endswith(NO_GROUNDED_EDITS) and result["edit_proposals"] == []

    @pytest.mark.anyio
    async def test_ordinary_answer_is_unchanged(self, respond):
        respond({"answer": "답."})
        result = await graph._answer_with_documents(state("연차 며칠?"), [EDIT_CHUNK], None)
        assert result["messages"][0].content == "답."


def test_prompts_do_not_claim_absence_or_point_to_cards():
    assert "문서에 없다고 단정하지 않는다" in SYSTEM_PROMPT
    assert "고르라고" not in EDIT_REQUEST_INSTRUCTION and "단정하지 않는다" in EDIT_REQUEST_INSTRUCTION
