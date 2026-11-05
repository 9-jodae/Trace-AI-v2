import json

import pytest
from conftest import chunk
from langchain_core.messages import HumanMessage

from app.agent import graph
from app.core import config
from app.llm.client import build_prompt

CONTEXT = [{**chunk("c1", "앞 문장. 이월은 최대 7일이다. 뒤 문장."), "source": "연차 지침", "document_id": "d1"}]


@pytest.fixture(autouse=True)
def proposals_disabled(monkeypatch):
    monkeypatch.delenv("EDIT_PROPOSALS_ENABLED", raising=False)


@pytest.mark.anyio
async def test_edit_request_gets_no_proposals(monkeypatch):
    seen = {}

    async def generate(question, context, history, **extra):
        seen.update(extra)
        payload = {
            "answer": "문서 수정은 지원하지 않습니다.",
            "edits": [{"chunk": 1, "before": "이월은 최대 7일이다.", "after": "5일", "reason": "r"}],
        }
        return json.dumps(payload, ensure_ascii=False)

    monkeypatch.setattr(graph, "generate_answer", generate)
    state = {"messages": [HumanMessage(content="5일로 수정해줘")], "edit_request": True}
    result = await graph._answer_with_documents(state, CONTEXT, None)
    assert seen == {"edit_unsupported": True}
    assert result["edit_proposals"] == [] and result["messages"][0].content == "문서 수정은 지원하지 않습니다."


@pytest.mark.anyio
async def test_judge_skips_edit_jev_call(monkeypatch):
    monkeypatch.setattr(config, "jev_api_key", lambda: "k")
    calls = []

    async def small(_question):
        return 0.0

    async def editing(_question):
        calls.append(_question)
        return 0.9

    monkeypatch.setattr(graph, "small_talk_probability", small)
    monkeypatch.setattr(graph, "edit_request_probability", editing)
    direct = await graph.judge_small_talk({"messages": [HumanMessage(content="5일로 수정해줘")]})
    vague = await graph.judge_small_talk({"messages": [HumanMessage(content="이 내용 좀 정리해 볼까")]})
    assert calls == [] and direct["edit_request"] is True and vague["edit_request"] is False


def test_unsupported_prompt_says_editing_is_not_supported():
    prompt = build_prompt("5일로 수정해줘", CONTEXT, edit_unsupported=True)
    assert "지원하지 않는다" in prompt and "edits" not in prompt
