import pytest
from langchain_core.messages import HumanMessage

from app.agent import graph
from app.agent.graph import CONTEXT_REUSE_THRESHOLD, judge_context, select_route
from app.core.schemas import SearchFilters

CONTEXT = [{"id": "doc-1", "text": "본문", "source": "문서"}]


@pytest.fixture(autouse=True)
def accessible(monkeypatch: pytest.MonkeyPatch):
    async def keep_all(chunks: list) -> list:
        return chunks

    monkeypatch.setattr(graph, "filter_accessible_context", keep_all)


def _state(context: list) -> dict:
    return {"messages": [HumanMessage(content="질문")], "context": context}


def _use_probability(monkeypatch: pytest.MonkeyPatch, probability: float) -> None:
    async def fake(*_args) -> float:
        return probability

    monkeypatch.setattr(graph.config, "jev_api_key", lambda: "test-key")
    monkeypatch.setattr(graph, "can_answer_probability", fake)


@pytest.mark.anyio
async def test_no_context_goes_to_search_without_judging():
    decision = await judge_context(_state([]))

    assert decision["route"] == "SEARCH"
    assert decision["fallback_reason"] is None


@pytest.mark.anyio
async def test_missing_api_key_falls_back_to_search(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(graph.config, "jev_api_key", lambda: None)

    decision = await judge_context(_state(CONTEXT))

    assert decision["route"] == "SEARCH"
    assert decision["fallback_reason"] is not None


@pytest.mark.anyio
async def test_threshold_reuses_context(monkeypatch: pytest.MonkeyPatch):
    _use_probability(monkeypatch, CONTEXT_REUSE_THRESHOLD)

    decision = await judge_context(_state(CONTEXT))

    assert decision["route"] == "CONTEXT"
    assert select_route(decision) == "generate"


@pytest.mark.anyio
async def test_below_threshold_searches_again(monkeypatch: pytest.MonkeyPatch):
    _use_probability(monkeypatch, CONTEXT_REUSE_THRESHOLD - 0.01)

    decision = await judge_context(_state(CONTEXT))

    assert decision["route"] == "SEARCH"
    assert select_route(decision) == "retrieve"


@pytest.mark.anyio
async def test_jev_failure_falls_back_to_search(monkeypatch: pytest.MonkeyPatch):
    async def fail(*_args) -> float:
        raise TimeoutError

    monkeypatch.setattr(graph.config, "jev_api_key", lambda: "test-key")
    monkeypatch.setattr(graph, "can_answer_probability", fail)

    decision = await judge_context(_state(CONTEXT))

    assert decision["route"] == "SEARCH"
    assert "TimeoutError" in decision["fallback_reason"]


class TestContextRevalidation:
    @pytest.fixture(autouse=True)
    def jev(self, monkeypatch: pytest.MonkeyPatch):
        self.sent = []

        async def fake(_question, context, _history) -> float:
            self.sent.append(context)
            return 0.9

        monkeypatch.setattr(graph.config, "jev_api_key", lambda: "test-key")
        monkeypatch.setattr(graph, "can_answer_probability", fake)

    @pytest.mark.anyio
    async def test_only_accessible_chunks_reach_jev_and_answer(self, monkeypatch: pytest.MonkeyPatch):
        other = {"id": "doc-2", "text": "기밀", "source": "문서"}

        async def only_first(chunks: list) -> list:
            return chunks[:1]

        monkeypatch.setattr(graph, "filter_accessible_context", only_first)
        decision = await judge_context(_state([*CONTEXT, other]))
        assert self.sent == [CONTEXT]
        assert decision["route"] == "CONTEXT" and decision["context"] == CONTEXT

    @pytest.mark.anyio
    async def test_no_accessible_chunk_searches_without_jev(self, monkeypatch: pytest.MonkeyPatch):
        async def none(chunks: list) -> list:
            return []

        monkeypatch.setattr(graph, "filter_accessible_context", none)
        decision = await judge_context(_state(CONTEXT))
        assert self.sent == []
        assert decision["route"] == "SEARCH" and decision["context"] == [] and decision["fallback_reason"] is None

    @pytest.mark.anyio
    async def test_revalidation_failure_drops_context_without_jev(self, monkeypatch: pytest.MonkeyPatch):
        async def fail(chunks: list) -> list:
            raise RuntimeError("db")

        monkeypatch.setattr(graph, "filter_accessible_context", fail)
        decision = await judge_context(_state(CONTEXT))
        assert self.sent == []
        assert decision["route"] == "SEARCH" and decision["context"] == []
        assert "RuntimeError" in decision["fallback_reason"]


@pytest.mark.anyio
async def test_filters_skip_context_reuse_without_jev(monkeypatch: pytest.MonkeyPatch):
    async def forbidden(*_args) -> float:
        raise AssertionError("JEV를 호출하면 안 됩니다")

    monkeypatch.setattr(graph, "can_answer_probability", forbidden)
    decision = await judge_context({**_state(CONTEXT), "filters": SearchFilters(extensions=["pdf"])})
    assert decision["route"] == "SEARCH" and decision["fallback_reason"] is None
