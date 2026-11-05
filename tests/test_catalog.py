import pytest
from langchain_core.messages import HumanMessage

from app.agent import graph
from app.agent.catalog import describe_catalog, is_catalog_question
from app.retrieval import service

DOCS = [
    {"id": "d1", "title": "leave-policy", "document_type": "HANDOVER", "chunk_count": 3},
    {"id": "d2", "title": "sales-guide", "document_type": "REPORT", "chunk_count": 2},
    {"id": "d3", "title": "travel-policy", "document_type": None, "chunk_count": 1},
]


@pytest.mark.parametrize(
    "text",
    [
        "검색 가능한 문서 뭐뭐 있어?",
        "AI에 등록된 문서 개수 알려줘",
        "청크 몇 개야?",
        "문서 몇 개 있어?",
        "볼 수 있는 문서 목록 보여줘",
        "전체 문서는 몇 개인가요?",
        "지금 등록 문서 수가 어떻게 됨?",
        "지금 등록된 문서 수가 얼마나 됨?",
        "청크 수는?",
    ],
)
def test_catalog_questions_match(text):
    assert is_catalog_question(text)


@pytest.mark.parametrize(
    "text",
    [
        "연차 관련 문서 뭐 있어?",
        "출장 문서 몇 개야?",
        "문서 간 충돌 찾아줘",
        "연차 며칠이야?",
        "안녕",
        "문서 어떻게 써?",
        "새결이 뭘까요?",
    ],
)
def test_topic_or_other_questions_do_not_match(text):
    assert not is_catalog_question(text)


def test_describe_counts_documents_chunks_and_types():
    answer = describe_catalog("문서 몇 개야?", DOCS, filtered=False)
    assert "**3개**" in answer and "**6개**" in answer
    assert "인수인계 1개" in answer and "미분류 1개" in answer
    assert "leave-policy" not in answer


def test_describe_lists_titles_and_filter_scope():
    answer = describe_catalog("문서 목록 보여줘", DOCS, filtered=True)
    assert answer.startswith("질문에 건 필터 안에서") and "leave-policy, sales-guide, travel-policy" in answer
    assert describe_catalog("문서 몇 개야?", [], filtered=False) == "지금 볼 수 있는 문서가 없습니다."


@pytest.mark.anyio
async def test_judge_routes_catalog_without_jev(monkeypatch):
    async def fail(_question):
        raise AssertionError("JEV를 부르면 안 된다")

    monkeypatch.setattr(graph, "small_talk_probability", fail)
    monkeypatch.setattr(graph, "edit_request_probability", fail)
    monkeypatch.setattr(graph.config, "jev_api_key", lambda: "k")
    result = await graph.judge_small_talk({"messages": [HumanMessage(content="문서 몇 개야?")]})
    assert result["catalog"] is True and graph.select_small_talk({**result, "small_talk": False}) == "catalog"


@pytest.mark.anyio
async def test_catalog_node_answers_from_store(monkeypatch):
    seen = {}

    async def documents(filters, scope):
        seen["args"] = (filters, scope)
        return DOCS

    monkeypatch.setattr(graph, "list_accessible_documents", documents)
    result = await graph.catalog(
        {"messages": [HumanMessage(content="문서 몇 개야?")], "filters": None, "access_scope": "s"}
    )
    assert "**3개**" in result["messages"][0].content
    assert result["context"] == [] and result["answer_type"] == "ANSWER" and seen["args"] == (None, "s")


@pytest.mark.anyio
async def test_service_skips_without_scope(monkeypatch):
    monkeypatch.setattr(service.config, "access_scope", lambda: None)
    monkeypatch.setattr(service.config, "use_sample_data", lambda: False)
    assert await service.list_accessible_documents() == []
