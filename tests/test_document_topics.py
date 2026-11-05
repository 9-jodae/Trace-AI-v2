import asyncio

import pytest

from app.core.errors import AiUnavailableError
from app.retrieval import document_topics
from app.retrieval.document_topics import parse_topics, shared_topic_edges

DOCUMENT = {"document_id": "d1", "version_id": "v1", "source": "전공 정리", "text": "가" * 3000}


@pytest.mark.parametrize(
    "reply, expected",
    [
        ('{"topics": ["공부", "전공"]}', ["공부", "전공"]),
        ('```json\n{"topics": ["공부"]}\n```', ["공부"]),
        ('{"topics": ["가", "나", "다", "라"]}', ["가", "나", "다"]),
        ('{"topics": ["공부", " 공부 ", "", 1, "길" ]}', ["공부", "길"]),
        (f'{{"topics": ["{"가" * 21}"]}}', []),
        ('{"topics": "공부"}', []),
        ('{"other": []}', None),
        ("설명만 있음", None),
        ('{"topics": [', None),
    ],
)
def test_parse_topics(reply, expected):
    assert parse_topics(reply) == expected


def test_sends_only_the_beginning_and_existing_topics(monkeypatch):
    sent = []

    async def respond(messages):
        sent.append(messages[-1].content)
        return '{"topics": ["공부"]}'

    monkeypatch.setattr(document_topics, "generate_structured_response", respond)
    assert asyncio.run(document_topics.extract_document_topics(DOCUMENT, ["전공"])) == ["공부"]
    assert sent[0].count("가") == document_topics.MAX_INPUT_CHARACTERS and "전공" in sent[0]


def test_llm_failure_returns_none(monkeypatch):
    async def fail(messages):
        raise AiUnavailableError("x")

    monkeypatch.setattr(document_topics, "generate_structured_response", fail)
    assert asyncio.run(document_topics.extract_document_topics(DOCUMENT, [])) is None


def test_edges_connect_documents_sharing_topics():
    edges = shared_topic_edges({"b": ["전공", "공부"], "a": ["공부", "전공"], "c": ["보안"], "d": []})
    assert edges == [{"source": "a", "target": "b", "label": "공부 · 전공", "topics": ["공부", "전공"]}]
