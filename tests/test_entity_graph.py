import json

import pytest

from app.retrieval import entity_extraction
from app.retrieval.relations import entity_id, mention_is_grounded, validate_assertion


def make_chunk():
    return {
        "id": "chunk-1", "document_id": "doc-1", "version_id": "version-1",
        "text": "재무팀은 경비 정산을 담당한다.", "source": "운영 규정",
        "char_start": 100, "char_end": 117, "chunk_index": 0,
    }


def test_entity_identity_is_stable_and_scoped_by_namespace():
    assert entity_id("ORG_UNIT", "company-a", "재무팀") == entity_id("ORG_UNIT", "company-a", " 재무팀 ")
    assert entity_id("ORG_UNIT", "company-a", "재무팀") != entity_id("ORG_UNIT", "company-b", "재무팀")
    assert entity_id("ORG_UNIT", "company-a", "재무팀") != entity_id("PROCESS", "company-a", "재무팀")


def test_mention_requires_exact_name_and_quote_within_chunk_offsets():
    chunk = make_chunk()
    mention = {
        "char_start": 100, "char_end": 103, "evidence_start": 100, "evidence_end": 117,
        "surface_form": "재무팀", "evidence_quote": chunk["text"],
    }
    assert mention_is_grounded(mention, chunk)
    assert not mention_is_grounded({**mention, "surface_form": "인사팀"}, chunk)
    assert not mention_is_grounded({**mention, "evidence_start": 99}, chunk)


def test_assertion_requires_affirmed_actual_exact_versioned_evidence():
    chunk = make_chunk()
    assertion = {
        "assertion_id": "assertion-1", "subject_entity_id": "entity-1", "predicate": "RESPONSIBLE_FOR",
        "object_entity_id": "entity-2", "source_document_id": "doc-1", "source_version_id": "version-1",
        "evidence_chunk_id": "chunk-1", "evidence_start": 100, "evidence_end": 117,
        "evidence_quote": chunk["text"], "polarity": "AFFIRMED", "modality": "ACTUAL",
        "condition_quote": None,
    }
    assert validate_assertion(assertion, {"chunk-1": chunk})
    assert not validate_assertion({**assertion, "polarity": "NEGATED"}, {"chunk-1": chunk})
    assert not validate_assertion({**assertion, "modality": "PLANNED"}, {"chunk-1": chunk})
    assert not validate_assertion({**assertion, "source_version_id": "old-version"}, {"chunk-1": chunk})
    assert not validate_assertion({**assertion, "evidence_quote": "다른 문장"}, {"chunk-1": chunk})


@pytest.mark.anyio
async def test_extraction_discards_ungrounded_mentions_and_non_actual_relations(monkeypatch):
    chunk = make_chunk()
    response = {
        "mentions": [
            {
                "entity_type": "ORG_UNIT", "canonical_name": "재무팀",
                "surface_form": "재무팀", "quote": chunk["text"],
            },
            {
                "entity_type": "PROCESS", "canonical_name": "경비 정산",
                "surface_form": "경비 정산", "quote": chunk["text"],
            },
            {
                "entity_type": "ORG_UNIT", "canonical_name": "인사팀",
                "surface_form": "인사팀", "quote": chunk["text"],
            },
        ],
        "assertions": [
            {
                "subject_name": "재무팀", "subject_type": "ORG_UNIT", "predicate": "RESPONSIBLE_FOR",
                "object_name": "경비 정산", "object_type": "PROCESS", "quote": chunk["text"],
                "polarity": "AFFIRMED", "modality": "ACTUAL", "condition_quote": None,
            },
            {
                "subject_name": "재무팀", "subject_type": "ORG_UNIT", "predicate": "RESPONSIBLE_FOR",
                "object_name": "경비 정산", "object_type": "PROCESS", "quote": chunk["text"],
                "polarity": "AFFIRMED", "modality": "PLANNED", "condition_quote": None,
            },
        ],
    }

    async def fake_response(_messages):
        return json.dumps(response, ensure_ascii=False)

    monkeypatch.setattr(entity_extraction, "generate_structured_response", fake_response)
    mentions, assertions = await entity_extraction.extract_entity_graph("doc-1", "version-1", [chunk])
    assert {mention["surface_form"] for mention in mentions} == {"재무팀", "경비 정산"}
    assert len(assertions) == 1
    assert assertions[0]["predicate"] == "RESPONSIBLE_FOR"


@pytest.mark.anyio
@pytest.mark.parametrize("raw", ["설명만 있음", '{"mentions": [', '{"mentions": [{"entity_type": "없는값"}]}'])
async def test_bad_response_skips_chunk(monkeypatch, raw):
    async def fake_response(_messages):
        return raw

    monkeypatch.setattr(entity_extraction, "generate_structured_response", fake_response)
    assert await entity_extraction.extract_entity_graph("doc-1", "version-1", [make_chunk()]) == ([], [])


@pytest.mark.anyio
async def test_code_block_response_is_parsed(monkeypatch):
    chunk = make_chunk()
    mention = {"entity_type": "ORG_UNIT", "canonical_name": "재무팀", "surface_form": "재무팀", "quote": chunk["text"]}

    async def fake_response(_messages):
        return "```json\n" + json.dumps({"mentions": [mention]}, ensure_ascii=False) + "\n```"

    monkeypatch.setattr(entity_extraction, "generate_structured_response", fake_response)
    mentions, _ = await entity_extraction.extract_entity_graph("doc-1", "version-1", [chunk])
    assert [item["surface_form"] for item in mentions] == ["재무팀"]


@pytest.mark.anyio
async def test_unverified_condition_quote_is_dropped(monkeypatch):
    chunk = make_chunk()
    names = [("재무팀", "ORG_UNIT"), ("경비 정산", "PROCESS")]
    mentions = [
        {"entity_type": kind, "canonical_name": name, "surface_form": name, "quote": chunk["text"]}
        for name, kind in names
    ]
    assertion = {
        "subject_name": "재무팀", "subject_type": "ORG_UNIT", "predicate": "RESPONSIBLE_FOR",
        "object_name": "경비 정산", "object_type": "PROCESS", "quote": chunk["text"],
        "polarity": "AFFIRMED", "modality": "ACTUAL", "condition_quote": "원문에 없는 조건",
    }

    async def fake_response(_messages):
        return json.dumps({"mentions": mentions, "assertions": [assertion]}, ensure_ascii=False)

    monkeypatch.setattr(entity_extraction, "generate_structured_response", fake_response)
    _, assertions = await entity_extraction.extract_entity_graph("doc-1", "version-1", [chunk])
    assert assertions == []


def test_replace_removes_assertions_of_every_version():
    from app.retrieval.neo4j_store import REPLACE_ENTITY_GRAPH_CYPHER

    assert "(old:RelationAssertion)" in REPLACE_ENTITY_GRAPH_CYPHER

