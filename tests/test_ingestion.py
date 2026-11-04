import asyncio
import json

import pytest
from pydantic import ValidationError
from qdrant_client import AsyncQdrantClient

from app.core.errors import AiUnavailableError
from app.ingestion import extraction, router, service
from app.ingestion.normalization import normalize_document
from app.ingestion.schemas import DocumentSubmission, ExtractionCandidate, IngestionResult
from app.retrieval.qdrant_store import QdrantVectorStore

TEXT = "출장 결과 보고서\n경비 처리 절차를 참조합니다."
INPUT = {
    "documentId": "doc-a", "versionId": "version-1", "versionNumber": 1,
    "title": "출장 결과 보고서", "text": TEXT, "status": "유효", "allowedGroupIds": ["staff"],
}
CANDIDATE = {
    "documentType": {"value": "REPORT", "quote": "출장 결과 보고서"},
    "topics": [{"value": "출장", "quote": "출장 결과 보고서"}],
    "references": [{"targetTitle": "경비 처리 절차", "quote": "경비 처리 절차를 참조합니다."}],
}


def submission(**overrides):
    return DocumentSubmission.model_validate({**INPUT, **overrides})


def candidate(**overrides):
    return ExtractionCandidate.model_validate({**CANDIDATE, **overrides})


@pytest.mark.parametrize("overrides", [
    {"text": "   "}, {"allowedGroupIds": []}, {"allowedGroupIds": [" "]},
    {"versionNumber": True}, {"versionNumber": 0}, {"unknown": "x"}, {"documentType": "규정"},
    {"sourceModifiedAt": "2026-11-05T12:00:00"}, {"text": "x" * 200_001},
])
def test_invalid_input_is_rejected(overrides):
    with pytest.raises(ValidationError):
        submission(**overrides)


def test_text_offsets_and_unknown_dates_are_preserved():
    doc = submission(text="  " + TEXT, allowedGroupIds=["staff", "staff"])
    normalized = normalize_document(doc, candidate())
    evidence = normalized.classification_evidence
    assert normalized.text[evidence.char_start:evidence.char_end] == evidence.quote
    assert evidence.char_start == 2
    assert normalized.allowed_group_ids == ["staff"] and normalized.source_modified_at is None
    assert normalized.classification_status == "CLASSIFIED"
    assert normalized.field_sources["documentType"] == "llm"
    assert normalized.field_sources["allowedGroupIds"] == "metadata"


def test_source_metadata_wins_over_model():
    normalized = normalize_document(submission(documentType="HANDOVER", author="원본 작성자"), candidate())
    assert normalized.document_type == "HANDOVER" and normalized.author == "원본 작성자"
    assert normalized.classification_evidence is None
    assert normalized.field_sources["documentType"] == "metadata"


def test_hallucinated_classification_does_not_become_etc():
    normalized = normalize_document(submission(), candidate(documentType={"value": "ETC", "quote": "없는 문장"}))
    assert normalized.document_type is None
    assert normalized.classification_status == "REVIEW_REQUIRED"
    assert "INVALID_CLASSIFICATION_EVIDENCE" in normalized.review_reasons


def test_hallucinated_topics_and_reference_quotes_are_discarded():
    normalized = normalize_document(submission(), candidate(
        topics=[{"value": "비밀", "quote": "없는 문장"}],
        references=[{"targetTitle": "없는 문서", "quote": "경비 처리 절차를 참조합니다."}],
    ))
    assert normalized.topics == [] and normalized.references == []
    assert set(normalized.review_reasons) == {"INVALID_TOPIC_EVIDENCE", "INVALID_REFERENCE_EVIDENCE"}


def test_duplicates_are_removed_and_unknown_type_is_pending():
    c = candidate(documentType=None, topics=CANDIDATE["topics"] * 2, references=CANDIDATE["references"] * 2)
    normalized = normalize_document(submission(), c)
    assert len(normalized.topics) == len(normalized.references) == 1
    assert normalized.document_type is None and normalized.classification_status == "REVIEW_REQUIRED"


def test_model_cannot_supply_permissions_or_document_ids():
    with pytest.raises(ValidationError):
        ExtractionCandidate.model_validate({**CANDIDATE, "allowedGroupIds": ["all"]})
    with pytest.raises(ValidationError):
        candidate(references=[{**CANDIDATE["references"][0], "targetDocumentId": "invented"}])


def test_extraction_uses_only_title_and_text_and_validates_json(monkeypatch):
    async def invoke(messages):
        data = json.loads(messages[1].content)
        assert data == {"title": INPUT["title"], "text": TEXT}
        assert "allowedGroupIds" not in messages[0].content
        return json.dumps(CANDIDATE, ensure_ascii=False)

    monkeypatch.setattr(extraction, "generate_structured_response", invoke)
    assert asyncio.run(extraction.extract_document(submission(author="작성자"))).document_type.value == "REPORT"


def test_extraction_limit_skips_llm_and_retains_searchable_text(monkeypatch):
    monkeypatch.setenv("EXTRACTION_MAX_TEXT_CHARS", "5")
    normalized = asyncio.run(service.prepare_document(submission()))
    assert normalized.text == TEXT and normalized.classification_status == "REVIEW_REQUIRED"
    assert normalized.review_reasons == ["EXTRACTION_LIMIT"]


def test_invalid_model_json_becomes_review_required(monkeypatch):
    async def invoke(messages):
        return '{"allowedGroupIds": ["all"]}'

    monkeypatch.setattr(extraction, "generate_structured_response", invoke)
    result = asyncio.run(service.prepare_document(submission()))
    assert result.classification_status == "REVIEW_REQUIRED"
    assert result.review_reasons == ["INVALID_EXTRACTION_SCHEMA"]


def test_model_failure_does_not_log_content(monkeypatch, caplog):
    async def fail(document):
        raise RuntimeError("비공개문서내용-금지")

    monkeypatch.setattr(service, "extract_document", fail)
    normalized = asyncio.run(service.prepare_document(submission()))
    assert normalized.review_reasons == ["EXTRACTION_UNAVAILABLE"]
    assert "비공개문서내용-금지" not in caplog.text


class FakeGraph:
    def __init__(self, matches=None, previous=None, fail=False):
        self.matches = matches or []
        self.previous = previous
        self.fail = fail
        self.events = []
        self.document = None

    async def find_reference_documents(self, title, source_document_id, groups):
        assert title == "경비 처리 절차" and groups == ["staff"]
        return self.matches

    async def replace_version(self, version_id, document_id, document, chunks):
        self.events.append("graph_write")
        self.document = document
        assert chunks == [] and document["index_status"] == "STAGING"
        if self.fail:
            raise RuntimeError("비공개-저장오류")
        return self.previous

    async def mark_ready(self, document_id, version_id, content_hash):
        self.events.append("graph_ready")


@pytest.mark.parametrize("matches, expected", [
    ([], "TARGET_UNRESOLVED"),
    ([{"document_id": "b", "version_id": "b1"}, {"document_id": "c", "version_id": "c1"}], "AMBIGUOUS_TARGET"),
])
def test_unresolved_or_ambiguous_references_stay_pending(matches, expected):
    normalized = normalize_document(submission(), candidate())
    asyncio.run(service.resolve_references(normalized, FakeGraph(matches)))
    assert normalized.references[0].target_document_id is None
    assert normalized.references[0].status == "REVIEW_REQUIRED"
    assert expected in normalized.review_reasons


def test_exact_accessible_reference_is_resolved():
    normalized = normalize_document(submission(), candidate())
    asyncio.run(service.resolve_references(normalized, FakeGraph([{"document_id": "b", "version_id": "b1"}])))
    assert normalized.references[0].target_document_id == "b"
    assert normalized.references[0].target_version_id == "b1"
    assert normalized.references[0].status == "VALIDATED"


@pytest.mark.parametrize("fail", [False, True])
def test_ingestion_visibility_and_metadata_with_real_qdrant_memory(monkeypatch, fail):
    async def embed(texts):
        return [[1.0, 0.0] for text in texts]

    monkeypatch.setattr(service, "embed_documents", embed)

    async def scenario():
        vector = QdrantVectorStore(AsyncQdrantClient(location=":memory:"), "test")
        graph = FakeGraph(fail=fail)
        normalized = normalize_document(submission(author="작성자"), None, "UNCLASSIFIED")
        try:
            if fail:
                with pytest.raises(AiUnavailableError, match="적재"):
                    await service.store_document(normalized, vector_store=vector, graph_store=graph)
                assert await vector.search([1.0, 0.0], 5, ["staff"]) == []
                assert graph.events == ["graph_write"]
            else:
                result = await service.store_document(normalized, vector_store=vector, graph_store=graph)
                assert result.classification_status == "REVIEW_REQUIRED" and result.chunk_count > 0
                hits = await vector.search([1.0, 0.0], 5, ["staff"])
                assert hits[0]["text"] == TEXT
                points, _ = await vector._client.scroll("test", with_payload=True)
                assert points[0].payload["author"] == "작성자"
                assert points[0].payload["classification_status"] == "REVIEW_REQUIRED"
                assert graph.document["classification_status"] == "REVIEW_REQUIRED"
                assert graph.events == ["graph_write", "graph_ready"]
        finally:
            await vector.close()

    asyncio.run(scenario())


def test_internal_api_validates_input_and_returns_summary(client, monkeypatch):
    async def ingest(document):
        return IngestionResult(document_id=document.document_id, version_id=document.version_id, chunk_count=1,
                               document_type=None, classification_status="REVIEW_REQUIRED",
                               review_reasons=["UNCLASSIFIED"], reference_count=0)

    monkeypatch.setattr(router, "ingest_document", ingest)
    response = client.post("/api/v1/ingestion/documents", json=INPUT)
    assert response.status_code == 200
    assert response.json()["classificationStatus"] == "REVIEW_REQUIRED"
    assert "text" not in response.json()
    invalid = client.post("/api/v1/ingestion/documents", json={**INPUT, "allowedGroupIds": []})
    assert invalid.status_code == 422 and TEXT not in invalid.text


def test_ingestion_endpoint_requires_internal_auth_in_production(client, monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    assert client.post("/api/v1/ingestion/documents", json=INPUT).status_code == 401


def test_internal_api_reports_version_conflicts_without_details(client, monkeypatch):
    async def conflict(document):
        raise router.DocumentVersionConflict("비공개-문서정보")

    monkeypatch.setattr(router, "ingest_document", conflict)
    response = client.post("/api/v1/ingestion/documents", json=INPUT)
    assert response.status_code == 409
    assert response.json()["code"] == "DOCUMENT_VERSION_CONFLICT"
    assert "비공개" not in response.text


def test_self_reference_stays_pending():
    normalized = normalize_document(submission(), candidate(
        references=[{"targetTitle": INPUT["title"], "quote": INPUT["title"]}],
    ))
    asyncio.run(service.resolve_references(normalized, FakeGraph()))
    assert normalized.references[0].reason == "SELF_REFERENCE"
    assert normalized.references[0].status == "REVIEW_REQUIRED"
