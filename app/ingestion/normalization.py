import hashlib

from pydantic.alias_generators import to_camel

from app.ingestion.schemas import (
    ClassificationStatus,
    DocumentSubmission,
    Evidence,
    ExtractionCandidate,
    GroundedTopic,
    NormalizedDocument,
    Reference,
)


def locate_quote(text: str, quote: str) -> Evidence | None:
    start = text.find(quote)
    if start < 0 or not quote.strip():
        return None
    return Evidence(quote=quote, char_start=start, char_end=start + len(quote))


def normalize_document(
    document: DocumentSubmission, candidate: ExtractionCandidate | None, failure_reason: str | None = None
) -> NormalizedDocument:
    """근거 위치는 모델의 숫자를 믿지 않고 원문에서 직접 계산한다."""
    reasons = [failure_reason] if failure_reason else []
    sources = {to_camel(key): "metadata" for key in document.model_fields_set if key != "text"}
    sources["contentHash"] = "system"
    document_type = document.document_type
    classification_evidence = None
    if document_type is not None:
        classification_reason = "SOURCE_METADATA"
        sources["documentType"] = "metadata"
    elif candidate is not None and candidate.document_type is not None:
        classification_evidence = locate_quote(document.text, candidate.document_type.quote)
        if classification_evidence:
            document_type = candidate.document_type.value
            classification_reason = "GROUNDED_MODEL_CLASSIFICATION"
            sources["documentType"] = "llm"
        else:
            classification_reason = "INVALID_CLASSIFICATION_EVIDENCE"
            reasons.append(classification_reason)
            sources["documentType"] = "unresolved"
    else:
        classification_reason = failure_reason or "UNCLASSIFIED"
        sources["documentType"] = "unresolved"
        reasons.append(classification_reason)

    topics = []
    references = []
    seen_topics = set()
    seen_references = set()
    if candidate is not None:
        for topic in candidate.topics:
            evidence = locate_quote(document.text, topic.quote)
            key = topic.value.casefold()
            if evidence is None:
                reasons.append("INVALID_TOPIC_EVIDENCE")
            elif key not in seen_topics:
                topics.append(GroundedTopic(value=topic.value, evidence=evidence))
                seen_topics.add(key)
        for reference in candidate.references:
            evidence = locate_quote(document.text, reference.quote)
            key = reference.target_title.casefold()
            if evidence is None or reference.target_title not in reference.quote:
                reasons.append("INVALID_REFERENCE_EVIDENCE")
            elif key not in seen_references:
                references.append(Reference(target_title=reference.target_title, evidence=evidence))
                seen_references.add(key)

    sources["topics"] = "llm" if topics else "unresolved"
    sources["references"] = "llm" if references else "unresolved"
    return NormalizedDocument(
        **document.model_dump(exclude={"document_type"}),
        document_type=document_type,
        content_hash=hashlib.sha256(document.text.encode("utf-8")).hexdigest(),
        classification_status=(
            ClassificationStatus.CLASSIFIED if document_type is not None else ClassificationStatus.REVIEW_REQUIRED
        ),
        classification_reason=classification_reason,
        classification_evidence=classification_evidence,
        topics=topics,
        references=references,
        review_reasons=list(dict.fromkeys(reasons)),
        field_sources=sources,
    )
