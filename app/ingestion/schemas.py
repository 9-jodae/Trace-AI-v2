from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, StringConstraints, field_validator

from app.core.schemas import CamelModel

Name = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=300)]
Quote = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=2000)]


class StrictModel(CamelModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class DocumentType(StrEnum):
    HANDOVER = "HANDOVER"
    APPROVAL = "APPROVAL"
    REPORT = "REPORT"
    PRESENTATION = "PRESENTATION"
    ETC = "ETC"


class ClassificationStatus(StrEnum):
    PENDING = "PENDING"
    CLASSIFIED = "CLASSIFIED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class DocumentSubmission(StrictModel):
    """AI 내부 입력. 권한과 문서 식별 정보는 신뢰할 수 있는 호출자가 제공한다."""

    document_id: Name
    version_id: Name
    version_number: int = Field(ge=1, strict=True)
    title: Name
    text: str = Field(min_length=1, max_length=200_000, strict=True)
    status: Literal["유효", "폐기"]
    allowed_group_ids: list[Name] = Field(min_length=1, max_length=1000)
    source_uri: str | None = Field(default=None, max_length=2000)
    author: Name | None = None
    authored_at: date | None = None
    source_modified_at: datetime | None = None
    department: Name | None = None
    document_type: DocumentType | None = None

    @field_validator("text")
    @classmethod
    def nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("원문이 비어 있습니다.")
        # 원문을 strip/정규화하면 근거 문자 위치가 달라진다.
        return value

    @field_validator("allowed_group_ids")
    @classmethod
    def unique_groups(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))

    @field_validator("source_modified_at")
    @classmethod
    def timezone_required(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.utcoffset() is None:
            raise ValueError("수정 시각에는 시간대가 필요합니다.")
        return value


class ClassificationCandidate(StrictModel):
    value: DocumentType
    quote: Quote


class TopicCandidate(StrictModel):
    value: Name
    quote: Quote


class ReferenceCandidate(StrictModel):
    target_title: Name
    quote: Quote


class ExtractionCandidate(StrictModel):
    """LLM이 권한·수정자 등을 덮어쓰거나 임의의 문서 ID를 만들어낼 수 없게 제한한다."""

    document_type: ClassificationCandidate | None = None
    topics: list[TopicCandidate] = Field(default_factory=list, max_length=20)
    references: list[ReferenceCandidate] = Field(default_factory=list, max_length=30)


class Evidence(StrictModel):
    quote: Quote
    char_start: int = Field(ge=0, strict=True)
    char_end: int = Field(gt=0, strict=True)


class GroundedTopic(StrictModel):
    value: Name
    evidence: Evidence


class Reference(StrictModel):
    target_title: Name
    evidence: Evidence
    target_document_id: Name | None = None
    target_version_id: Name | None = None
    status: Literal["VALIDATED", "REVIEW_REQUIRED"] = "REVIEW_REQUIRED"
    reason: str = "TARGET_UNRESOLVED"


class NormalizedDocument(DocumentSubmission):
    schema_version: Literal["1.0"] = "1.0"
    content_hash: str
    classification_status: ClassificationStatus
    classification_reason: str
    classification_evidence: Evidence | None = None
    topics: list[GroundedTopic] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)
    review_reasons: list[str] = Field(default_factory=list)
    field_sources: dict[str, Literal["metadata", "system", "llm", "unresolved"]]


class IngestionResult(StrictModel):
    document_id: str
    version_id: str
    chunk_count: int
    document_type: DocumentType | None
    classification_status: ClassificationStatus
    review_reasons: list[str]
    reference_count: int
