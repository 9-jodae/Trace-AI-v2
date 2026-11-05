from typing import Literal

from typing_extensions import NotRequired, TypedDict

EntityType = Literal["PROJECT", "PROCESS", "SYSTEM_SERVICE", "ORG_UNIT", "POLICY", "CONCEPT"]
RelationPredicate = Literal["RESPONSIBLE_FOR", "USES", "GOVERNED_BY"]
AssertionPolarity = Literal["AFFIRMED", "NEGATED", "UNKNOWN"]
AssertionModality = Literal["ACTUAL", "PLANNED", "CONDITIONAL", "UNKNOWN"]


class Chunk(TypedDict):
    id: str
    text: str
    source: str | None
    document_id: NotRequired[str | None]
    version_id: NotRequired[str | None]
    char_start: NotRequired[int | None]
    char_end: NotRequired[int | None]
    chunk_index: NotRequired[int]
    content_hash: NotRequired[str]


class DocumentInput(TypedDict):
    document_id: str
    version_id: str
    text: str
    source: str


class Mention(TypedDict):
    """원문에서 검증된 한 Entity 표기와 출처 위치."""

    mention_id: str
    entity_id: str
    entity_type: EntityType
    canonical_name: str
    surface_form: str
    chunk_id: str
    char_start: int
    char_end: int
    evidence_start: int
    evidence_end: int
    evidence_quote: str


class RelationAssertion(TypedDict):
    """문서 한 버전이 말하는 관계. 근거가 없거나 모호하면 적재하지 않는다."""

    assertion_id: str
    subject_entity_id: str
    predicate: RelationPredicate
    object_entity_id: str
    source_document_id: str
    source_version_id: str
    evidence_chunk_id: str
    evidence_start: int
    evidence_end: int
    evidence_quote: str
    polarity: AssertionPolarity
    modality: AssertionModality
    condition_quote: str | None
