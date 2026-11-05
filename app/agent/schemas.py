from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, StrictBool, StringConstraints

from app.core.schemas import AccessScope, CamelModel, SearchFilters

# TODO: 질문 길이 제한은 서버 계약에서 확정 (docs199-Server#14)
MAX_MESSAGE_LENGTH = 2000


class AccessScopeInput(CamelModel):
    """Spring이 호출자의 AccessPolicy.scope로 만들어 보내는 권한 범위. 인증된 Spring 요청이므로 그대로 믿는다."""

    department_id: UUID
    position: str = Field(max_length=30)
    clearance: str = Field(max_length=30)
    is_admin: StrictBool

    def to_scope(self) -> AccessScope:
        return AccessScope(str(self.department_id), self.position, self.clearance, self.is_admin)


class KnownConflictInput(CamelModel):
    """호출자가 두 문서를 모두 볼 수 있는 등록 충돌. Spring이 걸러 보낸다 (docs199-Server#101)."""

    relation_id: UUID
    document_id: UUID
    other_document_id: UUID
    item: str


class ChatRequest(CamelModel):
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_MESSAGE_LENGTH)]
    thread_id: str | None = Field(default=None, max_length=128)
    filters: SearchFilters | None = None
    access_scope: AccessScopeInput | None = None
    known_conflicts: list[KnownConflictInput] | None = None


class SourceChunk(CamelModel):
    id: str
    text: str
    source: str | None = None
    document_id: str | None = None
    version_id: str | None = None
    char_start: int | None = None
    char_end: int | None = None


class EvidenceQuote(CamelModel):
    # TODO: 응답 필드 이름은 서버 계약에서 확정 (docs199-Server#14)
    chunk_id: str
    source: str | None = None
    document_id: str | None = None
    version_id: str | None = None
    quote: str
    char_start: int | None = None
    char_end: int | None = None


class KeywordItem(CamelModel):
    # TODO: 형식은 서버 계약에서 확정 (docs199-Server#78)
    chunk_id: str
    text: str


class EditProposalItem(CamelModel):
    # TODO: 형식은 서버 계약에서 확정. 적용은 사용자가 고른다
    chunk_id: str
    source: str | None = None
    document_id: str | None = None
    version_id: str | None = None
    before: str
    after: str
    reason: str
    char_start: int | None = None
    char_end: int | None = None


class ChatResponse(CamelModel):
    thread_id: str
    answer: str
    # TODO: 이름과 값은 서버 계약에서 확정 (docs199-Server#76)
    answer_type: Literal["ANSWER", "NO_EVIDENCE", "CLARIFICATION", "SMALL_TALK"]
    route: Literal["CONTEXT", "SEARCH"]
    context_probability: float | None
    sources: list[SourceChunk]
    evidence: list[EvidenceQuote]
    keywords: list[KeywordItem]
    edit_proposals: list[EditProposalItem] = []
    failed_sources: list[str]
    fallback_reason: str | None
