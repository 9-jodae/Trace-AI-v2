from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from app.core.schemas import CamelModel, SearchFilters

# TODO: 질문 길이 제한은 서버 계약에서 확정 (docs199-Server#14)
MAX_MESSAGE_LENGTH = 2000


class ChatRequest(CamelModel):
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_MESSAGE_LENGTH)]
    thread_id: str | None = Field(default=None, max_length=128)
    filters: SearchFilters | None = None


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


class ChatResponse(CamelModel):
    thread_id: str
    answer: str
    route: Literal["CONTEXT", "SEARCH"]
    context_probability: float | None
    sources: list[SourceChunk]
    evidence: list[EvidenceQuote]
    failed_sources: list[str]
    fallback_reason: str | None
