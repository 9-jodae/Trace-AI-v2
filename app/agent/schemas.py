from typing import Literal

from pydantic import Field

from app.core.schemas import CamelModel


class ChatRequest(CamelModel):
    message: str = Field(min_length=1)
    thread_id: str | None = None


class SourceChunk(CamelModel):
    id: str
    text: str
    source: str


class ChatResponse(CamelModel):
    thread_id: str
    answer: str
    route: Literal["CONTEXT", "SEARCH"]
    context_probability: float | None
    sources: list[SourceChunk]
    failed_sources: list[str]
    fallback_reason: str | None
