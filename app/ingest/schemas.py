from typing import Annotated
from uuid import UUID

from pydantic import Field, StringConstraints

from app.core.schemas import CamelModel


class ConflictToCheck(CamelModel):
    """Spring이 넘기는 이 문서의 충돌. 형식은 docs199-Server#93 합의 전 제안이다."""

    relation_id: UUID
    item: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    other_document_id: UUID


class IngestRequest(CamelModel):
    """docs199-Server의 IngestRequest. 문서 ID와 제목은 ai_app 권한으로 버전에서 읽는다."""

    job_id: UUID
    version_id: UUID
    storage_key: Annotated[str, StringConstraints(min_length=1, max_length=1024)]
    extension: str = Field(max_length=20)
    conflicts: list[ConflictToCheck] = Field(default_factory=list, max_length=20)
