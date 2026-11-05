from typing import Annotated
from uuid import UUID

from pydantic import Field, StringConstraints

from app.core.schemas import CamelModel


class IngestRequest(CamelModel):
    """docs199-Server의 IngestRequest. 문서 ID와 제목은 ai_app 권한으로 버전에서 읽는다."""

    job_id: UUID
    version_id: UUID
    storage_key: Annotated[str, StringConstraints(min_length=1, max_length=1024)]
    extension: str = Field(max_length=20)
