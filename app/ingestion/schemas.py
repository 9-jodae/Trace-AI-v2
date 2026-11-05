from uuid import UUID

from pydantic import Field

from app.core.schemas import CamelModel


class IngestRequest(CamelModel):
    job_id: UUID
    version_id: UUID
    storage_key: str = Field(min_length=1, max_length=1024)
    extension: str = Field(min_length=1, max_length=20)
