from typing import NamedTuple

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    """Spring(Jackson) 기본 규칙에 맞춰 JSON 필드를 camelCase로 주고받는다."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ErrorResponse(CamelModel):
    code: str
    message: str


class AccessScope(NamedTuple):
    """Spring의 AccessScope. 순서는 서버 권한 함수의 인자 순서와 같다."""

    department_id: str | None
    position: str | None
    clearance: str | None
    is_admin: bool
