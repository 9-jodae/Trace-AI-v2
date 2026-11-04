from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    """Spring(Jackson) 기본 규칙에 맞춰 JSON 필드를 camelCase로 주고받는다."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ErrorResponse(CamelModel):
    code: str
    message: str
