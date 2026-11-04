from typing_extensions import NotRequired, TypedDict

# TODO: 문서 상태값은 서버 계약에서 확정 (docs199-Server#14)
ACTIVE_STATUS = "유효"


class Chunk(TypedDict):
    id: str
    text: str
    source: str | None
    document_id: NotRequired[str | None]
    version_id: NotRequired[str | None]
    char_start: NotRequired[int | None]
    char_end: NotRequired[int | None]
    status: NotRequired[str]
    allowed_group_ids: NotRequired[list[str]]


class DocumentInput(TypedDict):
    document_id: str
    version_id: str
    text: str
    source: str
    status: str
    allowed_group_ids: list[str]
