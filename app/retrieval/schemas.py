from typing_extensions import NotRequired, TypedDict


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
