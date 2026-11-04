from typing_extensions import TypedDict


class Chunk(TypedDict):
    id: str
    text: str
    source: str
