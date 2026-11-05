import re
import unicodedata
from pathlib import PurePath

from typing_extensions import TypedDict

from app.retrieval.schemas import Chunk, DocumentInput

REFERENCE_PATTERN = re.compile(r"《([^》\n]{1,200})》|\[\[([^\]\n]{1,200})\]\]")
MIN_NAME_LENGTH = 2
MAX_DOCUMENT_NAME_LENGTH = 200


class DocumentReference(TypedDict):
    source_document_id: str
    target_document_id: str
    chunk_id: str


def normalize_name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def document_names(document: DocumentInput) -> list[str]:
    """문서 제목만 참조 해석 이름으로 사용해 본문이 그래프 메타데이터에 복사되지 않게 한다."""
    source = document["source"]
    names = [name for name in (source, PurePath(source).stem) if name and len(name) <= MAX_DOCUMENT_NAME_LENGTH]
    return list(dict.fromkeys(normalized for name in names if (normalized := normalize_name(name))))


def _chunk_at(chunks: list[Chunk], position: int) -> Chunk | None:
    return next((chunk for chunk in chunks if chunk["char_start"] <= position < chunk["char_end"]), None)


def resolve_references(
    document: DocumentInput, chunks: list[Chunk], documents: list[DocumentInput]
) -> list[DocumentReference]:
    """문서 본문에 명시된 참조만 연결하고, 모호한 이름은 건너뛴다."""
    targets: dict[str, set[str]] = {}
    for candidate in documents:
        for name in document_names(candidate):
            targets.setdefault(name, set()).add(candidate["document_id"])

    links: dict[str, DocumentReference] = {}
    for match in REFERENCE_PATTERN.finditer(document["text"]):
        found = targets.get(normalize_name(match.group(1) or match.group(2)), set())
        chunk = _chunk_at(chunks, match.start())
        if len(found) != 1 or chunk is None:
            continue
        target = next(iter(found))
        if target != document["document_id"]:
            links.setdefault(
                target,
                {
                    "source_document_id": document["document_id"],
                    "target_document_id": target,
                    "chunk_id": chunk["id"],
                },
            )
    return list(links.values())
