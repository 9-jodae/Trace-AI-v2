import re
import unicodedata

from typing_extensions import TypedDict

from app.retrieval.schemas import Chunk, DocumentInput

REFERENCE_PATTERN = re.compile(r"《([^》\n]{1,200})》|\[\[([^\]\n]{1,200})\]\]")
MIN_NAME_LENGTH = 2
# 문서끼리의 연결을 묻는 질문만 2단계까지 넓힌다. 그 외에는 관련 없는 문서가 섞이기 쉽다
TWO_HOP_TERMS = ("관련", "연결", "영향", "연쇄", "참조", "의존")


class DocumentLink(TypedDict):
    source_document_id: str
    target_document_id: str
    chunk_id: str
    quote: str
    char_start: int
    char_end: int


def normalize_name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def name_in_query(name: str, query: str) -> bool:
    # 영문 약칭이 다른 단어 안에서 맞지 않게 하되 한국어 조사는 허용한다
    if len(name) < MIN_NAME_LENGTH:
        return False
    left = r"(?<![a-z0-9])" if name[0].isascii() and name[0].isalnum() else ""
    right = r"(?![a-z0-9])" if name[-1].isascii() and name[-1].isalnum() else ""
    return re.search(left + re.escape(name) + right, query) is not None


def hop_depth(query: str) -> int:
    return 2 if any(term in query for term in TWO_HOP_TERMS) else 1


def document_names(document: DocumentInput) -> list[str]:
    # 제목이 파일 이름 형식이어도 본문 첫 줄의 문서명으로 참조를 찾을 수 있게 한다
    first_line = next((line for line in document["text"].splitlines() if line.strip()), "")
    names = [document["source"], first_line.lstrip("# ").strip()]
    return list(dict.fromkeys(normalized for name in names if (normalized := normalize_name(name))))


def _chunk_at(chunks: list[Chunk], position: int) -> Chunk | None:
    return next((chunk for chunk in chunks if chunk["char_start"] <= position < chunk["char_end"]), None)


def resolve_references(
    document: DocumentInput, chunks: list[Chunk], documents: list[DocumentInput]
) -> list[DocumentLink]:
    # 본문에 명시된 문서명만 연결하고 이름이 겹치면 추측하지 않는다
    targets: dict[str, set[str]] = {}
    for candidate in documents:
        for name in document_names(candidate):
            targets.setdefault(name, set()).add(candidate["document_id"])
    links: dict[str, DocumentLink] = {}
    for match in REFERENCE_PATTERN.finditer(document["text"]):
        found = targets.get(normalize_name(match.group(1) or match.group(2)), set())
        chunk = _chunk_at(chunks, match.start())
        if len(found) != 1 or chunk is None:
            continue
        target = next(iter(found))
        if target != document["document_id"] and target not in links:
            links[target] = {
                "source_document_id": document["document_id"],
                "target_document_id": target,
                "chunk_id": chunk["id"],
                "quote": match.group(0),
                "char_start": match.start(),
                "char_end": match.end(),
            }
    return list(links.values())
