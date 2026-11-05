import re
import unicodedata
from pathlib import PurePath
from uuid import NAMESPACE_URL, uuid5

from typing_extensions import TypedDict

from app.retrieval.schemas import Chunk, DocumentInput, EntityType, RelationAssertion

REFERENCE_PATTERN = re.compile(r"《([^》\n]{1,200})》|\[\[([^\]\n]{1,200})\]\]")
MIN_NAME_LENGTH = 2
MAX_DOCUMENT_NAME_LENGTH = 200


def entity_id(entity_type: EntityType, namespace_id: str, document_id: str, canonical_name: str) -> str:
    """같은 문서 안에서만 동일 이름 Entity를 묶고 문서 간 동명은 별도 후보로 둔다."""
    return str(uuid5(
        NAMESPACE_URL,
        f"docs199:entity:v3:{namespace_id}:{document_id}:{entity_type}:{normalize_name(canonical_name)}",
    ))


def mention_is_grounded(mention: dict, chunk: Chunk) -> bool:
    start, end = mention.get("char_start"), mention.get("char_end")
    evidence_start, evidence_end = mention.get("evidence_start"), mention.get("evidence_end")
    if not all(isinstance(value, int) for value in (start, end, evidence_start, evidence_end)):
        return False
    chunk_start, chunk_end = chunk.get("char_start"), chunk.get("char_end")
    if not isinstance(chunk_start, int) or not isinstance(chunk_end, int):
        return False
    if not (chunk_start <= evidence_start <= start < end <= evidence_end <= chunk_end):
        return False
    local_start, local_end = evidence_start - chunk_start, evidence_end - chunk_start
    name_start, name_end = start - chunk_start, end - chunk_start
    quote = mention.get("evidence_quote")
    return (
        isinstance(quote, str)
        and chunk["text"][local_start:local_end] == quote
        and chunk["text"][name_start:name_end] == mention.get("surface_form")
        and mention["surface_form"] in quote
    )


def validate_assertion(assertion: RelationAssertion, chunks_by_id: dict[str, Chunk]) -> bool:
    """근거 원문과 버전·문서 소속이 일치하는 단순 현재 관계만 적재 대상으로 허용한다."""
    if assertion["polarity"] != "AFFIRMED" or assertion["modality"] != "ACTUAL":
        return False
    if assertion["subject_entity_id"] == assertion["object_entity_id"]:
        return False
    chunk = chunks_by_id.get(assertion["evidence_chunk_id"])
    if not chunk or chunk.get("version_id") != assertion["source_version_id"]:
        return False
    if chunk.get("document_id") != assertion["source_document_id"]:
        return False
    start, end = assertion["evidence_start"], assertion["evidence_end"]
    chunk_start, chunk_end = chunk.get("char_start"), chunk.get("char_end")
    if not isinstance(chunk_start, int) or not isinstance(chunk_end, int):
        return False
    if not chunk_start <= start < end <= chunk_end:
        return False
    local_start, local_end = start - chunk_start, end - chunk_start
    return chunk["text"][local_start:local_end] == assertion["evidence_quote"]


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
