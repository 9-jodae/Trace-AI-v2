import hashlib
from uuid import UUID, uuid5

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.retrieval.schemas import Chunk

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# 같은 버전과 같은 청킹 설정이면 재색인해도 같은 UUID가 나온다
CHUNK_ID_NAMESPACE = UUID("6f0f6a55-6a8e-4f0f-9f55-0d1a9c199a11")

# 문단, 줄, 문장, 단어 순으로 큰 단위부터 나눈다
SEPARATORS = ["\n\n", "\n", "다. ", ". ", "? ", "! ", " ", ""]


def settings_fingerprint(size: int, overlap: int, separators: list[str]) -> str:
    # 설정이 바뀌면 같은 순번의 청크 내용도 달라지므로 ID가 이전 청크를 가리키지 않게 한다
    return f"{size}:{overlap}:{separators!r}"


SETTINGS_FINGERPRINT = settings_fingerprint(CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=SEPARATORS,
    keep_separator="end",
    add_start_index=True,
)


def chunk_document(document_id: str, version_id: str, text: str, source: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    for index, piece in enumerate(splitter.create_documents([text])):
        start = piece.metadata["start_index"]
        chunks.append(
            {
                "id": str(uuid5(
                    CHUNK_ID_NAMESPACE,
                    f"{document_id}:{version_id}:{content_hash}:{SETTINGS_FINGERPRINT}:{index}",
                )),
                "text": piece.page_content,
                "source": source,
                "document_id": document_id,
                "version_id": version_id,
                "char_start": start,
                "char_end": start + len(piece.page_content),
            }
        )
    return chunks
