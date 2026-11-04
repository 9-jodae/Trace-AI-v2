from uuid import UUID, uuid5

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.retrieval.schemas import Chunk

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# 재색인해도 같은 위치의 청크가 같은 UUID를 갖도록 버전 ID와 순번으로 만든다
CHUNK_ID_NAMESPACE = UUID("6f0f6a55-6a8e-4f0f-9f55-0d1a9c199a11")

# 문단, 줄, 문장, 단어 순으로 큰 단위부터 나눈다
SEPARATORS = ["\n\n", "\n", "다. ", ". ", "? ", "! ", " ", ""]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=SEPARATORS,
    keep_separator="end",
    add_start_index=True,
)


def chunk_document(document_id: str, version_id: str, text: str, source: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for index, piece in enumerate(splitter.create_documents([text])):
        start = piece.metadata["start_index"]
        chunks.append(
            {
                "id": str(uuid5(CHUNK_ID_NAMESPACE, f"{version_id}:{index}")),
                "text": piece.page_content,
                "source": source,
                "document_id": document_id,
                "version_id": version_id,
                "char_start": start,
                "char_end": start + len(piece.page_content),
            }
        )
    return chunks
