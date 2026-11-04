from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.retrieval.schemas import Chunk

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# 문단, 줄, 문장, 단어 순으로 큰 단위부터 나눈다
SEPARATORS = ["\n\n", "\n", "다. ", ". ", "? ", "! ", " ", ""]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=SEPARATORS,
    keep_separator="end",
)


def chunk_document(document_id: str, text: str, source: str) -> list[Chunk]:
    return [
        {"id": f"{document_id}-{index}", "text": piece, "source": source}
        for index, piece in enumerate(splitter.split_text(text), start=1)
    ]
