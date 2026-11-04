from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings


def split_document(doc: dict) -> list[dict]:
    """Split source text into LangChain chunks and retain character offsets."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", "다. ", "요. ", " ", ""],
    )
    chunks = []
    for index, item in enumerate(splitter.create_documents([doc["text"]])):
        start = int(item.metadata["start_index"])
        text = item.page_content
        end = start + len(text)
        chunks.append({
            **{k: v for k, v in doc.items() if k != "text"},
            "chunk_id": f"{doc['doc_id']}#chunk-{index + 1:03d}",
            "text": text,
            "char_start": start,
            "char_end": end,
        })
    return chunks
