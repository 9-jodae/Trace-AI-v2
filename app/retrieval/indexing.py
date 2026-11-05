import argparse
import asyncio
import json
from pathlib import Path

from app.retrieval.chunking import chunk_document
from app.retrieval.embedding import embed_documents
from app.retrieval.schemas import DocumentInput
from app.retrieval.service import close_stores, get_real_stores


def load_documents(path: Path) -> list[DocumentInput]:
    # 파일 업로드와 파서가 생기기 전까지 쓰는 입력 형식. PR #17의 JSON 스냅샷을 그대로 읽는다
    documents = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "document_id": item["doc_id"],
            "version_id": item.get("version_id", item["doc_id"]),
            "text": item["text"],
            "source": item["title"],
        }
        for item in documents
    ]


async def index_document(document: DocumentInput) -> int:
    chunks = chunk_document(document["document_id"], document["version_id"], document["text"], document["source"])
    if not chunks:
        return 0
    embeddings = await embed_documents([chunk["text"] for chunk in chunks])
    vector_store, graph_store = get_real_stores()
    await vector_store.replace_version(document["version_id"], chunks, embeddings)
    metadata = {key: value for key, value in document.items() if key != "text"}
    await graph_store.replace_version(document["version_id"], document["document_id"], metadata, chunks)
    return len(chunks)


async def index_file(path: Path) -> int:
    try:
        return sum([await index_document(document) for document in load_documents(path)])
    finally:
        await close_stores()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSON 문서 파일을 저장소에 적재한다")
    parser.add_argument("path", type=Path)
    print(f"적재한 청크: {asyncio.run(index_file(parser.parse_args().path))}개")
