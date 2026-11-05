import argparse
import asyncio
import json
from pathlib import Path

from app.retrieval.chunking import chunk_document
from app.retrieval.embedding import embed_documents
from app.retrieval.entities import extract_mentions
from app.retrieval.relations import document_names, resolve_references
from app.retrieval.schemas import Chunk, DocumentInput
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


async def index_document(document: DocumentInput, extract_entities: bool = False) -> list[Chunk]:
    chunks = chunk_document(document["document_id"], document["version_id"], document["text"], document["source"])
    if not chunks:
        return []
    embeddings = await embed_documents([chunk["text"] for chunk in chunks])
    vector_store, graph_store = get_real_stores()
    await vector_store.replace_version(document["version_id"], chunks, embeddings)
    # 본문이 LLM으로 나가므로 명시했을 때만 추출한다
    mentions = await extract_mentions(chunks) if extract_entities else []
    await graph_store.replace_version(
        document["document_id"], document["source"], document_names(document), chunks, mentions
    )
    return chunks


async def index_file(path: Path, extract_entities: bool = False) -> int:
    try:
        documents = load_documents(path)
        written = [(document, await index_document(document, extract_entities)) for document in documents]
        # 참조 대상 문서가 모두 저장된 뒤에 연결한다
        _, graph_store = get_real_stores()
        for document, chunks in written:
            if chunks:
                links = resolve_references(document, chunks, documents)
                await graph_store.replace_references(document["document_id"], links)
        return sum(len(chunks) for _, chunks in written)
    finally:
        await close_stores()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSON 문서 파일을 저장소에 적재한다")
    parser.add_argument("path", type=Path)
    parser.add_argument("--extract-entities", action="store_true", help="청크마다 LLM으로 엔티티를 추출한다")
    arguments = parser.parse_args()
    print(f"적재한 청크: {asyncio.run(index_file(arguments.path, arguments.extract_entities))}개")
