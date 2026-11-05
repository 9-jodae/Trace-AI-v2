import argparse
import asyncio
import json
import logging
from pathlib import Path

from app.core import config
from app.retrieval.chunking import chunk_document
from app.retrieval.document_topics import extract_document_topics
from app.retrieval.embedding import embed_documents
from app.retrieval.entity_extraction import extract_entity_graph
from app.retrieval.neo4j_store import Neo4jGraphStore
from app.retrieval.relations import document_names, resolve_references
from app.retrieval.schemas import Chunk, DocumentInput
from app.retrieval.service import close_stores, get_real_stores

logger = logging.getLogger(__name__)


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


async def index_document(
    document: DocumentInput,
    reference_catalog: list[DocumentInput] | None = None,
    update_references: bool = True,
) -> list[Chunk]:
    chunks = chunk_document(document["document_id"], document["version_id"], document["text"], document["source"])
    if not chunks:
        return []
    vector_store, graph_store = get_real_stores()
    # 구버전 작업이면 임베딩 비용을 쓰지 않는다
    if not await vector_store.is_current_version(document["version_id"]):
        logger.warning("Indexing skipped: not the current version")
        return []
    embeddings = await embed_documents([chunk["text"] for chunk in chunks])
    if not await vector_store.replace_version(document["version_id"], chunks, embeddings):
        logger.warning("Indexing skipped: version changed while embedding")
        return []
    await graph_store.replace_document(
        document["document_id"], document["source"], document_names(document), chunks
    )
    if isinstance(graph_store, Neo4jGraphStore):
        # 추출을 꺼도 교체는 실행해, 켰을 때 저장한 관계와 인용이 남지 않게 한다
        mentions, assertions = [], []
        if config.entity_extraction_enabled():
            mentions, assertions = await extract_entity_graph(
                document["document_id"], document["version_id"], chunks
            )
        await graph_store.replace_entity_graph(document["document_id"], document["version_id"], mentions, assertions)
    if isinstance(graph_store, Neo4jGraphStore):
        # 꺼져 있어도 교체해, 켰을 때 저장한 주제가 남지 않게 한다
        topics = []
        if config.document_topics_enabled():
            topics = await extract_document_topics(document, await graph_store.list_topics())
        # 일시적인 실패로 기존 주제를 지우지 않는다
        if topics is not None:
            await graph_store.replace_document_topics(document["document_id"], topics)
    if update_references:
        catalog = reference_catalog or await _reference_catalog(graph_store)
        await _replace_document_references(graph_store, document, chunks, catalog)
    return chunks


async def _reference_catalog(graph_store, documents: list[DocumentInput] = ()) -> list[DocumentInput]:
    catalog = await graph_store.list_document_catalog()
    catalog_by_id: dict[str, DocumentInput] = {
        item["document_id"]: {
            "document_id": item["document_id"],
            "version_id": "",
            "text": "",
            "source": item["source"],
        }
        for item in catalog
    }
    catalog_by_id.update({document["document_id"]: document for document in documents})
    return list(catalog_by_id.values())


async def _replace_document_references(graph_store, document: DocumentInput, chunks: list[Chunk], catalog):
    catalog_by_id = {item["document_id"]: item for item in catalog}
    catalog_by_id[document["document_id"]] = document
    links = resolve_references(document, chunks, list(catalog_by_id.values()))
    await graph_store.replace_references(document["document_id"], links)


async def index_file(path: Path) -> int:
    try:
        documents = load_documents(path)
        _, graph_store = get_real_stores()
        reference_catalog = await _reference_catalog(graph_store, documents)
        written = 0
        errors: list[Exception] = []
        indexed: list[tuple[DocumentInput, list[Chunk]]] = []
        # 각 문서를 독립 처리해 하나의 실패가 앞선 문서의 참조 갱신까지 건너뛰게 하지 않는다.
        for document in documents:
            try:
                chunks = await index_document(document, reference_catalog, update_references=False)
                written += len(chunks)
                if chunks:
                    indexed.append((document, chunks))
            except Exception as exc:
                errors.append(exc)
                logger.error("Document indexing failed; remaining documents will continue")
        # 모든 성공 문서 노드가 생긴 뒤 관계를 바꿔 참조 대상의 입력 순서에 의존하지 않게 한다.
        for document, chunks in indexed:
            try:
                await _replace_document_references(graph_store, document, chunks, reference_catalog)
            except Exception as exc:
                errors.append(exc)
                logger.error("Reference indexing failed; remaining documents will continue")
        if errors:
            # 원래 예외가 문서 본문·연결 문자열을 포함할 수 있어 CLI에 전파하지 않는다.
            raise RuntimeError(f"{len(errors)}개 문서 적재에 실패했습니다.")
        return written
    finally:
        await close_stores()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSON 문서 파일을 저장소에 적재한다")
    parser.add_argument("path", type=Path)
    print(f"적재한 청크: {asyncio.run(index_file(parser.parse_args().path))}개")
