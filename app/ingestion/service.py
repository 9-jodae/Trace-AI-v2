import logging

from pydantic import ValidationError

from app.core.errors import AiUnavailableError, describe_exception
from app.ingestion.extraction import ExtractionLimitError, extract_document
from app.ingestion.normalization import normalize_document
from app.ingestion.schemas import DocumentSubmission, IngestionResult, NormalizedDocument
from app.retrieval.chunking import chunk_document
from app.retrieval.embedding import embed_documents
from app.retrieval.neo4j_store import DocumentVersionConflict
from app.retrieval.qdrant_store import QdrantVectorStore
from app.retrieval.service import get_real_stores

logger = logging.getLogger(__name__)


async def prepare_document(document: DocumentSubmission) -> NormalizedDocument:
    try:
        candidate = await extract_document(document)
    except ExtractionLimitError:
        return normalize_document(document, None, "EXTRACTION_LIMIT")
    except ValidationError:
        return normalize_document(document, None, "INVALID_EXTRACTION_SCHEMA")
    except Exception as exc:
        # 모델 장애 때문에 검증된 원문의 검색까지 막지는 않는다. 오류 원문은 기록하지 않는다.
        logger.warning("Document extraction unavailable: %s", describe_exception(exc))
        return normalize_document(document, None, "EXTRACTION_UNAVAILABLE")
    return normalize_document(document, candidate)


async def resolve_references(document: NormalizedDocument, graph_store) -> None:
    for reference in document.references:
        if reference.target_title.casefold() == document.title.casefold():
            reference.reason = "SELF_REFERENCE"
        else:
            candidates = await graph_store.find_reference_documents(
                reference.target_title, document.document_id, document.allowed_group_ids
            )
            if len(candidates) == 1 and candidates[0].get("version_id"):
                reference.target_document_id = candidates[0]["document_id"]
                reference.target_version_id = candidates[0]["version_id"]
                reference.status = "VALIDATED"
                reference.reason = "EXACT_TITLE_AND_SOURCE_EVIDENCE"
            else:
                reference.reason = "AMBIGUOUS_TARGET" if candidates else "TARGET_UNRESOLVED"
        if reference.status != "VALIDATED" and reference.reason not in document.review_reasons:
            document.review_reasons.append(reference.reason)


async def store_document(document: NormalizedDocument, *, vector_store=None, graph_store=None) -> IngestionResult:
    if vector_store is None or graph_store is None:
        vector_store, graph_store = get_real_stores()
        if not isinstance(vector_store, QdrantVectorStore):
            raise AiUnavailableError("문서 적재에는 Qdrant 저장소 설정이 필요합니다.")
    try:
        await resolve_references(document, graph_store)
        chunks = chunk_document(document.document_id, document.version_id, document.text, document.title)
        payload = {
            "status": document.status, "allowed_group_ids": document.allowed_group_ids, "index_status": "STAGING",
            "document_type": document.document_type, "classification_status": document.classification_status,
            "classification_reason": document.classification_reason, "schema_version": document.schema_version,
            "content_hash": document.content_hash, "topics": [topic.value for topic in document.topics],
            "author": document.author,
            "authored_at": document.authored_at.isoformat() if document.authored_at else None,
            "source_modified_at": (
                document.source_modified_at.isoformat() if document.source_modified_at else None
            ),
            "department": document.department,
        }
        chunks = [{**chunk, **payload} for chunk in chunks]
        embeddings = await embed_documents([chunk["text"] for chunk in chunks])
        await vector_store.replace_version(document.version_id, chunks, embeddings)
        properties = {**document.model_dump(mode="json"), "source": document.title, "index_status": "STAGING"}
        previous = await graph_store.replace_version(
            document.version_id, document.document_id, properties, []
        )
        await vector_store.publish_version(document.document_id, document.version_id)
        await graph_store.mark_ready(document.document_id, document.version_id, document.content_hash)
        if previous and previous != document.version_id:
            await vector_store.retire_version(document.document_id, previous)
    except DocumentVersionConflict:
        raise
    except Exception as exc:
        logger.error("Document indexing failed: %s", describe_exception(exc))
        raise AiUnavailableError("문서 적재를 완료하지 못했습니다. 재시도가 필요합니다.") from exc
    return IngestionResult(
        document_id=document.document_id, version_id=document.version_id, chunk_count=len(chunks),
        document_type=document.document_type, classification_status=document.classification_status,
        review_reasons=document.review_reasons,
        reference_count=sum(reference.status == "VALIDATED" for reference in document.references),
    )


async def ingest_document(document: DocumentSubmission) -> IngestionResult:
    return await store_document(await prepare_document(document))
