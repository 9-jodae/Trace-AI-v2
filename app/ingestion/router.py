import asyncio

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.core import config
from app.core.auth import require_internal_secret
from app.core.errors import AiUnavailableError
from app.core.schemas import ErrorResponse
from app.ingestion.schemas import DocumentSubmission, IngestionResult
from app.ingestion.service import ingest_document
from app.retrieval.neo4j_store import DocumentVersionConflict

# AI 내부 텍스트 입력 API. Spring의 원본 파일 /ingest 계약을 대신하지 않는다.
router = APIRouter(prefix="/api/v1/ingestion", dependencies=[Depends(require_internal_secret)])


@router.post("/documents", response_model=IngestionResult)
async def ingest(document: DocumentSubmission):
    try:
        return await asyncio.wait_for(ingest_document(document), config.ingestion_timeout_seconds())
    except DocumentVersionConflict:
        return JSONResponse(
            status_code=409,
            content=ErrorResponse(
                code="DOCUMENT_VERSION_CONFLICT", message="현재 문서 버전과 충돌합니다. 입력을 확인한 후 재시도하세요."
            ).model_dump(by_alias=True),
        )
    except TimeoutError as exc:
        raise AiUnavailableError("문서 적재 제한 시간을 초과했습니다.") from exc
