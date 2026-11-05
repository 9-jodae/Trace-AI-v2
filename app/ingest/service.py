import asyncio
import logging

import httpx

from app.core import config
from app.core.errors import describe_exception
from app.ingest.extraction import UnsupportedDocumentError, extract_text
from app.ingest.schemas import IngestRequest
from app.ingest.storage import read_object
from app.retrieval.indexing import index_document
from app.retrieval.service import get_real_stores

logger = logging.getLogger(__name__)

CALLBACK_ATTEMPTS = 3
CALLBACK_RETRY_SECONDS = 1.0
CALLBACK_TIMEOUT_SECONDS = 10.0


class IngestFailure(Exception):
    """콜백으로 보내도 되는 고정 문구만 담는다. 예외 원문이나 본문은 넣지 않는다."""


async def send_result(job_id: str, status: str, error_message: str | None = None) -> None:
    # 콜백이 끝내 실패하면 작업은 Spring에서 PENDING이나 PROCESSING으로 남고, Spring 복구 작업이 다시 제출한다
    url = f"{config.spring_internal_url()}/api/internal/ingestion-jobs/{job_id}/result"
    headers = {"X-Internal-Secret": config.internal_secret() or ""}
    body = {"status": status, "errorMessage": error_message}
    for attempt in range(1, CALLBACK_ATTEMPTS + 1):
        try:
            async with httpx.AsyncClient(timeout=CALLBACK_TIMEOUT_SECONDS) as client:
                response = await client.post(url, headers=headers, json=body)
                response.raise_for_status()
            return
        except httpx.HTTPError as exc:
            logger.warning("Ingest callback failed: attempt=%d %s", attempt, type(exc).__name__)
            if attempt < CALLBACK_ATTEMPTS:
                await asyncio.sleep(CALLBACK_RETRY_SECONDS)


async def _index(request: IngestRequest) -> int:
    vector_store, _ = get_real_stores()
    version_id = str(request.version_id)
    info = await vector_store.current_version_document(version_id)
    if info is None:
        # 그사이 새 버전이 저장됐거나 삭제됐다. 처리할 것이 없으므로 실패가 아니다
        logger.info("Ingest skipped: not the current version")
        return 0
    try:
        data = await read_object(request.storage_key)
    except Exception as exc:
        logger.error("Ingest storage read failed: %s", describe_exception(exc))
        raise IngestFailure("원본 파일을 읽지 못했습니다.") from exc
    try:
        text = extract_text(request.extension, data)
    except UnsupportedDocumentError as exc:
        raise IngestFailure("지원하지 않는 문서 형식입니다.") from exc
    except Exception as exc:
        logger.error("Ingest extraction failed: %s", type(exc).__name__)
        raise IngestFailure("문서 본문을 추출하지 못했습니다.") from exc
    if not text.strip():
        raise IngestFailure("문서 본문이 비어 있습니다.")
    chunks = await index_document(
        {"document_id": info["document_id"], "version_id": version_id, "text": text, "source": info["title"]},
        reuse_embeddings=True,
    )
    return len(chunks)


async def process_ingest(request: IngestRequest) -> None:
    job_id = str(request.job_id)
    await send_result(job_id, "PROCESSING")
    try:
        count = await _index(request)
    except IngestFailure as exc:
        logger.warning("Ingest failed: job=%s", job_id)
        await send_result(job_id, "FAILED", str(exc))
    except Exception as exc:
        logger.error("Ingest failed: job=%s %s", job_id, describe_exception(exc))
        await send_result(job_id, "FAILED", "문서를 처리하지 못했습니다.")
    else:
        logger.info("Ingest finished: job=%s chunks=%d", job_id, count)
        await send_result(job_id, "DONE")
