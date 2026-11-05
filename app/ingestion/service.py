import asyncio
import logging
from collections import OrderedDict
from functools import lru_cache

import boto3
import httpx

from app.core import config
from app.core.errors import describe_exception
from app.ingestion.parsers import UnsupportedFormatError, extract_text
from app.ingestion.schemas import IngestRequest
from app.retrieval.indexing import index_document
from app.retrieval.relations import resolve_references
from app.retrieval.schemas import DocumentInput
from app.retrieval.service import get_real_stores

logger = logging.getLogger(__name__)

MAX_CONCURRENT_JOBS = 2
MAX_REMEMBERED_JOBS = 1000
MAX_FILE_BYTES = 50_000_000
JOB_TIMEOUT_SECONDS = 300
CALLBACK_TIMEOUT_SECONDS = 10
CALLBACK_ATTEMPTS = 3
RUNNING, DONE, FAILED = "PROCESSING", "DONE", "FAILED"


class IngestionError(Exception):
    """Spring에 그대로 전달해도 되는 실패 사유. 문서 내용과 내부 예외를 담지 않는다."""


@lru_cache(maxsize=1)
def _s3_client():
    # 서버에서는 인스턴스 역할을 쓴다
    return boto3.client("s3")


def _download(storage_key: str) -> bytes:
    bucket = config.s3_bucket()
    if not bucket:
        raise IngestionError("S3_BUCKET이 설정되지 않았습니다.")
    response = _s3_client().get_object(Bucket=bucket, Key=config.s3_prefix() + storage_key.lstrip("/"))
    if response["ContentLength"] > MAX_FILE_BYTES:
        raise IngestionError("파일이 너무 큽니다.")
    with response["Body"] as stream:
        return stream.read()


async def _index(request: IngestRequest) -> None:
    vector_store, graph_store = get_real_stores()
    version = await vector_store.find_version(str(request.version_id))
    if version is None:
        raise IngestionError("문서 버전을 찾을 수 없습니다.")
    data = await asyncio.to_thread(_download, request.storage_key)
    try:
        text = await asyncio.to_thread(extract_text, request.extension, data)
    except UnsupportedFormatError as exc:
        raise IngestionError("지원하지 않는 파일 형식입니다.") from exc
    if not text.strip():
        raise IngestionError("추출한 본문이 비어 있습니다.")
    document: DocumentInput = {
        "document_id": version["document_id"],
        "version_id": str(request.version_id),
        "text": text,
        "source": version["title"],
    }
    chunks = await index_document(document)
    if chunks:
        links = resolve_references(document, chunks, await graph_store.name_targets())
        await graph_store.replace_references(document["document_id"], links)


async def _send_result(job_id: str, status: str, error: str | None = None) -> None:
    base_url = config.spring_base_url()
    if not base_url:
        logger.error("Ingestion callback skipped: SPRING_BASE_URL is not set")
        return
    url = f"{base_url.rstrip('/')}/api/internal/ingestion-jobs/{job_id}/result"
    headers = {"X-Internal-Secret": config.internal_secret() or ""}
    async with httpx.AsyncClient(timeout=CALLBACK_TIMEOUT_SECONDS) as client:
        for attempt in range(1, CALLBACK_ATTEMPTS + 1):
            try:
                response = await client.post(url, headers=headers, json={"status": status, "errorMessage": error})
                response.raise_for_status()
                return
            except httpx.HTTPError as exc:
                logger.warning("Ingestion callback failed: attempt=%d %s", attempt, describe_exception(exc))
    logger.error("Ingestion callback gave up: status=%s", status)


class IngestionService:
    def __init__(self) -> None:
        self._results: OrderedDict[str, tuple[str, str | None]] = OrderedDict()
        self._tasks: set[asyncio.Task] = set()
        self._slots = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

    def submit(self, request: IngestRequest) -> None:
        # Spring은 복구 때 같은 jobId를 다시 보낸다. 처리 중이면 무시하고, 끝났으면 결과만 다시 알린다
        job_id = str(request.job_id)
        known = self._results.get(job_id)
        if known and known[0] == RUNNING:
            return
        work = _send_result(job_id, *known) if known else self._run(job_id, request)
        task = asyncio.create_task(work)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        if not known:
            self._remember(job_id, RUNNING, None)

    def _remember(self, job_id: str, status: str, error: str | None) -> None:
        self._results[job_id] = (status, error)
        self._results.move_to_end(job_id)
        while len(self._results) > MAX_REMEMBERED_JOBS:
            self._results.popitem(last=False)

    async def _run(self, job_id: str, request: IngestRequest) -> None:
        async with self._slots:
            await _send_result(job_id, RUNNING)
            status, error = DONE, None
            try:
                await asyncio.wait_for(_index(request), JOB_TIMEOUT_SECONDS)
            except IngestionError as exc:
                status, error = FAILED, str(exc)
            except TimeoutError:
                status, error = FAILED, "처리 시간 상한을 넘었습니다."
            except Exception as exc:
                logger.error("Ingestion failed: %s", describe_exception(exc))
                status, error = FAILED, f"문서 처리에 실패했습니다 ({type(exc).__name__})."
            self._remember(job_id, status, error)
            await _send_result(job_id, status, error)

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
