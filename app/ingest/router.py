import asyncio
from uuid import UUID

from fastapi import APIRouter, Depends, Request, status

from app.core.auth import require_internal_secret
from app.ingest.schemas import IngestRequest
from app.ingest.service import process_ingest

router = APIRouter(tags=["ingest"], dependencies=[Depends(require_internal_secret)])


class IngestTasks:
    """처리 중인 작업. Spring이 같은 jobId를 다시 보내도 겹쳐 돌지 않게 한다."""

    def __init__(self) -> None:
        self._tasks: dict[UUID, asyncio.Task] = {}
        # 임베딩과 LLM 호출이 몰리지 않게 문서를 하나씩 처리한다
        self._lock = asyncio.Lock()

    def submit(self, request: IngestRequest) -> None:
        if request.job_id in self._tasks:
            return
        task = asyncio.create_task(self._run(request))
        self._tasks[request.job_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(request.job_id, None))

    async def _run(self, request: IngestRequest) -> None:
        async with self._lock:
            await process_ingest(request)

    async def close(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
async def ingest(body: IngestRequest, request: Request) -> dict[str, bool]:
    request.app.state.ingest_tasks.submit(body)
    return {"accepted": True}
