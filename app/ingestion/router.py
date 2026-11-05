from fastapi import APIRouter, Depends, Request

from app.core.auth import require_internal_secret
from app.ingestion.schemas import IngestRequest

router = APIRouter(tags=["ingestion"], dependencies=[Depends(require_internal_secret)])


@router.post("/ingest", status_code=202)
async def ingest(body: IngestRequest, request: Request) -> dict[str, bool]:
    # 처리 결과는 Spring 콜백으로 알린다. 파일을 받고 임베딩하는 동안 Spring 요청을 붙잡지 않는다
    request.app.state.ingestion.submit(body)
    return {"accepted": True}
