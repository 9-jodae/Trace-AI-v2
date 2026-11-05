import asyncio
import logging
import time
from uuid import uuid4

from fastapi import APIRouter, Depends, Request
from langchain_core.messages import HumanMessage

from app.agent.schemas import ChatRequest, ChatResponse
from app.core import config
from app.core.auth import require_internal_secret
from app.core.errors import AiUnavailableError
from app.retrieval.service import KnownConflict

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/agent", tags=["agent"], dependencies=[Depends(require_internal_secret)])


@router.post("/chat")
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    thread_id = body.thread_id or str(uuid4())
    conversations = request.app.state.conversations
    started = time.perf_counter()
    try:
        # 상한을 넘기면 진행 중인 호출을 취소하므로 대체 모델도 더 시도하지 않는다
        async with asyncio.timeout(config.request_timeout_seconds()):
            # 같은 대화의 요청을 차례로 처리해야 앞선 턴이 덮어써지지 않는다
            async with conversations.lock(thread_id):
                saved = conversations.get(thread_id)
                result = await request.app.state.agent.ainvoke(
                    {
                        "messages": [*saved["messages"], HumanMessage(content=body.message)],
                        "context": saved["context"],
                        "filters": body.filters,
                        "access_scope": body.access_scope.to_scope() if body.access_scope else None,
                        "known_conflicts": [
                            KnownConflict(conflict.document_id, conflict.other_document_id, conflict.item)
                            for conflict in body.known_conflicts or []
                        ],
                    },
                )
                # 실패한 요청은 여기까지 오지 않으므로 대화 기록과 컨텍스트에 남지 않는다
                conversations.save(thread_id, result["messages"], result["context"])
    except TimeoutError as exc:
        raise AiUnavailableError("요청 전체 시간 상한을 넘었습니다.") from exc
    logger.info("Chat finished: route=%s total_ms=%.0f", result["route"], (time.perf_counter() - started) * 1000)
    return ChatResponse(
        thread_id=thread_id,
        answer=result["messages"][-1].text,
        answer_type=result["answer_type"],
        route=result["route"],
        context_probability=result["context_probability"],
        sources=result["context"],
        evidence=result["evidence"],
        keywords=result["keywords"],
        edit_proposals=result.get("edit_proposals", []),
        failed_sources=result["failed_sources"],
        fallback_reason=result["fallback_reason"],
    )
