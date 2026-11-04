from uuid import uuid4

from fastapi import APIRouter, Depends, Request
from langchain_core.messages import HumanMessage

from app.agent.schemas import ChatRequest, ChatResponse
from app.core.auth import require_internal_secret

router = APIRouter(prefix="/api/v1/agent", tags=["agent"], dependencies=[Depends(require_internal_secret)])


@router.post("/chat")
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    thread_id = body.thread_id or str(uuid4())
    conversations = request.app.state.conversations
    saved = conversations.get(thread_id)
    result = await request.app.state.agent.ainvoke(
        {"messages": [*saved["messages"], HumanMessage(content=body.message)], "context": saved["context"]},
    )
    # 실패한 요청은 여기까지 오지 않으므로 대화 기록과 컨텍스트에 남지 않는다
    conversations.save(thread_id, result["messages"], result["context"])
    return ChatResponse(
        thread_id=thread_id,
        answer=result["messages"][-1].text,
        route=result["route"],
        context_probability=result["context_probability"],
        sources=result["context"],
        failed_sources=result["failed_sources"],
        fallback_reason=result["fallback_reason"],
    )
