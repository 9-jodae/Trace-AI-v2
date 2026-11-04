from uuid import uuid4

from fastapi import APIRouter, Request
from langchain_core.messages import HumanMessage

from app.agent.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


@router.post("/chat")
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    thread_id = body.thread_id or str(uuid4())
    result = await request.app.state.agent.ainvoke(
        {"messages": [HumanMessage(content=body.message)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return ChatResponse(
        thread_id=thread_id,
        answer=result["messages"][-1].text,
        route=result["route"],
        context_probability=result["context_probability"],
        sources=result["context"],
        failed_sources=result["failed_sources"],
        fallback_reason=result["fallback_reason"],
    )
