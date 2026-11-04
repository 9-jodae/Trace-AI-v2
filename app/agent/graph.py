import logging
import time
from typing import Annotated, Awaitable, Callable, Literal

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.core import config
from app.core.errors import describe_exception
from app.jev.client import can_answer_probability
from app.llm.client import generate_answer
from app.retrieval.schemas import Chunk
from app.retrieval.service import search

logger = logging.getLogger(__name__)

# 잘못 재사용하면 틀린 답이 나가므로 애매하면 검색하도록 높게 잡는다
CONTEXT_REUSE_THRESHOLD = 0.7
HISTORY_LIMIT = 6
NO_RESULT_ANSWER = "관련 문서를 찾지 못했습니다."


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    context: list[Chunk]
    route: Literal["CONTEXT", "SEARCH"]
    context_probability: float | None
    failed_sources: list[str]
    fallback_reason: str | None


def _history(state: AgentState) -> list[AnyMessage]:
    return state["messages"][:-1][-HISTORY_LIMIT:]


async def judge_context(state: AgentState) -> dict:
    decision = {"route": "SEARCH", "context_probability": None, "failed_sources": [], "fallback_reason": None}
    context = state.get("context")
    if not context:
        return decision
    if not config.jev_api_key():
        return {**decision, "fallback_reason": "JEV_API_KEY가 설정되지 않았습니다."}

    history = [f"{message.type}: {message.text}" for message in _history(state)]
    try:
        probability = await can_answer_probability(state["messages"][-1].text, context, history)
    except Exception as exc:
        logger.error("JEV judge failed: %s", describe_exception(exc))
        return {**decision, "fallback_reason": f"JEV 호출 실패 ({type(exc).__name__})"}
    route = "CONTEXT" if probability >= CONTEXT_REUSE_THRESHOLD else "SEARCH"
    return {**decision, "route": route, "context_probability": probability}


def select_route(state: AgentState) -> str:
    return "generate" if state["route"] == "CONTEXT" else "retrieve"


async def retrieve(state: AgentState) -> dict:
    result = await search(state["messages"][-1].text)
    return {"context": result["chunks"], "failed_sources": result["failed_sources"]}


async def generate(state: AgentState) -> dict:
    if not state["context"]:
        return {"messages": [AIMessage(content=NO_RESULT_ANSWER)]}
    answer = await generate_answer(state["messages"][-1].text, state["context"], _history(state))
    return {"messages": [AIMessage(content=answer)]}


def _timed(name: str, step: Callable[[AgentState], Awaitable[dict]]) -> Callable[[AgentState], Awaitable[dict]]:
    async def run(state: AgentState) -> dict:
        started = time.perf_counter()
        try:
            return await step(state)
        finally:
            # 질문과 문서 본문은 남기지 않는다
            logger.info("Step finished: step=%s elapsed_ms=%.0f", name, (time.perf_counter() - started) * 1000)

    return run


def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("judge_context", _timed("judge_context", lambda state: judge_context(state)))
    graph.add_node("retrieve", _timed("retrieve", lambda state: retrieve(state)))
    graph.add_node("generate", _timed("generate", lambda state: generate(state)))
    graph.add_edge(START, "judge_context")
    graph.add_conditional_edges("judge_context", select_route, ["generate", "retrieve"])
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()
