import logging
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.core import config
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
        logger.exception("JEV judge failed")
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


def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("judge_context", judge_context)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "judge_context")
    graph.add_conditional_edges("judge_context", select_route, ["generate", "retrieve"])
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    # InMemorySaver는 프로세스 재시작 시 대화가 사라지므로 운영에서는 영속 저장소로 교체
    return graph.compile(checkpointer=InMemorySaver())
