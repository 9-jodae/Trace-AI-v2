import logging
import re
import time
from typing import Annotated, Awaitable, Callable, Literal

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.agent.evidence import Evidence, parse_answer
from app.core import config
from app.core.errors import AiUnavailableError, describe_exception
from app.core.schemas import AccessScope, SearchFilters
from app.jev.client import can_answer_probability, small_talk_probability
from app.llm.client import (
    CLARIFICATION_KEY,
    NO_EVIDENCE_ANSWER,
    generate_answer,
    generate_reply_without_documents,
    is_clarification,
    rewrite_search_query,
)
from app.retrieval.schemas import Chunk
from app.retrieval.service import SearchResult, filter_accessible_context, search

logger = logging.getLogger(__name__)

# 잘못 재사용하면 틀린 답이 나가므로 애매하면 검색하도록 높게 잡는다
CONTEXT_REUSE_THRESHOLD = 0.7
# 문서 질문이 잘못 넘어가면 근거 없이 거절되므로 높게 잡는다
SMALL_TALK_THRESHOLD = 0.9
HISTORY_LIMIT = 6
# 질문만으로는 검색어가 부족한 후속 질문. "그룹"처럼 단어의 일부인 경우는 제외한다
FOLLOW_UP_PATTERN = re.compile(r"^(그럼|그러면|그건|그게|그거|그것|그 |이건|이게|이거|이 경우|앞서|위 )")


# TODO: 이름과 값은 서버 계약에서 확정 (docs199-Server#76)
AnswerType = Literal["ANSWER", "NO_EVIDENCE", "CLARIFICATION", "SMALL_TALK"]


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    small_talk: bool
    alternate_query: str | None
    answer_type: AnswerType
    context: list[Chunk]
    filters: SearchFilters | None
    access_scope: AccessScope | None
    evidence: list[Evidence]
    route: Literal["CONTEXT", "SEARCH"]
    context_probability: float | None
    failed_sources: list[str]
    fallback_reason: str | None


def _history(state: AgentState) -> list[AnyMessage]:
    return state["messages"][:-1][-HISTORY_LIMIT:]


async def judge_small_talk(state: AgentState) -> dict:
    # "네"처럼 되물음에 답한 말은 일상 대화로 보내지 않는다
    if not config.jev_api_key() or _follows_clarification(state):
        return {"small_talk": False}
    try:
        probability = await small_talk_probability(state["messages"][-1].text)
    except Exception as exc:
        # 판단하지 못하면 문서 검색으로 진행한다
        logger.error("Small talk judge failed: %s", describe_exception(exc))
        return {"small_talk": False}
    return {"small_talk": probability >= SMALL_TALK_THRESHOLD}


def select_small_talk(state: AgentState) -> str:
    return "small_talk" if state["small_talk"] else "judge_context"


def select_after_small_talk(state: AgentState) -> str:
    return END if state["small_talk"] else "judge_context"


async def small_talk(state: AgentState) -> dict:
    reply = await _reply_without_documents(state)
    # 응대 모델이 회사 질문으로 보고 거절하면 판단이 틀린 것이므로 문서 검색으로 넘긴다
    if reply["answer_type"] == "NO_EVIDENCE":
        return {"small_talk": False}
    # 쓰이지 않은 청크를 출처로 보여 주지 않는다. route 값은 서버 계약이 정해질 때까지 늘리지 않는다
    return {
        **reply,
        "context": [],
        "route": "SEARCH",
        "context_probability": None,
        "failed_sources": [],
        "fallback_reason": None,
    }


async def judge_context(state: AgentState) -> dict:
    decision = {"route": "SEARCH", "context_probability": None, "failed_sources": [], "fallback_reason": None}
    context = state.get("context")
    if not context:
        return decision
    # 이전 근거에 필터 밖 문서가 섞이지 않도록 필터가 있으면 재사용하지 않고 새로 검색한다
    if state.get("filters") is not None:
        return decision
    if not config.jev_api_key():
        return {**decision, "fallback_reason": "JEV_API_KEY가 설정되지 않았습니다."}

    # 외부 JEV로 보내기 전에 확인한다. 확인하지 못한 근거는 재사용하지 않는다
    try:
        context = await filter_accessible_context(context, state.get("access_scope"))
    except Exception as exc:
        logger.error("Context revalidation failed: %s", describe_exception(exc))
        return {**decision, "context": [], "fallback_reason": f"근거 재검증 실패 ({type(exc).__name__})"}
    if not context:
        return {**decision, "context": []}
    decision["context"] = context

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


def _follows_clarification(state: AgentState) -> bool:
    history = _history(state)
    return bool(history) and is_clarification(history[-1])


def _search_query(state: AgentState) -> str:
    question = state["messages"][-1].text
    # "숙박비요"처럼 되물음에 답한 말은 그것만으로 검색하면 질문의 뜻을 잃는다
    if not FOLLOW_UP_PATTERN.match(question) and not _follows_clarification(state):
        return question
    previous = next((message.text for message in reversed(_history(state)) if message.type == "human"), None)
    return f"{previous}\n{question}" if previous else question


async def retrieve(state: AgentState) -> dict:
    result = await search(_search_query(state), state.get("filters"), state.get("access_scope"))
    update = {"context": result["chunks"], "failed_sources": result["failed_sources"]}
    if result["chunks"]:
        return update
    alternate = await _alternate_search(state)
    if alternate is None or not alternate[1]["chunks"]:
        return update
    query, found = alternate
    return {"context": found["chunks"], "failed_sources": found["failed_sources"], "alternate_query": query}


async def _reply_without_documents(state: AgentState) -> dict:
    follows = _follows_clarification(state)
    try:
        answer, clarification = await generate_reply_without_documents(state["messages"][-1].text, not follows)
    except AiUnavailableError as exc:
        # 문서가 없는 응대가 실패했다고 오류로 끝내지 않고 이전처럼 거절 문장으로 답한다
        logger.warning("Reply without documents failed: %s", exc)
        answer, clarification = NO_EVIDENCE_ANSWER, False
    # 두 번 연달아 되묻지 않는다. 문서가 없어 답할 수 없으므로 모델이 어기면 거절한다
    if clarification and follows:
        answer, clarification = NO_EVIDENCE_ANSWER, False
    if clarification:
        answer_type = "CLARIFICATION"
    else:
        answer_type = "NO_EVIDENCE" if answer == NO_EVIDENCE_ANSWER else "SMALL_TALK"
    message = AIMessage(content=answer, additional_kwargs={CLARIFICATION_KEY: True} if clarification else {})
    return {"messages": [message], "evidence": [], "answer_type": answer_type}


async def _alternate_search(state: AgentState) -> tuple[str, SearchResult] | None:
    # "리브 팔러시"처럼 표기만 다른 질문을 찾는다. 다른 표기가 없으면 다시 검색하지 않는다
    try:
        query = await rewrite_search_query(_search_query(state))
        if query is None:
            return None
        return query, await search(query, state.get("filters"), state.get("access_scope"))
    except AiUnavailableError as exc:
        logger.warning("Alternate search failed: %s", exc)
        return None


async def _answer_with_documents(state: AgentState, context: list[Chunk], alternate_query: str | None) -> dict:
    question = state["messages"][-1].text
    if alternate_query:
        question = f"{question}\n(같은 뜻의 다른 표기: {alternate_query})"
    response = await generate_answer(question, context, _history(state))
    answer, evidence, clarification = parse_answer(response, context)
    # 두 번 연달아 되묻지 않는다. 모델이 어기면 일반 답변으로 내보낸다
    clarification = clarification and not _follows_clarification(state)
    # 다음 턴의 대화 기록에는 JSON이 아니라 답변만 남긴다
    message = AIMessage(content=answer, additional_kwargs={CLARIFICATION_KEY: True} if clarification else {})
    refused = answer == NO_EVIDENCE_ANSWER
    answer_type = "CLARIFICATION" if clarification else "NO_EVIDENCE" if refused else "ANSWER"
    # 쓰이지 않은 청크를 출처로 보여 주거나 다음 턴에 재사용하지 않는다
    return {
        "messages": [message],
        "evidence": evidence,
        "answer_type": answer_type,
        "context": [] if clarification or refused else context,
    }


async def generate(state: AgentState) -> dict:
    if not state["context"]:
        return await _reply_without_documents(state)
    alternate_query = state.get("alternate_query")
    update = await _answer_with_documents(state, state["context"], alternate_query)
    # 무관한 문서만 찾아 거절했으면 다른 표기로 한 번만 더 찾는다
    if update["answer_type"] != "NO_EVIDENCE" or alternate_query or state["route"] != "SEARCH":
        return update
    alternate = await _alternate_search(state)
    if alternate is None or not alternate[1]["chunks"]:
        return update
    query, result = alternate
    return await _answer_with_documents(state, result["chunks"], query)


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
    graph.add_node("judge_small_talk", _timed("judge_small_talk", lambda state: judge_small_talk(state)))
    graph.add_node("small_talk", _timed("small_talk", lambda state: small_talk(state)))
    graph.add_node("judge_context", _timed("judge_context", lambda state: judge_context(state)))
    graph.add_node("retrieve", _timed("retrieve", lambda state: retrieve(state)))
    graph.add_node("generate", _timed("generate", lambda state: generate(state)))
    graph.add_edge(START, "judge_small_talk")
    graph.add_conditional_edges("judge_small_talk", select_small_talk, ["small_talk", "judge_context"])
    graph.add_conditional_edges("small_talk", select_after_small_talk, ["judge_context", END])
    graph.add_conditional_edges("judge_context", select_route, ["generate", "retrieve"])
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()
