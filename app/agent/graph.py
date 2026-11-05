import asyncio
import logging
import re
import time
from typing import Annotated, Awaitable, Callable, Literal

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.agent.evidence import (
    EditProposal,
    Evidence,
    Keyword,
    describe_edits,
    parse_answer,
    parse_edit_proposals,
)
from app.core import config
from app.core.errors import AiUnavailableError, describe_exception
from app.core.schemas import AccessScope, SearchFilters
from app.jev.client import can_answer_probability, edit_request_probability, small_talk_probability
from app.llm.client import (
    CLARIFICATION_KEY,
    CONFLICT_SCAN_SEARCH_TERMS,
    NO_EVIDENCE_ANSWER,
    generate_answer,
    generate_reply_without_documents,
    is_clarification,
    rewrite_search_query,
)
from app.retrieval.schemas import Chunk, RegisteredConflict
from app.retrieval.service import (
    KnownConflict,
    SearchResult,
    fetch_known_conflicts,
    filter_accessible_context,
    search,
)

logger = logging.getLogger(__name__)

# 잘못 재사용하면 틀린 답이 나가므로 애매하면 검색하도록 높게 잡는다
CONTEXT_REUSE_THRESHOLD = 0.7
# 문서 질문이 잘못 넘어가면 근거 없이 거절되므로 높게 잡는다
SMALL_TALK_THRESHOLD = 0.9
# 잘못 판단해도 제안만 나가고 문서는 바뀌지 않는다. 일반 질문에 제안이 붙지 않게 높게 잡는다
EDIT_REQUEST_THRESHOLD = 0.8
# 응답 출처와 다음 턴에 남길 근거 수. 답변에 쓴 청크를 먼저 남긴다
SOURCE_LIMIT = 5
# "각 문서 간 충돌 찾아줘"처럼 주제 없이 문서 전체의 충돌을 묻는 말만 찾는다. 낱말이 모두 아래 일반 표현이어야 한다
CONFLICT_WORD_PATTERN = re.compile(r"충돌|모순|불일치|상충|서로 다른 (값|내용)|다르게 (기재|적)|다른 말로")
CONFLICT_SCAN_GENERIC_WORDS = (
    "여기",
    "지금",
    "현재",
    "남은",
    "전체",
    "모든",
    "모두",
    "각",
    "어떤",
    "어느",
    "문서",
    "간",
    "사이",
    "서로",
    "같은",
    "내용",
    "다른",
    "다르게",
    "말로",
    "값",
    "기재",
    "적",
    "발생",
    "충돌",
    "모순",
    "불일치",
    "상충",
    "문제",
    "경고",
    "리뷰",
    "있는지",
    "있어",
    "있나",
    "있는",
    "있을",
    "있습니까",
    "찾아",
    "알려",
    "보여",
    "확인",
    "목록",
    "어디",
    "뭐",
    "무엇",
    "좀",
    "해줘",
    "해",
    "줘",
    "게",
)
# 값을 직접 바꾸라는 말은 판단 모델이 놓쳐도 제안으로 보낸다. "수정 방법이 뭐야" 같은 질문은 걸리지 않는다
EDIT_REQUEST_PATTERN = re.compile(
    r"(수정|변경|반영|통일|해결|교체|정정)\s*(을|를)?\s*(해\s*(줘|주세요|주라|봐|라)|하자|부탁)"
    r"|(바꿔|고쳐|맞춰|바꾸어|고치어)\s*(줘|주세요|주라|봐|라)?\s*(줘|주세요|$|[.!])"
    r"|\d+\s*(일|개|시간|만원|원|%)?\s*(으로|로)\s*(해\s*(줘|주세요)|하자)"
)
HISTORY_LIMIT = 6
# 질문만으로는 검색어가 부족한 후속 질문. "그룹"처럼 단어의 일부인 경우는 제외한다
FOLLOW_UP_PATTERN = re.compile(r"^(그럼|그러면|그건|그게|그거|그것|그 |이건|이게|이거|이 경우|앞서|위 )")


# TODO: 이름과 값은 서버 계약에서 확정 (docs199-Server#76)
AnswerType = Literal["ANSWER", "NO_EVIDENCE", "CLARIFICATION", "SMALL_TALK"]


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    small_talk: bool
    edit_request: bool
    edit_proposals: list[EditProposal]
    alternate_query: str | None
    answer_type: AnswerType
    context: list[Chunk]
    filters: SearchFilters | None
    access_scope: AccessScope | None
    known_conflicts: list[KnownConflict]
    registered_conflicts: list[RegisteredConflict]
    evidence: list[Evidence]
    keywords: list[Keyword]
    route: Literal["CONTEXT", "SEARCH"]
    context_probability: float | None
    failed_sources: list[str]
    fallback_reason: str | None


def _history(state: AgentState) -> list[AnyMessage]:
    return state["messages"][:-1][-HISTORY_LIMIT:]


async def judge_small_talk(state: AgentState) -> dict:
    # "네"처럼 되물음에 답한 말은 일상 대화로 보내지 않는다
    question = state["messages"][-1].text
    if _follows_clarification(state):
        return {"small_talk": False, "edit_request": False, "edit_proposals": []}
    pattern_match = EDIT_REQUEST_PATTERN.search(question) is not None
    if not config.jev_api_key():
        return {"small_talk": False, "edit_request": pattern_match, "edit_proposals": []}
    small_talk_result, edit_result = await asyncio.gather(
        small_talk_probability(question), edit_request_probability(question), return_exceptions=True
    )
    # 판단하지 못하면 문서 검색으로 진행한다
    if isinstance(small_talk_result, Exception):
        logger.error("Small talk judge failed: %s", describe_exception(small_talk_result))
    if isinstance(edit_result, Exception):
        logger.error("Edit request judge failed: %s", describe_exception(edit_result))
    edit_request = pattern_match or (not isinstance(edit_result, Exception) and edit_result >= EDIT_REQUEST_THRESHOLD)
    return {
        # 수정 요청은 문서를 찾아야 하므로 일상 대화로 보내지 않는다
        "small_talk": not edit_request
        and not _is_conflict_scan(question)
        and not isinstance(small_talk_result, Exception)
        and small_talk_result >= SMALL_TALK_THRESHOLD,
        "edit_request": edit_request,
        "edit_proposals": [],
    }


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
    # 이전 근거로 답하면 다른 문서의 충돌을 볼 수 없으므로 새로 검색한다
    if _is_conflict_scan(state["messages"][-1].text):
        return decision
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


def _is_conflict_scan(question: str) -> bool:
    if not CONFLICT_WORD_PATTERN.search(question):
        return False
    # 일반 표현 밖의 낱말이 있으면 "연차 이월 문서에 충돌 있어?"처럼 주제가 있는 질문이다
    words = re.findall(r"[\w]+", question)
    return all(word.startswith(CONFLICT_SCAN_GENERIC_WORDS) for word in words)


def _search_query(state: AgentState) -> str:
    question = state["messages"][-1].text
    # 질문 문장만으로는 어떤 문서도 가깝지 않으므로 불일치를 밝히는 표현으로 찾는다
    if _is_conflict_scan(question):
        return f"{question}\n{CONFLICT_SCAN_SEARCH_TERMS}"
    # "숙박비요"처럼 되물음에 답한 말은 그것만으로 검색하면 질문의 뜻을 잃는다
    # "최대 5일로 수정해줘"도 앞선 질문 없이는 어떤 문서인지 알 수 없다
    if not FOLLOW_UP_PATTERN.match(question) and not _follows_clarification(state) and not state.get("edit_request"):
        return question
    previous = next((message.text for message in reversed(_history(state)) if message.type == "human"), None)
    return f"{previous}\n{question}" if previous else question


async def retrieve(state: AgentState) -> dict:
    update = await _with_previous_context(state, await _search_documents(state))
    known = state.get("known_conflicts") or []
    if not _is_conflict_scan(state["messages"][-1].text):
        return update
    found = (
        await fetch_known_conflicts(known, state.get("filters"), state.get("access_scope"))
        if known
        else {"chunks": [], "conflicts": []}
    )
    # Spring이 보낸 수와 근거로 쓴 수만 남긴다. 항목과 문서는 남기지 않는다
    logger.info(
        "Known conflicts: received=%d used=%d filtered=%s",
        len(known),
        len(found["conflicts"]),
        state.get("filters") is not None,
    )
    if not known:
        return update
    # 등록 충돌의 근거를 앞에 두고, 같은 청크는 한 번만 넣는다
    present = {chunk["id"] for chunk in found["chunks"]}
    others = [chunk for chunk in update["context"] if chunk["id"] not in present]
    return {**update, "context": [*found["chunks"], *others], "registered_conflicts": found["conflicts"]}


async def _with_previous_context(state: AgentState, update: dict) -> dict:
    """수정 요청은 직전 답변의 대상을 가리키므로, 직전 근거 중 아직 볼 수 있는 청크를 앞에 둔다."""
    previous = state.get("context") or []
    # 필터 밖 문서가 섞이지 않도록 필터가 있으면 새 검색 결과만 쓴다
    if not state.get("edit_request") or not previous or state.get("filters") is not None:
        return update
    try:
        kept = await filter_accessible_context(previous, state.get("access_scope"))
    except Exception as exc:
        logger.error("Context revalidation failed: %s", describe_exception(exc))
        return update
    present = {chunk["id"] for chunk in kept}
    return {**update, "context": [*kept, *(chunk for chunk in update["context"] if chunk["id"] not in present)]}


async def _search_documents(state: AgentState) -> dict:
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
    return {"messages": [message], "evidence": [], "keywords": [], "answer_type": answer_type}


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
    original = state["messages"][-1].text
    question = f"{original}\n(같은 뜻의 다른 표기: {alternate_query})" if alternate_query else original
    edit_request = state.get("edit_request", False)
    extra = {"edit_request": True} if edit_request else {}
    if _is_conflict_scan(original):
        extra["conflict_scan"] = True
    registered = state.get("registered_conflicts") or []
    if registered:
        extra["registered_conflicts"] = registered
    known = state.get("known_conflicts") or []
    if extra.get("conflict_scan") and len(registered) < len(known):
        # 필터나 상한 때문에 빠진 등록 충돌이 있으면 "찾지 못했다"로만 끝나지 않게 알린다
        extra["skipped_conflicts"] = len(known) - len(registered)
        extra["filters_applied"] = state.get("filters") is not None
    response = await generate_answer(question, context, _history(state), **extra)
    try:
        answer, evidence, clarification, keywords = parse_answer(response, context)
    except AiUnavailableError:
        # 호출마다 다르게 생성되므로 형식이 깨졌을 때만 한 번 더 만든다. 또 깨지면 오류로 끝낸다
        logger.warning("LLM answer retry: malformed response")
        response = await generate_answer(question, context, _history(state), **extra)
        answer, evidence, clarification, keywords = parse_answer(response, context)
    # 두 번 연달아 되묻지 않는다. 모델이 어기면 일반 답변으로 내보낸다
    clarification = clarification and not _follows_clarification(state)
    # 다음 턴의 대화 기록에는 JSON이 아니라 답변만 남긴다
    message = AIMessage(content=answer, additional_kwargs={CLARIFICATION_KEY: True} if clarification else {})
    refused = answer == NO_EVIDENCE_ANSWER
    answer_type = "CLARIFICATION" if clarification else "NO_EVIDENCE" if refused else "ANSWER"
    refuse_or_ask = clarification or refused
    edits = parse_edit_proposals(response, context) if edit_request and not refuse_or_ask else []
    if edit_request and not refuse_or_ask:
        # 원문과 맞춰 본 제안만 답변에 붙여, 화면에 카드가 없어도 바꿀 문장을 바로 볼 수 있게 한다
        answer = f"{answer}\n\n{describe_edits(edits)}"
        message = AIMessage(content=answer)
    # 쓰이지 않은 청크를 출처로 보여 주거나 다음 턴에 재사용하지 않는다
    return {
        "messages": [message],
        "evidence": evidence,
        "keywords": keywords,
        "edit_proposals": edits,
        "answer_type": answer_type,
        "context": [] if refuse_or_ask else _cited_first(context, evidence, keywords, edits),
    }


def _cited_first(
    context: list[Chunk], evidence: list[Evidence], keywords: list[Keyword], edits: list[EditProposal]
) -> list[Chunk]:
    cited = {item["chunk_id"] for item in [*evidence, *keywords, *edits]}
    ordered = [chunk for chunk in context if chunk["id"] in cited] + [
        chunk for chunk in context if chunk["id"] not in cited
    ]
    # 근거 인용은 출처에 있어야 화면에서 이어지므로, 인용한 청크가 많으면 그만큼은 남긴다
    return ordered[: max(SOURCE_LIMIT, len(cited & {chunk["id"] for chunk in context}))]


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
