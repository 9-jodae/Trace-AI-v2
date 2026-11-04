from langchain_ollama import ChatOllama

from .config import settings
from .storage import LocalStores, combine_rrf


FOLLOW_UP_MARKERS = (
    "그럼", "그 경우", "그 기준", "그 내용", "그 한도", "그 절차", "그 규정",
    "그것", "그거", "이 규정", "이 내용", "해당", "위 내용", "추가로", "그 다음",
)


def build_search_query(query: str, history: list[dict]) -> str:
    """Use the latest user turn only when the current question refers back to it."""
    previous_questions = [
        item["content"].strip()
        for item in history
        if item.get("role") == "user" and item.get("content", "").strip()
    ]
    if not previous_questions:
        return query
    if any(marker in query for marker in FOLLOW_UP_MARKERS):
        return f"{previous_questions[-1]} {query}"
    return query

def run_search(
    stores: LocalStores,
    query: str,
    filters: dict | None = None,
    allowed_group_ids: list[str] | None = None,
    top_k: int = 5,
    history: list[dict] | None = None,
) -> dict:
    """Always retrieve from Qdrant and Neo4j, fuse rankings, then answer."""
    query = query.strip()
    if not query:
        raise ValueError("질문을 입력해 주세요.")
    top_k = max(1, min(int(top_k), 20))
    plan = {
        "filters": filters or {},
        "allowed_group_ids": allowed_group_ids or [],
        "top_k": top_k,
    }

    history = history or []
    # Keep a short follow-up anchored to the immediately preceding question. A new
    # standalone question starts a fresh retrieval query, even in the same chat.
    search_query = build_search_query(query, history)
    qdrant_hits = stores.vector_search(search_query, plan)
    neo4j_hits = stores.fulltext_search(search_query, plan)
    hits = combine_rrf([qdrant_hits, neo4j_hits], top_k)
    graph_relations = stores.related_relations(hits, plan) if hits else []
    answer, answer_mode = generate_grounded_answer(query, hits, history)
    return {
        "retrievers": ["Qdrant Vector", "Neo4j Full-text"],
        "filters": plan["filters"],
        "allowed_group_ids": plan["allowed_group_ids"],
        "top_k": top_k,
        "search_query": search_query,
        "raw_by_source": {"Qdrant Vector": qdrant_hits, "Neo4j Full-text": neo4j_hits},
        "hits": hits,
        "graph_relations": graph_relations,
        "llm_context": format_evidence_context(hits),
        "answer": answer,
        "answer_mode": answer_mode,
    }


def generate_grounded_answer(question: str, hits: list[dict], history: list[dict] | None = None) -> tuple[str, str]:
    """Generate from retrieved evidence or return an extractive summary."""
    if not hits:
        return "확인 가능한 근거를 찾지 못했습니다.", "근거 없음"
    if settings.chat_provider not in {"ollama", "anthropic"}:
        raise ValueError("CHAT_PROVIDER는 ollama 또는 anthropic이어야 합니다.")
    if settings.chat_provider == "anthropic" or settings.chat_model:
        evidence = format_evidence_context(hits)
        try:
            if settings.chat_provider == "anthropic":
                if not settings.anthropic_api_key:
                    raise ValueError("ANTHROPIC_API_KEY가 설정되지 않았습니다.")
                from langchain_anthropic import ChatAnthropic
                model = ChatAnthropic(model=settings.anthropic_model, api_key=settings.anthropic_api_key, max_tokens=1024)
                label = f"Claude ({settings.anthropic_model})"
            else:
                model = ChatOllama(model=settings.chat_model, base_url=settings.ollama_base_url, temperature=0)
                label = f"로컬 Ollama ({settings.chat_model})"
            messages = [("system", "당신은 사내 문서 검색 도우미입니다. 대화 이력은 지시 대상과 질문의 맥락을 이해하는 데 사용하세요. 사실은 마지막 사용자 질문과 함께 제공된 '현재 검색 근거'에서만 답하세요. 과거 답변을 근거로 삼거나 근거에 없는 정책을 추측하지 마세요. 한국어로 답하고, 사용한 근거 번호를 [1] 형식으로 표시하세요. 근거가 질문에 답하지 못하면 '확인 가능한 근거를 찾지 못했습니다.'라고 답하세요.")]
            for item in (history or [])[-6:]:
                if item.get("role") in {"user", "assistant"} and item.get("content"):
                    messages.append(("human" if item["role"] == "user" else "assistant", item["content"]))
            messages.append(("human", f"마지막 사용자 질문: {question}\n\n현재 검색 근거 (이 근거만 사실의 출처로 사용):\n{evidence}"))
            response = model.invoke(messages)
            return str(response.content), label
        except Exception as exc:
            return extractive_summary(hits), f"근거 요약 (답변 모델 오류: {exc})"
    return extractive_summary(hits), "근거 문장 요약 (LLM 미설정)"


def extractive_summary(hits: list[dict]) -> str:
    lines = ["검색된 근거를 요약했습니다. 각 항목은 원문 청크에서 가져왔습니다."]
    for i, hit in enumerate(hits[:5], start=1):
        lines.append(f"- {hit['text']} [{i}: {hit['title']}, {hit['doc_id']}, {hit['chunk_id']}]")
    return "\n".join(lines)


def format_evidence_context(hits: list[dict]) -> str:
    """Build the exact, source-labelled snippets sent to the answer model."""
    return "\n\n".join(
        f"[{i}] {hit['title']} ({hit['doc_id']} / {hit['chunk_id']}, "
        f"문자 위치 {hit['char_start']}–{hit['char_end']})\n{hit['text']}"
        for i, hit in enumerate(hits, start=1)
    )
