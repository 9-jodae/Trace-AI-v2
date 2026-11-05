import json
import logging
import re
from typing import Annotated, Any

from pydantic import BaseModel, StrictInt, StringConstraints, ValidationError
from typing_extensions import TypedDict

from app.core.errors import AiUnavailableError
from app.llm.client import NO_EVIDENCE_ANSWER
from app.retrieval.schemas import Chunk

logger = logging.getLogger(__name__)

MAX_EVIDENCE = 3
ANSWER_OBJECT_PATTERN = re.compile(r'\{\s*"answer"\s*:')
JSON_START_PATTERN = re.compile(r"\s*(```(?:json)?\s*)?\{")


class EvidenceCandidate(BaseModel):
    chunk: StrictInt
    quote: str


class AnswerCandidate(BaseModel):
    answer: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    evidence: Any = None


class Evidence(TypedDict):
    chunk_id: str
    source: str | None
    document_id: str | None
    version_id: str | None
    quote: str
    char_start: int | None
    char_end: int | None


def _locate(chunk: Chunk, quote: str) -> Evidence | None:
    # 위치는 모델의 숫자를 믿지 않고 원문에서 직접 계산한다. 원문에 없는 문구는 근거로 내보내지 않는다
    quote = quote.strip()
    position = chunk["text"].find(quote) if quote else -1
    if position < 0:
        return None
    base = chunk.get("char_start")
    return {
        "chunk_id": chunk["id"],
        "source": chunk.get("source"),
        "document_id": chunk.get("document_id"),
        "version_id": chunk.get("version_id"),
        "quote": quote,
        "char_start": None if base is None else base + position,
        "char_end": None if base is None else base + position + len(quote),
    }


def ground_evidence(candidates: list[EvidenceCandidate], context: list[Chunk]) -> list[Evidence]:
    grounded: list[Evidence] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        # 청크 번호는 프롬프트에 적은 1부터 시작하는 번호다
        if not 1 <= candidate.chunk <= len(context):
            continue
        evidence = _locate(context[candidate.chunk - 1], candidate.quote)
        if evidence is None or (evidence["chunk_id"], evidence["quote"]) in seen:
            continue
        seen.add((evidence["chunk_id"], evidence["quote"]))
        grounded.append(evidence)
    return grounded[:MAX_EVIDENCE]


def _valid_candidates(raw: Any) -> list[EvidenceCandidate]:
    # 인용 하나가 형식을 어겨도 답변과 나머지 인용은 살린다
    candidates = []
    for item in raw if isinstance(raw, list) else []:
        try:
            candidates.append(EvidenceCandidate.model_validate(item))
        except ValidationError:
            continue
    return candidates


def _load_candidate(text: str) -> AnswerCandidate | None:
    # 모델이 형식을 어기고 설명이나 코드 블록을 덧붙여도 JSON 답변을 찾아 쓴다. JSON이 없으면 None
    match = ANSWER_OBJECT_PATTERN.search(text)
    if match is None and not JSON_START_PATTERN.match(text):
        return None
    try:
        if match is None:
            raise ValueError("answer 없음")
        # 답변 문자열 안의 실제 줄바꿈은 허용한다
        data, _ = json.JSONDecoder(strict=False).raw_decode(text, match.start())
        return AnswerCandidate.model_validate(data)
    except ValueError as exc:
        # 잘린 JSON이나 빈 답변을 사용자에게 그대로 보여 주지 않는다. 응답 본문은 기록하지 않는다
        reason = exc.errors()[0]["type"] if isinstance(exc, ValidationError) else type(exc).__name__
        logger.warning("LLM answer parse failed: %s length=%d", reason, len(text))
        raise AiUnavailableError("LLM 응답 형식이 올바르지 않습니다.") from exc


def parse_answer(response: str, context: list[Chunk]) -> tuple[str, list[Evidence]]:
    candidate = _load_candidate(response)
    if candidate is None:
        # 모델이 형식을 무시하고 글로 답하면 근거만 빼고 답변은 살린다. 응답 본문은 기록하지 않는다
        logger.warning("LLM answer is not in the expected format")
        return response, []
    evidence = ground_evidence(_valid_candidates(candidate.evidence), context)
    # 근거 없다고 답하면서 문의처 같은 추측을 덧붙이지 못하게 고정 문장으로 돌린다
    if NO_EVIDENCE_ANSWER in candidate.answer and not evidence:
        return NO_EVIDENCE_ANSWER, []
    return candidate.answer, evidence
