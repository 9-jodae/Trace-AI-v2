import logging
import re

from pydantic import BaseModel, Field, StrictInt, ValidationError
from typing_extensions import TypedDict

from app.retrieval.schemas import Chunk

logger = logging.getLogger(__name__)

MAX_EVIDENCE = 3
CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class EvidenceCandidate(BaseModel):
    chunk: StrictInt
    quote: str


class AnswerCandidate(BaseModel):
    answer: str = Field(min_length=1)
    evidence: list[EvidenceCandidate] = Field(default_factory=list)


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
    position = chunk["text"].find(quote) if quote.strip() else -1
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
        if evidence is None or (evidence["chunk_id"], candidate.quote) in seen:
            continue
        seen.add((evidence["chunk_id"], candidate.quote))
        grounded.append(evidence)
    return grounded[:MAX_EVIDENCE]


def parse_answer(response: str, context: list[Chunk]) -> tuple[str, list[Evidence]]:
    text = response.strip()
    fenced = CODE_FENCE_PATTERN.match(text)
    try:
        candidate = AnswerCandidate.model_validate_json(fenced.group(1) if fenced else text)
    except ValidationError:
        # 근거를 못 읽어도 답변은 살린다. 응답 본문은 기록하지 않는다
        logger.warning("LLM answer is not in the expected format")
        return response, []
    return candidate.answer, ground_evidence(candidate.evidence, context)
