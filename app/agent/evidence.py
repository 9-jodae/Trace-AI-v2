import json
import logging
import re
from typing import Annotated, Any, NamedTuple

from pydantic import BaseModel, StrictInt, StringConstraints, ValidationError
from typing_extensions import TypedDict

from app.core.errors import AiUnavailableError
from app.llm.client import NO_EVIDENCE_ANSWER
from app.retrieval.schemas import Chunk

logger = logging.getLogger(__name__)

MAX_EVIDENCE = 3
MIN_CLARIFICATION_OPTIONS = 2
MAX_KEYWORDS = 3
MAX_KEYWORD_LENGTH = 50
MAX_EDITS = 8
MAX_EDIT_LENGTH = 1000
ANSWER_OBJECT_PATTERN = re.compile(r'\{\s*"answer"\s*:')
JSON_START_PATTERN = re.compile(r"\s*(```(?:json)?\s*)?\{")


class EvidenceCandidate(BaseModel):
    chunk: StrictInt
    quote: str


class KeywordCandidate(BaseModel):
    chunk: StrictInt
    text: str


class OptionCandidate(BaseModel):
    label: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    chunk: StrictInt


class AnswerCandidate(BaseModel):
    answer: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    evidence: Any = None
    keywords: Any = None
    options: Any = None
    edits: Any = None


class Keyword(TypedDict):
    chunk_id: str
    text: str


class EditProposal(TypedDict):
    chunk_id: str
    source: str | None
    document_id: str | None
    version_id: str | None
    before: str
    after: str
    reason: str
    char_start: int | None
    char_end: int | None


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


def ground_keywords(raw: Any, context: list[Chunk]) -> list[Keyword]:
    # 인용과 같이 원문에 있는 문구만 내보낸다. 하나가 형식을 어겨도 나머지는 살린다
    keywords: list[Keyword] = []
    seen: set[str] = set()
    for item in raw if isinstance(raw, list) else []:
        try:
            candidate = KeywordCandidate.model_validate(item)
        except ValidationError:
            continue
        text = candidate.text.strip()
        if not text or len(text) > MAX_KEYWORD_LENGTH or text in seen or not 1 <= candidate.chunk <= len(context):
            continue
        chunk = context[candidate.chunk - 1]
        if text in chunk["text"]:
            seen.add(text)
            keywords.append({"chunk_id": chunk["id"], "text": text})
    return keywords[:MAX_KEYWORDS]


class ParsedAnswer(NamedTuple):
    answer: str
    evidence: list[Evidence]
    clarification: bool
    keywords: list[Keyword]


def _is_valid_clarification(raw: Any, context: list[Chunk]) -> bool:
    # 선택지가 실제 검색 결과의 서로 다른 청크를 가리킬 때만 되묻는 답변으로 인정한다
    chunks = set()
    for item in raw if isinstance(raw, list) else []:
        try:
            option = OptionCandidate.model_validate(item)
        except ValidationError:
            continue
        if 1 <= option.chunk <= len(context):
            chunks.add(option.chunk)
    return len(chunks) >= MIN_CLARIFICATION_OPTIONS


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


def parse_answer(response: str, context: list[Chunk]) -> ParsedAnswer:
    candidate = _load_candidate(response)
    if candidate is None:
        # 모델이 형식을 무시하고 글로 답하면 근거만 빼고 답변은 살린다. 응답 본문은 기록하지 않는다
        logger.warning("LLM answer is not in the expected format")
        return ParsedAnswer(response, [], False, [])
    evidence = ground_evidence(_valid_candidates(candidate.evidence), context)
    # 근거 없다고 답하면서 문의처 같은 추측을 덧붙이지 못하게 고정 문장으로 돌린다
    if NO_EVIDENCE_ANSWER in candidate.answer and not evidence:
        return ParsedAnswer(NO_EVIDENCE_ANSWER, [], False, [])
    if _is_valid_clarification(candidate.options, context):
        return ParsedAnswer(candidate.answer, [], True, [])
    return ParsedAnswer(candidate.answer, evidence, False, ground_keywords(candidate.keywords, context))


class EditCandidate(BaseModel):
    chunk: StrictInt
    before: str
    after: str
    reason: str = ""


def ground_edits(raw: Any, context: list[Chunk]) -> list[EditProposal]:
    # 원문에 한 번만 있는 문장만 제안한다. 두 곳 이상이면 어디를 고칠지 알 수 없다. 위치는 원문에서 직접 계산한다
    edits: list[EditProposal] = []
    seen: set[tuple[str, str]] = set()
    for item in raw if isinstance(raw, list) else []:
        try:
            candidate = EditCandidate.model_validate(item)
        except ValidationError:
            continue
        before, after = candidate.before.strip(), candidate.after.strip()
        if not before or not after or before == after or max(len(before), len(after)) > MAX_EDIT_LENGTH:
            continue
        if not 1 <= candidate.chunk <= len(context):
            continue
        chunk = context[candidate.chunk - 1]
        if chunk["text"].count(before) != 1 or (chunk["id"], before) in seen:
            continue
        seen.add((chunk["id"], before))
        position = chunk["text"].index(before)
        base = chunk.get("char_start")
        edits.append(
            {
                "chunk_id": chunk["id"],
                "source": chunk.get("source"),
                "document_id": chunk.get("document_id"),
                "version_id": chunk.get("version_id"),
                "before": before,
                "after": after,
                "reason": candidate.reason.strip()[:MAX_EDIT_LENGTH],
                "char_start": None if base is None else base + position,
                "char_end": None if base is None else base + position + len(before),
            }
        )
    return edits[:MAX_EDITS]


NO_GROUNDED_EDITS = (
    "원문과 정확히 일치하는 문장을 찾지 못해 수정안을 만들지 못했습니다. 바꿀 문장이 있는 문서 이름을 함께 알려 주세요."
)


def describe_edits(edits: list[EditProposal]) -> str:
    """답변 아래에 붙일 수정안. 문서는 바꾸지 않았다는 것을 함께 밝힌다."""
    if not edits:
        return NO_GROUNDED_EDITS
    lines = [
        f"- 「{edit['source'] or '출처 미상'}」\n  - 현재: {edit['before']}\n  - 수정안: {edit['after']}"
        for edit in edits
    ]
    return "**수정 제안** (문서는 아직 바뀌지 않았습니다)\n" + "\n".join(lines)


def parse_edit_proposals(response: str, context: list[Chunk]) -> list[EditProposal]:
    candidate = _load_candidate(response)
    return ground_edits(candidate.edits, context) if candidate else []
