import re

from typesafe_sdk import AsyncTypeSafeClient, Noul

from app.core import config
from app.retrieval.schemas import Chunk

TIMEOUT_SECONDS = 10.0

SECTION_MARKER_PATTERN = re.compile(r"\[\s*(이전 대화|검색 컨텍스트|사용자 질문)\s*\]")

CAN_ANSWER = Noul(
    instructions="이전 대화와 검색 컨텍스트에 적힌 내용만으로 사용자 질문에 정확히 답할 수 있다",
)

SMALL_TALK = Noul(
    instructions="사용자의 말이 문서 내용을 묻지 않는 인사, 감사, 안부, 도우미 소개 요청 같은 일상 대화다",
)

_client: AsyncTypeSafeClient | None = None
_client_settings: tuple[str | None, str] | None = None


def _neutralize_markers(text: str) -> str:
    # 본문이 구획 제목을 흉내 내 판단 대상을 바꾸지 못하게 한다
    return SECTION_MARKER_PATTERN.sub(r"(\1)", text)


def build_state(question: str, context: list[Chunk], history: list[str]) -> str:
    chunks = "\n".join(f"- {_neutralize_markers(chunk['text'])}" for chunk in context)
    conversation = _neutralize_markers("\n".join(history)) or "없음"
    return (
        f"[이전 대화]\n{conversation}\n\n[검색 컨텍스트]\n{chunks}\n\n"
        f"[사용자 질문]\n{_neutralize_markers(question)}"
    )


async def _get_client() -> AsyncTypeSafeClient:
    # 연결을 재사용하도록 클라이언트를 프로세스에서 하나만 유지한다
    global _client, _client_settings
    settings = (config.jev_api_key(), config.jev_model())
    if _client is None or _client_settings != settings:
        await close_client()
        _client = AsyncTypeSafeClient(api_key=settings[0], model=settings[1], timeout=TIMEOUT_SECONDS)
        _client_settings = settings
    return _client


async def close_client() -> None:
    global _client, _client_settings
    if _client is not None:
        await _client.aclose()
    _client = None
    _client_settings = None


async def can_answer_probability(question: str, context: list[Chunk], history: list[str]) -> float:
    client = await _get_client()
    response = await client.system_one(
        state=build_state(question, context, history),
        questions={"can_answer": CAN_ANSWER},
    )
    return response.nouls["can_answer"].noul


async def small_talk_probability(question: str) -> float:
    client = await _get_client()
    response = await client.system_one(
        state=f"[사용자 질문]\n{_neutralize_markers(question)}",
        questions={"small_talk": SMALL_TALK},
    )
    return response.nouls["small_talk"].noul
