from typesafe_sdk import AsyncTypeSafeClient, Noul

from app.core import config
from app.retrieval.schemas import Chunk

TIMEOUT_SECONDS = 10.0

CAN_ANSWER = Noul(
    instructions="이전 대화와 검색 컨텍스트에 적힌 내용만으로 사용자 질문에 정확히 답할 수 있다",
)


def build_state(question: str, context: list[Chunk], history: list[str]) -> str:
    chunks = "\n".join(f"- {chunk['text']}" for chunk in context)
    conversation = "\n".join(history) or "없음"
    return f"[이전 대화]\n{conversation}\n\n[검색 컨텍스트]\n{chunks}\n\n[사용자 질문]\n{question}"


async def can_answer_probability(question: str, context: list[Chunk], history: list[str]) -> float:
    async with AsyncTypeSafeClient(
        api_key=config.jev_api_key(),
        model=config.jev_model(),
        timeout=TIMEOUT_SECONDS,
    ) as client:
        response = await client.system_one(
            state=build_state(question, context, history),
            questions={"can_answer": CAN_ANSWER},
        )
    return response.nouls["can_answer"].noul
