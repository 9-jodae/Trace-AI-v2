import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.schemas import ErrorResponse

logger = logging.getLogger(__name__)


class AiUnavailableError(Exception):
    """검색이나 LLM을 사용할 수 없어 답변을 만들 수 없는 상태."""


async def handle_ai_unavailable(request: Request, exc: AiUnavailableError) -> JSONResponse:
    logger.error("AI unavailable: %s %s: %s", request.method, request.url.path, exc)
    body = ErrorResponse(code="AI_UNAVAILABLE", message="현재 답변을 생성할 수 없습니다.")
    return JSONResponse(status_code=503, content=body.model_dump(by_alias=True))


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error: %s %s", request.method, request.url.path, exc_info=exc)
    body = ErrorResponse(code="INTERNAL_ERROR", message="요청 처리 중 오류가 발생했습니다.")
    return JSONResponse(status_code=500, content=body.model_dump(by_alias=True))


async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    fields = ", ".join(".".join(str(part) for part in error["loc"]) for error in exc.errors())
    body = ErrorResponse(code="INVALID_REQUEST", message=f"요청 값이 올바르지 않습니다: {fields}")
    return JSONResponse(status_code=422, content=body.model_dump(by_alias=True))


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(AiUnavailableError, handle_ai_unavailable)
    app.add_exception_handler(Exception, handle_unexpected_error)
