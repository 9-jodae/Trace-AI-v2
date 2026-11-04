import logging
import traceback
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.core.schemas import ErrorResponse

logger = logging.getLogger(__name__)

HTTP_ERRORS = {
    401: ("UNAUTHORIZED", "인증에 실패했습니다."),
    404: ("NOT_FOUND", "요청한 경로를 찾을 수 없습니다."),
    405: ("METHOD_NOT_ALLOWED", "허용되지 않은 메서드입니다."),
}


def describe_exception(exc: BaseException) -> str:
    """외부 오류 원문에는 질문이나 문서 본문이 섞일 수 있어 유형, 상태 코드, 발생 위치만 남긴다."""
    status = next(
        (value for name in ("status_code", "code") if isinstance(value := getattr(exc, name, None), int)),
        None,
    )
    frames = traceback.extract_tb(exc.__traceback__)
    location = f" at {Path(frames[-1].filename).name}:{frames[-1].lineno}" if frames else ""
    return f"{type(exc).__name__}{f' ({status})' if status else ''}{location}"


class AiUnavailableError(Exception):
    """검색이나 LLM을 사용할 수 없어 답변을 만들 수 없는 상태."""


async def handle_ai_unavailable(request: Request, exc: AiUnavailableError) -> JSONResponse:
    logger.error("AI unavailable: %s %s: %s", request.method, request.url.path, exc)
    body = ErrorResponse(code="AI_UNAVAILABLE", message="현재 답변을 생성할 수 없습니다.")
    return JSONResponse(status_code=503, content=body.model_dump(by_alias=True))


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled error: %s %s: %s", request.method, request.url.path, describe_exception(exc))
    body = ErrorResponse(code="INTERNAL_ERROR", message="요청 처리 중 오류가 발생했습니다.")
    return JSONResponse(status_code=500, content=body.model_dump(by_alias=True))


async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    fields = ", ".join(".".join(str(part) for part in error["loc"]) for error in exc.errors())
    body = ErrorResponse(code="INVALID_REQUEST", message=f"요청 값이 올바르지 않습니다: {fields}")
    return JSONResponse(status_code=422, content=body.model_dump(by_alias=True))


async def handle_http_error(request: Request, exc: HTTPException) -> JSONResponse:
    code, message = HTTP_ERRORS.get(exc.status_code, ("HTTP_ERROR", "요청을 처리할 수 없습니다."))
    body = ErrorResponse(code=code, message=message)
    return JSONResponse(status_code=exc.status_code, content=body.model_dump(by_alias=True), headers=exc.headers)


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(HTTPException, handle_http_error)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(AiUnavailableError, handle_ai_unavailable)
    app.add_exception_handler(Exception, handle_unexpected_error)
