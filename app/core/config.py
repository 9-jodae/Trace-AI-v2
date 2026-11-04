import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_JEV_MODEL = "jev-latest"
DEFAULT_EMBEDDING_DIMENSIONS = 1024


def is_production() -> bool:
    # 설정이 빠져도 인증과 샘플 차단이 꺼지지 않도록 local을 명시할 때만 개발 모드로 본다
    return os.getenv("APP_ENV") != "local"


def server_options() -> dict:
    # 로컬은 내 PC에서만 접속하게 하고, 운영은 Compose 네트워크의 Spring이 접속할 수 있게 연다
    default_host = "0.0.0.0" if is_production() else "127.0.0.1"
    return {
        "host": os.getenv("HOST") or default_host,
        "port": int(os.getenv("PORT") or 8000),
        "reload": not is_production(),
    }


def internal_secret() -> str | None:
    return os.getenv("INTERNAL_SECRET") or None


def use_sample_data() -> bool:
    return not is_production() and os.getenv("USE_SAMPLE_DATA") == "true"


def validate_startup() -> None:
    if not is_production():
        return
    if not internal_secret():
        raise RuntimeError("운영 환경에는 INTERNAL_SECRET이 필요합니다. 로컬 개발이면 APP_ENV=local을 설정하세요.")
    if os.getenv("USE_SAMPLE_DATA") == "true":
        raise RuntimeError("운영 환경에서는 USE_SAMPLE_DATA를 사용할 수 없습니다.")


def uses_aws_credentials(model: str) -> bool:
    return model.startswith("bedrock")


def jev_api_key() -> str | None:
    return os.getenv("JEV_API_KEY") or None


def jev_model() -> str:
    return os.getenv("JEV_MODEL") or DEFAULT_JEV_MODEL


def embedding_api_key() -> str | None:
    return os.getenv("EMBEDDING_API_KEY") or None


def embedding_model() -> str | None:
    return os.getenv("EMBEDDING_MODEL") or None


def embedding_dimensions() -> int:
    return int(os.getenv("EMBEDDING_DIMENSIONS") or DEFAULT_EMBEDDING_DIMENSIONS)


def llm_api_key() -> str | None:
    return os.getenv("LLM_API_KEY") or None


def llm_model() -> str | None:
    return os.getenv("LLM_MODEL") or None


def llm_fallback_model() -> str | None:
    return os.getenv("LLM_FALLBACK_MODEL") or None


def database_url() -> str | None:
    return os.getenv("DATABASE_URL") or None
