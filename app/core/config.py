import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_JEV_MODEL = "jev-latest"
DEFAULT_EMBEDDING_DIMENSIONS = 1024
DEFAULT_MAX_CONVERSATIONS = 1000
DEFAULT_CONVERSATION_TTL_SECONDS = 3600
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60


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


def allowed_group_ids() -> list[str]:
    # 운영에서는 사용자별 권한을 요청으로 받아야 하므로 환경 변수로 열어 주지 않는다 (AccessScope 계약 대기)
    if is_production():
        return []
    return [group.strip() for group in os.getenv("ALLOWED_GROUP_IDS", "").split(",") if group.strip()]


def vector_store() -> str | None:
    return os.getenv("VECTOR_STORE") or None


def qdrant_url() -> str | None:
    return os.getenv("QDRANT_URL") or None


def qdrant_collection() -> str:
    return os.getenv("QDRANT_COLLECTION") or "internal_chunks"


def neo4j_uri() -> str | None:
    return os.getenv("NEO4J_URI") or None


def neo4j_user() -> str:
    return os.getenv("NEO4J_USER") or "neo4j"


def neo4j_password() -> str | None:
    return os.getenv("NEO4J_PASSWORD") or None


def neo4j_database() -> str | None:
    return os.getenv("NEO4J_DATABASE") or None


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


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name) or default)
    if value <= 0:
        raise ValueError(f"{name}은 1 이상이어야 합니다.")
    return value


def max_conversations() -> int:
    return _positive_int("MAX_CONVERSATIONS", DEFAULT_MAX_CONVERSATIONS)


def conversation_ttl_seconds() -> int:
    return _positive_int("CONVERSATION_TTL_SECONDS", DEFAULT_CONVERSATION_TTL_SECONDS)


def request_timeout_seconds() -> int:
    return _positive_int("REQUEST_TIMEOUT_SECONDS", DEFAULT_REQUEST_TIMEOUT_SECONDS)


def extraction_max_text_chars() -> int:
    return _positive_int("EXTRACTION_MAX_TEXT_CHARS", 24_000)


def ingestion_timeout_seconds() -> int:
    return _positive_int("INGESTION_TIMEOUT_SECONDS", 180)


def database_url() -> str | None:
    return os.getenv("DATABASE_URL") or None
