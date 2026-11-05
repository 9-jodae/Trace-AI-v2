import os
from pathlib import Path

from dotenv import load_dotenv

from app.core.schemas import AccessScope

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_JEV_MODEL = "jev-latest"
DEFAULT_EMBEDDING_DIMENSIONS = 1024
DEFAULT_MAX_CONVERSATIONS = 1000
DEFAULT_CONVERSATION_TTL_SECONDS = 3600
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60
DEFAULT_DB_POOL_MAX_SIZE = 10
# 시연 문서 18개와 질문 15개로 측정한 값. 관련 질문의 정답은 0.82 이하, 무관 질문은 대부분 0.85 이상이었다
DEFAULT_VECTOR_MAX_COSINE_DISTANCE = 0.85


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


def access_scope() -> AccessScope | None:
    # 요청에 accessScope가 없을 때 쓰는 로컬 시험용 값. 운영에서는 환경 변수로 권한을 열어 주지 않는다
    if is_production():
        return None
    is_admin = os.getenv("ACCESS_IS_ADMIN") == "true"
    clearance = os.getenv("ACCESS_CLEARANCE") or None
    if not is_admin and not clearance:
        return None
    department_id = os.getenv("ACCESS_DEPARTMENT_ID") or None
    return AccessScope(department_id, os.getenv("ACCESS_POSITION") or None, clearance, is_admin)


def neo4j_uri() -> str | None:
    return os.getenv("NEO4J_URI") or None


def neo4j_user() -> str:
    return os.getenv("NEO4J_USER") or "neo4j"


def neo4j_password() -> str | None:
    return os.getenv("NEO4J_PASSWORD") or None


def entity_extraction_enabled() -> bool:
    """켜면 적재할 때 청크마다 본문을 LLM으로 보낸다. 외부 전송 범위를 정한 뒤에만 켠다."""
    return os.getenv("ENTITY_EXTRACTION_ENABLED", "").strip().lower() == "true"


def graph_search_enabled() -> bool:
    """Neo4j를 연결하면 질문 검색에도 쓴다. 장애나 측정 때 false로 끌 수 있다."""
    return os.getenv("GRAPH_SEARCH_ENABLED", "").strip().lower() != "false"


def document_topics_enabled() -> bool:
    """켜면 적재할 때 문서마다 앞부분을 LLM으로 보내 주제를 만든다."""
    return os.getenv("DOCUMENT_TOPICS_ENABLED", "").strip().lower() == "true"


def neo4j_database() -> str | None:
    return os.getenv("NEO4J_DATABASE") or None


def validate_startup() -> None:
    # 잘못된 값이면 질문마다 벡터 검색이 실패하므로 시작할 때 막는다
    vector_max_cosine_distance()
    if not is_production():
        return
    if not internal_secret():
        raise RuntimeError("운영 환경에는 INTERNAL_SECRET이 필요합니다. 로컬 개발이면 APP_ENV=local을 설정하세요.")
    if os.getenv("USE_SAMPLE_DATA") == "true":
        raise RuntimeError("운영 환경에서는 USE_SAMPLE_DATA를 사용할 수 없습니다.")


def uses_aws_credentials(model: str) -> bool:
    return model.startswith("bedrock")


def bedrock_options() -> dict:
    # 서버의 AWS_REGION은 다른 서비스용일 수 있고 .env로 덮어쓰이지 않으므로 Bedrock 리전을 따로 지정한다
    region = os.getenv("BEDROCK_REGION")
    return {"region_name": region} if region else {}


def jev_api_key() -> str | None:
    return os.getenv("JEV_API_KEY") or None


def jev_model() -> str:
    return os.getenv("JEV_MODEL") or DEFAULT_JEV_MODEL


def embedding_api_key() -> str | None:
    return os.getenv("EMBEDDING_API_KEY") or None


def upstage_test_api_key() -> str | None:
    return None if is_production() else os.getenv("UPSTAGE_API_KEY") or None


def embedding_model() -> str | None:
    return os.getenv("EMBEDDING_MODEL") or None


def embedding_proxy_url() -> str | None:
    # 노트북 개발 중 Titan을 쓰기 위한 임시 중계. 서버에서는 인스턴스 권한으로 직접 호출한다
    return None if is_production() else os.getenv("EMBEDDING_PROXY_URL") or None


def embedding_dimensions() -> int:
    return int(os.getenv("EMBEDDING_DIMENSIONS") or DEFAULT_EMBEDDING_DIMENSIONS)


def llm_api_key() -> str | None:
    return os.getenv("LLM_API_KEY") or None


def llm_model() -> str | None:
    return os.getenv("LLM_MODEL") or None


def chat_llm_model() -> str | None:
    """문서 없이 응대할 때 쓰는 모델. 비우면 LLM_MODEL을 쓴다."""
    return os.getenv("CHAT_LLM_MODEL") or llm_model()


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


def db_pool_max_size() -> int:
    return _positive_int("DB_POOL_MAX_SIZE", DEFAULT_DB_POOL_MAX_SIZE)


def vector_max_cosine_distance() -> float:
    """pgvector cosine distance cutoff; range is [0, 2]."""
    raw = os.getenv("VECTOR_MAX_COSINE_DISTANCE")
    value = DEFAULT_VECTOR_MAX_COSINE_DISTANCE if not raw else float(raw)
    if not 0.0 <= value <= 2.0:
        raise ValueError("VECTOR_MAX_COSINE_DISTANCE는 0과 2 사이여야 합니다.")
    return value


def database_url() -> str | None:
    return os.getenv("DATABASE_URL") or None
