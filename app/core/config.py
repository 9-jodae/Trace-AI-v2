import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_JEV_MODEL = "jev-latest"
DEFAULT_EMBEDDING_DIMENSIONS = 1536


def jev_api_key() -> str | None:
    return os.getenv("JEV_API_KEY") or None


def jev_model() -> str:
    return os.getenv("JEV_MODEL") or DEFAULT_JEV_MODEL


def embedding_api_key() -> str | None:
    return os.getenv("EMBEDDING_API_KEY") or None


def embedding_model() -> str | None:
    return os.getenv("EMBEDDING_MODEL") or None


def llm_api_key() -> str | None:
    return os.getenv("LLM_API_KEY") or None


def llm_model() -> str | None:
    return os.getenv("LLM_MODEL") or None


def embedding_dimensions() -> int:
    return int(os.getenv("EMBEDDING_DIMENSIONS") or DEFAULT_EMBEDDING_DIMENSIONS)


def database_url() -> str | None:
    return os.getenv("DATABASE_URL") or None
