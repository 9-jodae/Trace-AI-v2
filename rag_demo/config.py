from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "local-dev-password")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "internal_chunks")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    chat_model: str = os.getenv("CHAT_MODEL", "")
    chat_provider: str = os.getenv("CHAT_PROVIDER", "ollama").strip().lower()
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "420"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "60"))


settings = Settings()
