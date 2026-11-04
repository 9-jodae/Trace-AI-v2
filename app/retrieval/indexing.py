from app.core import config
from app.core.errors import AiUnavailableError
from app.retrieval.chunking import chunk_document
from app.retrieval.embedding import embed_documents
from app.retrieval.stores import PostgresVectorStore


async def index_document(document_id: str, text: str, source: str) -> int:
    if not config.database_url():
        raise AiUnavailableError("DATABASE_URL이 설정되지 않았습니다.")
    chunks = chunk_document(document_id, text, source)
    embeddings = await embed_documents([chunk["text"] for chunk in chunks])
    await PostgresVectorStore(config.database_url()).replace_document(source, chunks, embeddings)
    return len(chunks)
