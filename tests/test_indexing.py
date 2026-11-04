import asyncio

import pytest

from app.core.errors import AiUnavailableError
from app.retrieval import indexing

DOC = "재택근무 규정은 인사팀이 소유한다. " * 100


@pytest.fixture
def setup(monkeypatch):
    state = {"embedded": [], "replaced": []}

    async def embed(texts):
        state["embedded"].append(texts)
        return [[0.1] for _ in texts]

    class Store:
        def __init__(self, url):
            state["url"] = url

        async def replace_version(self, version_id, chunks, embeddings):
            state["replaced"].append((version_id, chunks, embeddings))

    monkeypatch.setenv("DATABASE_URL", "postgresql://db")
    monkeypatch.setattr(indexing, "embed_documents", embed)
    monkeypatch.setattr(indexing, "PostgresVectorStore", Store)
    return state


def test_chunks_embeds_and_stores(setup):
    count = asyncio.run(indexing.index_document("doc", "ver", DOC, "규정"))
    source, chunks, embeddings = setup["replaced"][0]
    assert count == len(chunks) == len(embeddings) > 1
    assert source == "ver"
    assert setup["url"] == "postgresql://db"


def test_embeds_chunk_text(setup):
    asyncio.run(indexing.index_document("doc", "ver", DOC, "규정"))
    assert setup["embedded"][0] == [c["text"] for c in setup["replaced"][0][1]]


def test_missing_database_url_raises_before_embedding(setup, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "")
    with pytest.raises(AiUnavailableError):
        asyncio.run(indexing.index_document("doc", "ver", DOC, "규정"))
    assert setup["embedded"] == []


def test_blank_document_does_not_delete_existing_chunks(setup):
    assert asyncio.run(indexing.index_document("doc", "ver", "   ", "규정")) == 0
    assert setup["replaced"] == []
    assert setup["embedded"] == []


def test_reindex_is_scoped_to_version_not_source(setup):
    # 출처 이름이 같은 다른 문서의 청크가 함께 지워지면 안 된다
    asyncio.run(indexing.index_document("doc", "ver-A", DOC, "같은 출처"))
    assert setup["replaced"][0][0] == "ver-A"
