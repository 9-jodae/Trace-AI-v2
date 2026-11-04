import asyncio
import json
from pathlib import Path

import pytest

from app.core.errors import AiUnavailableError
from app.retrieval import indexing

DOC_TEXT = "재택근무 규정은 인사팀이 소유한다. " * 100
DOCUMENT = {
    "document_id": "doc",
    "version_id": "ver",
    "text": DOC_TEXT,
    "source": "규정",
    "status": "유효",
    "allowed_group_ids": ["hr"],
}


@pytest.fixture
def setup(monkeypatch):
    state = {"embedded": [], "vector": [], "graph": [], "closed": 0}

    async def embed(texts):
        state["embedded"].append(texts)
        return [[0.1] for _ in texts]

    class Vector:
        async def replace_version(self, version_id, chunks, embeddings):
            state["vector"].append((version_id, chunks, embeddings))

    class Graph:
        async def replace_version(self, version_id, document_id, document, chunks):
            state["graph"].append((version_id, document_id, document, chunks))

    async def close():
        state["closed"] += 1

    monkeypatch.setattr(indexing, "embed_documents", embed)
    monkeypatch.setattr(indexing, "get_real_stores", lambda: (Vector(), Graph()))
    monkeypatch.setattr(indexing, "close_stores", close)
    return state


def test_chunks_embeds_and_writes_both_stores(setup):
    count = asyncio.run(indexing.index_document(DOCUMENT))
    version, chunks, embeddings = setup["vector"][0]
    assert count == len(chunks) == len(embeddings) > 1
    assert version == "ver"
    graph_version, document_id, metadata, graph_chunks = setup["graph"][0]
    assert (graph_version, document_id) == ("ver", "doc") and graph_chunks == chunks


def test_embeds_chunk_text(setup):
    asyncio.run(indexing.index_document(DOCUMENT))
    assert setup["embedded"][0] == [c["text"] for c in setup["vector"][0][1]]


def test_chunks_carry_permission_fields(setup):
    asyncio.run(indexing.index_document(DOCUMENT))
    assert all(c["status"] == "유효" and c["allowed_group_ids"] == ["hr"] for c in setup["vector"][0][1])


def test_document_text_is_not_copied_to_metadata(setup):
    asyncio.run(indexing.index_document(DOCUMENT))
    assert "text" not in setup["graph"][0][2]


def test_blank_document_does_not_touch_stores(setup):
    assert asyncio.run(indexing.index_document({**DOCUMENT, "text": "   "})) == 0
    assert setup["vector"] == [] and setup["graph"] == [] and setup["embedded"] == []


def test_missing_stores_fail_without_silent_success(monkeypatch):
    async def embed(texts):
        return [[0.1] for _ in texts]

    monkeypatch.setattr(indexing, "embed_documents", embed)
    with pytest.raises(AiUnavailableError):
        asyncio.run(indexing.index_document(DOCUMENT))


def test_load_demo_documents():
    path = Path(__file__).parents[1] / "data/demo_documents.json"
    documents = indexing.load_documents(path)
    assert len(documents) == 8
    assert all(d["text"] and d["allowed_group_ids"] and d["version_id"] for d in documents)


def test_explicit_version_is_kept(tmp_path):
    path = tmp_path / "docs.json"
    item = {"doc_id": "d", "version_id": "v2", "title": "t", "text": "x", "status": "유효", "allowed_group_ids": ["a"]}
    path.write_text(json.dumps([item]), encoding="utf-8")
    assert indexing.load_documents(path)[0]["version_id"] == "v2"


def test_index_file_indexes_all_and_closes(setup):
    path = Path(__file__).parents[1] / "data/demo_documents.json"
    assert asyncio.run(indexing.index_file(path)) > 0
    assert len(setup["vector"]) == 8 and setup["closed"] == 1
