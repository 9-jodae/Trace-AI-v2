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
}


@pytest.fixture
def setup(monkeypatch):
    state = {"embedded": [], "vector": [], "graph": [], "references": [], "closed": 0, "current": True, "stored": True}

    async def embed(texts):
        state["embedded"].append(texts)
        return [[0.1] for _ in texts]

    class Vector:
        async def is_current_version(self, version_id):
            return state["current"]

        async def replace_version(self, version_id, chunks, embeddings):
            state["vector"].append((version_id, chunks, embeddings))
            return state["stored"]

    class Graph:
        async def replace_document(self, document_id, source, names, chunks):
            state["graph"].append((document_id, source, names, chunks))

        async def replace_references(self, document_id, links):
            state["references"].append((document_id, links))

    async def close():
        state["closed"] += 1

    monkeypatch.setattr(indexing, "embed_documents", embed)
    monkeypatch.setattr(indexing, "get_real_stores", lambda: (Vector(), Graph()))
    monkeypatch.setattr(indexing, "close_stores", close)
    return state


def test_chunks_embeds_and_writes_both_stores(setup):
    asyncio.run(indexing.index_document(DOCUMENT))
    version, chunks, embeddings = setup["vector"][0]
    assert len(chunks) == len(embeddings) > 1
    assert version == "ver"
    document_id, source, names, graph_chunks = setup["graph"][0]
    assert (document_id, source, names) == ("doc", "규정", ["규정"])
    assert graph_chunks == chunks


def test_embeds_chunk_text(setup):
    asyncio.run(indexing.index_document(DOCUMENT))
    assert setup["embedded"][0] == [c["text"] for c in setup["vector"][0][1]]


def test_chunks_carry_server_columns(setup):
    asyncio.run(indexing.index_document(DOCUMENT))
    chunks = setup["vector"][0][1]
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert all(len(c["content_hash"]) == 64 for c in chunks)


def test_document_text_is_not_copied_to_metadata(setup):
    asyncio.run(indexing.index_document(DOCUMENT))
    assert DOC_TEXT not in repr(setup["graph"][0])


def test_blank_document_does_not_touch_stores(setup):
    assert asyncio.run(indexing.index_document({**DOCUMENT, "text": "   "})) == []
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
    assert all(d["text"] and d["version_id"] for d in documents)


def test_explicit_version_is_kept(tmp_path):
    path = tmp_path / "docs.json"
    item = {"doc_id": "d", "version_id": "v2", "title": "t", "text": "x", "status": "유효", "allowed_group_ids": ["a"]}
    path.write_text(json.dumps([item]), encoding="utf-8")
    assert indexing.load_documents(path)[0]["version_id"] == "v2"


def test_index_file_indexes_all_and_closes(setup):
    path = Path(__file__).parents[1] / "data/demo_documents.json"
    assert asyncio.run(indexing.index_file(path)) > 0
    assert len(setup["vector"]) == 8 and setup["closed"] == 1
    assert len(setup["references"]) == 8


def test_index_file_resolves_explicit_reference_after_all_documents_exist(setup, monkeypatch, tmp_path):
    source = {
        "document_id": "source",
        "version_id": "source-v1",
        "source": "출장 신청 안내.pdf",
        "text": "출장 신청 안내\n경비는 《경비 처리 절차》를 따릅니다.",
    }
    target = {
        "document_id": "target",
        "version_id": "target-v1",
        "source": "경비 처리 절차.pdf",
        "text": "경비 처리 절차\n영수증을 첨부합니다.",
    }
    monkeypatch.setattr(indexing, "load_documents", lambda path: [source, target])
    count = asyncio.run(indexing.index_file(tmp_path / "documents.json"))
    assert count > 0
    source_links = next(links for document_id, links in setup["references"] if document_id == "source")
    assert source_links == [{
        "source_document_id": "source",
        "target_document_id": "target",
        "chunk_id": setup["graph"][0][3][0]["id"],
    }]


class TestStaleVersion:
    def test_stale_document_skips_embedding_and_stores(self, setup, caplog):
        setup["current"] = False
        assert asyncio.run(indexing.index_document(DOCUMENT)) == []
        assert setup["embedded"] == [] and setup["vector"] == [] and setup["graph"] == []
        assert "not the current version" in caplog.text and DOC_TEXT not in caplog.text

    def test_version_changed_during_embedding_skips_graph(self, setup, caplog):
        setup["stored"] = False
        assert asyncio.run(indexing.index_document(DOCUMENT)) == []
        assert len(setup["vector"]) == 1 and setup["graph"] == []
        assert "version changed" in caplog.text

