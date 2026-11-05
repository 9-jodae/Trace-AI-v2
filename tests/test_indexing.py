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
    state = {"embedded": [], "vector": [], "graph": [], "references": [], "closed": 0}
    state.update(current=[True], stored=True)

    async def embed(texts):
        state["embedded"].append(texts)
        return [[0.1] for _ in texts]

    class Vector:
        async def is_current_version(self, version_id):
            return state["current"][0]

        async def replace_version(self, version_id, chunks, embeddings):
            state["vector"].append((version_id, chunks, embeddings))
            return state["stored"]

    class Graph:
        async def replace_version(self, document_id, source, names, chunks, mentions):
            state["graph"].append((document_id, source, names, chunks, mentions))

        async def replace_references(self, document_id, links):
            state["references"].append((document_id, links))

    async def close():
        state["closed"] += 1

    monkeypatch.setattr(indexing, "embed_documents", embed)
    monkeypatch.setattr(indexing, "get_real_stores", lambda: (Vector(), Graph()))
    monkeypatch.setattr(indexing, "close_stores", close)
    return state


def test_chunks_embeds_and_writes_both_stores(setup):
    count = asyncio.run(indexing.index_document(DOCUMENT))
    version, chunks, embeddings = setup["vector"][0]
    assert len(count) == len(chunks) == len(embeddings) > 1
    assert version == "ver"
    document_id, source, names, graph_chunks, mentions = setup["graph"][0]
    assert (document_id, source) == ("doc", "규정") and graph_chunks == chunks and mentions == []
    assert "규정" in names


def test_embeds_chunk_text(setup):
    asyncio.run(indexing.index_document(DOCUMENT))
    assert setup["embedded"][0] == [c["text"] for c in setup["vector"][0][1]]


def test_chunks_carry_server_columns(setup):
    asyncio.run(indexing.index_document(DOCUMENT))
    chunks = setup["vector"][0][1]
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert all(len(c["content_hash"]) == 64 for c in chunks)


def test_graph_receives_no_document_text(setup):
    asyncio.run(indexing.index_document(DOCUMENT))
    _, source, names, _, _ = setup["graph"][0]
    assert DOC_TEXT not in (source, *names)


def test_entities_are_not_extracted_by_default(setup, monkeypatch):
    async def forbidden(chunks):
        raise AssertionError("LLM을 호출하면 안 됩니다")

    monkeypatch.setattr(indexing, "extract_mentions", forbidden)
    asyncio.run(indexing.index_document(DOCUMENT))
    assert setup["graph"][0][4] == []


def test_entities_are_extracted_when_requested(setup, monkeypatch):
    async def extract(chunks):
        return [{"chunk_id": chunks[0]["id"]}]

    monkeypatch.setattr(indexing, "extract_mentions", extract)
    asyncio.run(indexing.index_document(DOCUMENT, extract_entities=True))
    assert len(setup["graph"][0][4]) == 1


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


def test_index_file_links_references_after_all_documents(setup, tmp_path):
    items = [
        {"doc_id": "a", "version_id": "va", "title": "출장비 규정", "text": "결재는 《경비 결재 절차》를 따른다."},
        {"doc_id": "b", "version_id": "vb", "title": "expense", "text": "경비 결재 절차\n내용"},
    ]
    path = tmp_path / "docs.json"
    path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    assert asyncio.run(indexing.index_file(path)) == 2
    links = dict(setup["references"])
    assert [link["target_document_id"] for link in links["a"]] == ["b"]
    assert links["b"] == [] and setup["closed"] == 1


class TestStaleVersion:
    def test_stale_document_skips_embedding_and_stores(self, setup, caplog):
        setup["current"][0] = False
        assert asyncio.run(indexing.index_document(DOCUMENT)) == []
        assert setup["embedded"] == [] and setup["vector"] == [] and setup["graph"] == []
        assert "not the current version" in caplog.text and DOC_TEXT not in caplog.text

    def test_version_changed_during_embedding_keeps_graph(self, setup, caplog):
        setup["stored"] = False
        assert asyncio.run(indexing.index_document(DOCUMENT)) == []
        assert len(setup["vector"]) == 1 and setup["graph"] == []
        assert "version changed" in caplog.text

    def test_stale_document_gets_no_references(self, setup, tmp_path):
        setup["current"][0] = False
        path = tmp_path / "docs.json"
        path.write_text(json.dumps([{"doc_id": "a", "title": "t", "text": "《u》"}]), encoding="utf-8")
        assert asyncio.run(indexing.index_file(path)) == 0
        assert setup["references"] == []

