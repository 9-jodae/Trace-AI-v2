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
    state = {
        "embedded": [], "vector": [], "graph": [], "references": [], "catalog": [], "closed": 0,
        "current": True, "stored": True,
    }

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
            state["catalog"] = [item for item in state["catalog"] if item["document_id"] != document_id]
            state["catalog"].append({"document_id": document_id, "source": source})

        async def replace_references(self, document_id, links):
            existing = {item["document_id"] for item in state["catalog"]}
            available = [link for link in links if link["target_document_id"] in existing]
            state["references"].append((document_id, available))

        async def list_document_catalog(self):
            return state["catalog"]

        async def replace_entity_graph(self, document_id, version_id, mentions, assertions):
            state["entities"] = (document_id, version_id)

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


def test_partial_file_uses_existing_graph_catalog_for_cross_file_reference(setup, monkeypatch, tmp_path):
    source = {
        "document_id": "source",
        "version_id": "source-v1",
        "source": "출장 신청 안내.pdf",
        "text": "본문 첫 줄은 제목이 아닙니다.\n《경비 처리 절차》를 참조합니다.",
    }
    setup["catalog"] = [{"document_id": "target", "source": "경비 처리 절차.pdf"}]
    monkeypatch.setattr(indexing, "load_documents", lambda path: [source])

    asyncio.run(indexing.index_file(tmp_path / "partial.json"))

    assert setup["references"] == [("source", [{
        "source_document_id": "source",
        "target_document_id": "target",
        "chunk_id": setup["graph"][0][3][0]["id"],
    }])]
    assert "본문 첫 줄은 제목이 아닙니다." not in repr(setup["graph"][0][2])


def test_index_file_continues_after_one_document_fails(setup, monkeypatch, tmp_path, caplog):
    first = {**DOCUMENT, "document_id": "bad", "version_id": "bad-v1", "text": "민감 본문 A"}
    second = {**DOCUMENT, "document_id": "good", "version_id": "good-v1", "text": "공개 본문 B"}
    monkeypatch.setattr(indexing, "load_documents", lambda path: [first, second])

    async def embed(texts):
        if "민감 본문 A" in texts[0]:
            raise RuntimeError("민감 본문 A를 포함한 예외")
        return [[0.1] for _ in texts]

    monkeypatch.setattr(indexing, "embed_documents", embed)
    with pytest.raises(RuntimeError, match="1개 문서 적재에 실패했습니다"):
        asyncio.run(indexing.index_file(tmp_path / "partial.json"))

    assert [item[0] for item in setup["vector"]] == ["good-v1"]
    assert [item[0] for item in setup["graph"]] == ["good"]
    assert "민감 본문 A" not in caplog.text


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


class TestEntityExtractionSwitch:
    @pytest.fixture
    def extraction(self, setup, monkeypatch):
        calls = []

        async def extract(document_id, version_id, chunks):
            calls.append(document_id)
            return [], []

        # 테스트 저장소도 Neo4j 저장소처럼 보이게 한다
        monkeypatch.setattr(indexing, "Neo4jGraphStore", object)
        monkeypatch.setattr(indexing, "extract_entity_graph", extract)
        return calls

    def test_is_off_by_default(self, setup, extraction):
        asyncio.run(indexing.index_document(DOCUMENT))
        assert extraction == [] and "entities" not in setup

    def test_runs_when_enabled(self, setup, extraction, monkeypatch):
        monkeypatch.setenv("ENTITY_EXTRACTION_ENABLED", "true")
        asyncio.run(indexing.index_document(DOCUMENT))
        assert extraction == ["doc"] and setup["entities"] == ("doc", "ver")

