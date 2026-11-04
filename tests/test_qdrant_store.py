"""실제 Qdrant 클라이언트의 메모리 모드로 필터와 재색인 동작을 확인한다. 서버 성능은 재지 않는다."""

import asyncio

import pytest
from qdrant_client import AsyncQdrantClient

from app.retrieval.qdrant_store import QdrantVectorStore
from app.retrieval.schemas import ACTIVE_STATUS


def make_chunk(number, version="v1", status=ACTIVE_STATUS, groups=("hr",)):
    return {
        "id": f"00000000-0000-0000-0000-{number:012d}",
        "text": f"청크 {number}",
        "source": "문서",
        "document_id": "doc",
        "version_id": version,
        "char_start": 0,
        "char_end": 3,
        "status": status,
        "allowed_group_ids": list(groups),
    }


def run(scenario):
    async def wrapper():
        store = QdrantVectorStore(AsyncQdrantClient(location=":memory:"), "test")
        try:
            return await scenario(store)
        finally:
            await store.close()

    return asyncio.run(wrapper())


def texts(results):
    return [chunk["text"] for chunk in results]


def test_returns_only_allowed_groups():
    async def scenario(store):
        await store.replace_version("v1", [make_chunk(1), make_chunk(2, groups=("it",))], [[1.0, 0.0], [1.0, 0.0]])
        return await store.search([1.0, 0.0], 5, ["hr"])

    assert texts(run(scenario)) == ["청크 1"]


def test_excludes_inactive_documents():
    async def scenario(store):
        await store.replace_version("v1", [make_chunk(1, status="폐지")], [[1.0, 0.0]])
        return await store.search([1.0, 0.0], 5, ["hr"])

    assert run(scenario) == []


def test_no_groups_returns_nothing():
    async def scenario(store):
        await store.replace_version("v1", [make_chunk(1)], [[1.0, 0.0]])
        return await store.search([1.0, 0.0], 5, [])

    assert run(scenario) == []


def test_no_groups_does_not_query_the_server():
    class Forbidden:
        async def query_points(self, **kwargs):
            raise AssertionError("서버에 조회하면 안 됩니다")

    assert asyncio.run(QdrantVectorStore(Forbidden(), "test").search([1.0, 0.0], 5, [])) == []


def test_orders_by_similarity_and_limits():
    async def scenario(store):
        await store.replace_version("v1", [make_chunk(1), make_chunk(2)], [[0.0, 1.0], [1.0, 0.0]])
        return await store.search([1.0, 0.0], 1, ["hr"])

    assert texts(run(scenario)) == ["청크 2"]


def test_results_have_contract_fields_only():
    async def scenario(store):
        await store.replace_version("v1", [make_chunk(1)], [[1.0, 0.0]])
        return await store.search([1.0, 0.0], 1, ["hr"])

    assert set(run(scenario)[0]) == {"id", "text", "source", "document_id", "version_id", "char_start", "char_end"}


def test_reindex_replaces_only_that_version():
    async def scenario(store):
        await store.replace_version("v1", [make_chunk(1), make_chunk(2)], [[1.0, 0.0], [1.0, 0.0]])
        await store.replace_version("v2", [make_chunk(3, "v2")], [[1.0, 0.0]])
        await store.replace_version("v1", [make_chunk(4)], [[1.0, 0.0]])
        return await store.search([1.0, 0.0], 10, ["hr"])

    assert sorted(texts(run(scenario))) == ["청크 3", "청크 4"]


def test_dimension_mismatch_keeps_existing_data():
    async def scenario(store):
        await store.replace_version("v1", [make_chunk(1)], [[1.0, 0.0]])
        with pytest.raises(ValueError):
            await store.replace_version("v2", [make_chunk(2, "v2")], [[1.0, 0.0, 0.0]])
        return await store.search([1.0, 0.0], 5, ["hr"])

    assert texts(run(scenario)) == ["청크 1"]


def test_count_mismatch_changes_nothing():
    async def scenario(store):
        await store.replace_version("v1", [make_chunk(1)], [[1.0, 0.0]])
        with pytest.raises(ValueError):
            await store.replace_version("v1", [make_chunk(2), make_chunk(3)], [[1.0, 0.0]])
        return await store.search([1.0, 0.0], 5, ["hr"])

    assert texts(run(scenario)) == ["청크 1"]


def test_upsert_failure_keeps_existing_chunks(monkeypatch):
    async def scenario(store):
        await store.replace_version("v1", [make_chunk(1)], [[1.0, 0.0]])

        async def fail(*args, **kwargs):
            raise RuntimeError("저장 실패")

        with monkeypatch.context() as patch:
            patch.setattr(store._client, "upsert", fail)
            with pytest.raises(RuntimeError):
                await store.replace_version("v1", [make_chunk(2)], [[1.0, 0.0]])
        return await store.search([1.0, 0.0], 5, ["hr"])

    assert texts(run(scenario)) == ["청크 1"]


def test_reindex_removes_chunks_no_longer_present():
    async def scenario(store):
        await store.replace_version("v1", [make_chunk(1), make_chunk(2)], [[1.0, 0.0], [1.0, 0.0]])
        await store.replace_version("v1", [make_chunk(1)], [[1.0, 0.0]])
        return await store.search([1.0, 0.0], 5, ["hr"])

    assert texts(run(scenario)) == ["청크 1"]



def test_staging_is_hidden_until_published():
    async def scenario(store):
        chunk = {**make_chunk(1), "index_status": "STAGING"}
        await store.replace_version("v1", [chunk], [[1.0, 0.0]])
        assert await store.search([1.0, 0.0], 5, ["hr"]) == []
        await store.publish_version("doc", "v1")
        return await store.search([1.0, 0.0], 5, ["hr"])

    assert texts(run(scenario)) == ["청크 1"]


def test_graph_candidates_keep_document_order_and_version_scope():
    async def scenario(store):
        await store.replace_version("v1", [make_chunk(1)], [[1.0, 0.0]])
        other = {**make_chunk(2, "b1"), "document_id": "b"}
        await store.replace_version("b1", [other], [[0.0, 1.0]])
        docs = [{"document_id": "b", "version_id": "b1"}, {"document_id": "doc", "version_id": "v1"}]
        hits = await store.search_in_documents([1.0, 0.0], docs, 5, ["hr"])
        assert texts(hits) == ["청크 2", "청크 1"]
        assert await store.search_in_documents([1.0, 0.0], docs, 5, ["it"]) == []
        assert await store.search_in_documents([1.0, 0.0], [{"document_id": "b", "version_id": "old"}], 5, ["hr"]) == []

    run(scenario)


def test_retirement_is_scoped_to_document_and_version():
    async def scenario(store):
        await store.replace_version("v1", [make_chunk(1)], [[1.0, 0.0]])
        other = {**make_chunk(2), "document_id": "b"}
        await store.replace_version("v1", [other], [[1.0, 0.0]])
        await store.retire_version("doc", "v1")
        return await store.search([1.0, 0.0], 5, ["hr"])

    assert texts(run(scenario)) == ["청크 2"]
