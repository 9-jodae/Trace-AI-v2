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
