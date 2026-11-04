import asyncio

import pytest
from conftest import chunk

from app.retrieval import stores
from app.retrieval.schemas import ACTIVE_STATUS


async def fake_embed_documents(texts):
    fake_embed_documents.calls += 1
    return [[1.0, 0.0] if "승인" in text else [0.0, 1.0] for text in texts]


class TestCosine:
    def test_identical(self):
        assert stores._cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert stores._cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector_does_not_crash(self):
        assert stores._cosine([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_high_dimensions(self):
        vector = [0.5] * 1024
        assert stores._cosine(vector, vector) == pytest.approx(1.0)


def test_to_vector_format():
    assert stores._to_vector([0.5, 1.0, -2.0]) == "[0.5,1.0,-2.0]"


class TestSampleVectorStore:
    @pytest.fixture(autouse=True)
    def embed(self, monkeypatch):
        fake_embed_documents.calls = 0
        monkeypatch.setattr(stores, "embed_documents", fake_embed_documents)

    def test_most_similar_first(self):
        result = asyncio.run(stores.SampleVectorStore().search([1.0, 0.0], 1, []))
        assert result[0]["id"] == "remote-1"

    def test_limit(self):
        assert len(asyncio.run(stores.SampleVectorStore().search([1.0, 0.0], 2, []))) == 2

    def test_document_embeddings_are_cached(self):
        store = stores.SampleVectorStore()

        async def twice():
            await store.search([1.0, 0.0], 1, [])
            await store.search([1.0, 0.0], 1, [])

        asyncio.run(twice())
        assert fake_embed_documents.calls == 1


class TestSampleGraphStore:
    @pytest.mark.parametrize(
        "query, expected",
        [
            ("재택근무 규정을 승인한 곳", ["remote-1", "remote-3"]),
            ("재택근무규정 승인", ["remote-1", "remote-3"]),
            ("결제 서비스 담당", ["payment-1", "payment-2"]),
            ("관계없는 질문", []),
        ],
    )
    def test_entity_match(self, query, expected):
        assert [c["id"] for c in asyncio.run(stores.SampleGraphStore().search(query, 10, []))] == expected

    def test_limit(self):
        assert len(asyncio.run(stores.SampleGraphStore().search("재택근무 규정", 1, []))) == 1


FULL_A = {
    "id": "a",
    "text": "A",
    "source": "s",
    "document_id": "doc",
    "version_id": "ver",
    "char_start": 0,
    "char_end": 1,
    "status": "유효",
    "allowed_group_ids": ["hr"],
}
FULL_B = {**FULL_A, "id": "b", "text": "B"}


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    async def fetchall(self):
        return self.rows


class FakeBatchCursor:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def executemany(self, sql, rows):
        self.connection.batches.append((sql, rows))
        if self.connection.fail_on and sql.strip().startswith(self.connection.fail_on):
            raise RuntimeError("db error")


class FakeConnection:
    def __init__(self, rows=None, fail_on=None):
        self.rows = rows or []
        self.fail_on = fail_on
        self.executed = []
        self.batches = []
        self.exit_error = "not closed"

    async def __aenter__(self):
        return self

    async def __aexit__(self, error_type, error, traceback):
        self.exit_error = error_type

    async def execute(self, sql, params):
        self.executed.append((sql, params))
        if self.fail_on and sql.strip().startswith(self.fail_on):
            raise RuntimeError("db error")
        return FakeCursor(self.rows)

    def cursor(self):
        return FakeBatchCursor(self)


@pytest.fixture
def connect(monkeypatch):
    def install(connection):
        async def fake_connect(url, **kwargs):
            install.url = url
            install.kwargs = kwargs
            return connection

        monkeypatch.setattr(stores.psycopg.AsyncConnection, "connect", fake_connect)

    return install


class TestPostgresVectorStore:
    def test_search_query_and_params(self, connect):
        connection = FakeConnection(rows=[chunk("a")])
        connect(connection)
        result = asyncio.run(stores.PostgresVectorStore("postgresql://db").search([0.5, 1.0], 3, ["hr"]))
        assert result == [chunk("a")]
        sql, params = connection.executed[0]
        assert "<=>" in sql and "LIMIT" in sql and sql.lstrip().startswith("SELECT")
        assert params == (ACTIVE_STATUS, ["hr"], "[0.5,1.0]", 3)
        assert connect.url == "postgresql://db"
        assert connect.kwargs["connect_timeout"] == stores.CONNECT_TIMEOUT_SECONDS
        assert connect.kwargs["options"] == f"-c statement_timeout={stores.SEARCH_STATEMENT_TIMEOUT_MS}"

    def test_no_groups_returns_nothing_without_connecting(self):
        assert asyncio.run(stores.PostgresVectorStore("x").search([1.0], 1, [])) == []

    def test_search_filters_status_and_groups(self):
        assert "status = %s" in stores.VECTOR_SEARCH_SQL and "allowed_group_ids &&" in stores.VECTOR_SEARCH_SQL

    def test_search_never_writes(self, connect):
        connection = FakeConnection()
        connect(connection)
        asyncio.run(stores.PostgresVectorStore("x").search([1.0], 1, ["hr"]))
        assert not any(word in connection.executed[0][0].upper() for word in ["INSERT", "DELETE", "UPDATE", "DROP"])

    def test_values_are_parameterized(self):
        assert "%s" in stores.VECTOR_SEARCH_SQL and "{" not in stores.VECTOR_SEARCH_SQL

    def test_replace_deletes_version_then_inserts_in_one_batch(self, connect):
        connection = FakeConnection()
        connect(connection)
        asyncio.run(stores.PostgresVectorStore("x").replace_version("ver", [FULL_A, FULL_B], [[0.1], [0.2]]))
        assert [sql.split()[0] for sql, _ in connection.executed] == ["DELETE"]
        assert connection.executed[0][1] == ("ver",)
        assert len(connection.batches) == 1
        sql, rows = connection.batches[0]
        assert sql.strip().startswith("INSERT")
        assert rows == [
            ("a", "A", "s", "doc", "ver", 0, 1, "유효", ["hr"], "[0.1]"),
            ("b", "B", "s", "doc", "ver", 0, 1, "유효", ["hr"], "[0.2]"),
        ]
        assert connection.exit_error is None
        assert connect.kwargs["options"] == f"-c statement_timeout={stores.WRITE_STATEMENT_TIMEOUT_MS}"

    def test_replace_failure_reaches_transaction_exit(self, connect):
        connection = FakeConnection(fail_on="INSERT")
        connect(connection)
        with pytest.raises(RuntimeError):
            asyncio.run(stores.PostgresVectorStore("x").replace_version("ver", [FULL_A], [[0.1]]))
        assert connection.exit_error is RuntimeError

    def test_replace_rejects_count_mismatch_before_deleting(self, connect):
        connection = FakeConnection()
        connect(connection)
        with pytest.raises(ValueError):
            asyncio.run(stores.PostgresVectorStore("x").replace_version("ver", [FULL_A, FULL_B], [[0.1]]))
        assert connection.executed == [] and connection.batches == []
