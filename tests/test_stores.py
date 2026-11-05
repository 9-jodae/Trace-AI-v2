import asyncio
from datetime import date

import pytest
from conftest import chunk

from app.core.schemas import AccessScope, SearchFilters
from app.retrieval import stores


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
        assert [c["id"] for c in asyncio.run(stores.SampleGraphStore().search(query, 10))] == expected

    def test_limit(self):
        assert len(asyncio.run(stores.SampleGraphStore().search("재택근무 규정", 1))) == 1


FULL_A = {
    "id": "a",
    "text": "A",
    "source": "s",
    "document_id": "doc",
    "version_id": "ver",
    "char_start": 0,
    "char_end": 1,
    "chunk_index": 0,
    "content_hash": "hash-a",
}
FULL_B = {**FULL_A, "id": "b", "text": "B", "chunk_index": 1, "content_hash": "hash-b"}
SCOPE = AccessScope("dept", "STAFF", "INTERNAL", False)


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


class FakePool:
    instances = []
    open_error = None

    def __init__(self, conninfo, connection, **kwargs):
        self.conninfo = conninfo
        self.kwargs = kwargs
        self._connection = connection
        self.open_calls = []
        self.closed = 0
        FakePool.instances.append(self)

    async def open(self, wait, timeout):
        self.open_calls.append((wait, timeout))
        if FakePool.open_error:
            raise FakePool.open_error

    async def close(self):
        self.closed += 1

    def connection(self):
        return self._connection

    @staticmethod
    async def check_connection(connection):
        pass


@pytest.fixture
def connect(monkeypatch):
    FakePool.instances = []
    FakePool.open_error = None

    def install(connection):
        def factory(conninfo, **kwargs):
            return FakePool(conninfo, connection, **kwargs)

        factory.check_connection = FakePool.check_connection
        monkeypatch.setattr(stores, "AsyncConnectionPool", factory)

    return install


def data_statements(connection):
    return [item for item in connection.executed if item[0] != stores.SET_STATEMENT_TIMEOUT_SQL]


class TestPostgresVectorStore:
    def test_search_query_and_params(self, connect):
        connection = FakeConnection(rows=[chunk("a")])
        connect(connection)
        result = asyncio.run(stores.PostgresVectorStore("postgresql://db").search([0.5, 1.0], 3, SCOPE))
        assert result == [chunk("a")]
        sql, params = data_statements(connection)[0]
        assert "<=>" in sql and "LIMIT" in sql and sql.lstrip().startswith("SELECT")
        assert params == ("dept", "STAFF", "INTERNAL", False, None, None, None, None, None, "[0.5,1.0]", 3)
        assert connection.executed[0] == (stores.SET_STATEMENT_TIMEOUT_SQL, (str(stores.SEARCH_STATEMENT_TIMEOUT_MS),))

    def test_search_passes_filters_in_function_order(self, connect):
        connection = FakeConnection()
        connect(connection)
        filters = SearchFilters(
            extensions=["pdf"],
            document_types=["REPORT"],
            source_kinds=["LOCAL"],
            authored_from=date(2026, 1, 1),
            authored_to=date(2026, 12, 31),
        )
        asyncio.run(stores.PostgresVectorStore("x").search([1.0], 2, SCOPE, filters))
        assert data_statements(connection)[0][1] == (
            *SCOPE, ["pdf"], ["REPORT"], ["LOCAL"], date(2026, 1, 1), date(2026, 12, 31), "[1.0]", 2,
        )

    def test_filter_accessible_with_filters_uses_filtered_function(self, connect):
        connection = FakeConnection(rows=[{"id": "a"}])
        connect(connection)
        filters = SearchFilters(extensions=["pdf"])
        result = asyncio.run(stores.PostgresVectorStore("x").filter_accessible(["a", "b"], SCOPE, filters))
        assert result == {"a"}
        sql, params = data_statements(connection)[0]
        assert "FROM accessible_chunks(" in sql and "ANY(%s::uuid[])" in sql
        assert params == (*SCOPE, ["pdf"], None, None, None, None, ["a", "b"])

    def test_no_scope_returns_nothing_without_connecting(self):
        assert asyncio.run(stores.PostgresVectorStore("x").search([1.0], 1, None)) == []

    def test_search_goes_through_permission_function(self):
        assert "FROM accessible_chunks(" in stores.VECTOR_SEARCH_SQL
        assert "FROM chunks" not in stores.VECTOR_SEARCH_SQL

    def test_search_skips_chunks_without_embedding(self):
        assert "c.embedding IS NOT NULL" in stores.VECTOR_SEARCH_SQL

    def test_search_returns_fields_of_response_contract(self):
        for alias in ["AS id", "AS text", "AS source", "AS document_id", "AS version_id", "char_start", "char_end"]:
            assert alias in stores.VECTOR_SEARCH_SQL

    def test_sql_uses_only_server_chunk_columns(self):
        for sql in [stores.VECTOR_SEARCH_SQL, stores.INSERT_CHUNK_SQL, stores.DELETE_VERSION_SQL]:
            assert "allowed_group_ids" not in sql and "status" not in sql

    def test_search_never_writes(self, connect):
        connection = FakeConnection()
        connect(connection)
        asyncio.run(stores.PostgresVectorStore("x").search([1.0], 1, SCOPE))
        sql = data_statements(connection)[0][0].upper()
        assert not any(word in sql for word in ["INSERT", "DELETE", "UPDATE", "DROP"])

    def test_values_are_parameterized(self):
        assert "%s" in stores.VECTOR_SEARCH_SQL and "{" not in stores.VECTOR_SEARCH_SQL

    def test_filter_accessible_query_and_params(self, connect):
        connection = FakeConnection(rows=[{"id": "a"}])
        connect(connection)
        result = asyncio.run(stores.PostgresVectorStore("x").filter_accessible(["a", "b"], SCOPE))
        assert result == {"a"}
        sql, params = data_statements(connection)[0]
        assert "FROM filter_accessible_chunk_ids(" in sql
        assert params == ("dept", "STAFF", "INTERNAL", False, ["a", "b"])

    @pytest.mark.parametrize("ids, scope", [(["a"], None), ([], SCOPE)])
    def test_filter_accessible_without_input_does_not_connect(self, ids, scope):
        assert asyncio.run(stores.PostgresVectorStore("x").filter_accessible(ids, scope)) == set()

    def test_replace_deletes_version_then_inserts_in_one_batch(self, connect):
        connection = FakeConnection()
        connect(connection)
        asyncio.run(stores.PostgresVectorStore("x").replace_version("ver", [FULL_A, FULL_B], [[0.1], [0.2]]))
        assert [sql.split()[0] for sql, _ in data_statements(connection)] == ["DELETE"]
        assert data_statements(connection)[0][1] == ("ver",)
        assert len(connection.batches) == 1
        sql, rows = connection.batches[0]
        assert sql.strip().startswith("INSERT")
        assert rows == [
            ("a", "ver", 0, "A", 0, 1, "hash-a", "[0.1]"),
            ("b", "ver", 1, "B", 0, 1, "hash-b", "[0.2]"),
        ]
        assert "document_version_id = %s" in data_statements(connection)[0][0]
        assert connection.exit_error is None
        assert connection.executed[0] == (stores.SET_STATEMENT_TIMEOUT_SQL, (str(stores.WRITE_STATEMENT_TIMEOUT_MS),))

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
        assert FakePool.instances == []

    def test_statement_timeout_applies_to_transaction_only(self):
        assert stores.SET_STATEMENT_TIMEOUT_SQL.rstrip().endswith("true)")


class TestConnectionPool:
    def test_requests_share_one_pool(self, connect):
        connect(FakeConnection())
        store = stores.PostgresVectorStore("postgresql://db")

        async def twice():
            await store.search([1.0], 1, SCOPE)
            await store.filter_accessible(["a"], SCOPE)

        asyncio.run(twice())
        assert len(FakePool.instances) == 1

    def test_concurrent_first_requests_create_one_pool(self, connect):
        connect(FakeConnection())
        store = stores.PostgresVectorStore("x")

        async def together():
            await asyncio.gather(*(store.search([1.0], 1, SCOPE) for _ in range(5)))

        asyncio.run(together())
        assert len(FakePool.instances) == 1

    def test_pool_settings(self, connect, monkeypatch):
        monkeypatch.setenv("DB_POOL_MAX_SIZE", "3")
        connect(FakeConnection())
        asyncio.run(stores.PostgresVectorStore("postgresql://db").search([1.0], 1, SCOPE))
        pool = FakePool.instances[0]
        assert pool.conninfo == "postgresql://db" and pool.kwargs["max_size"] == 3
        assert pool.kwargs["min_size"] == stores.POOL_MIN_SIZE and pool.kwargs["open"] is False
        assert pool.kwargs["timeout"] == stores.POOL_WAIT_SECONDS
        assert pool.kwargs["kwargs"]["connect_timeout"] == stores.CONNECT_TIMEOUT_SECONDS
        assert pool.kwargs["check"] is FakePool.check_connection
        assert pool.open_calls == [(True, stores.CONNECT_TIMEOUT_SECONDS)]

    def test_default_max_size(self, connect):
        connect(FakeConnection())
        asyncio.run(stores.PostgresVectorStore("x").search([1.0], 1, SCOPE))
        assert FakePool.instances[0].kwargs["max_size"] == 10

    def test_invalid_max_size_is_rejected(self, connect, monkeypatch):
        monkeypatch.setenv("DB_POOL_MAX_SIZE", "0")
        connect(FakeConnection())
        with pytest.raises(ValueError):
            asyncio.run(stores.PostgresVectorStore("x").search([1.0], 1, SCOPE))

    def test_failed_open_closes_pool_and_retries_next_time(self, connect):
        connect(FakeConnection())
        store = stores.PostgresVectorStore("x")
        FakePool.open_error = OSError("refused")
        with pytest.raises(OSError):
            asyncio.run(store.search([1.0], 1, SCOPE))
        assert FakePool.instances[0].closed == 1
        FakePool.open_error = None
        store._pool_lock = asyncio.Lock()
        asyncio.run(store.search([1.0], 1, SCOPE))
        assert len(FakePool.instances) == 2

    def test_close_closes_pool_once(self, connect):
        connect(FakeConnection())
        store = stores.PostgresVectorStore("x")

        async def run():
            await store.search([1.0], 1, SCOPE)
            await store.close()
            await store.close()

        asyncio.run(run())
        assert FakePool.instances[0].closed == 1

    def test_close_without_use_is_safe(self):
        asyncio.run(stores.PostgresVectorStore("x").close())
