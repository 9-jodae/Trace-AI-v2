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
        result = asyncio.run(stores.PostgresVectorStore("postgresql://db").search([0.5, 1.0], 3, SCOPE))
        assert result == [chunk("a")]
        sql, params = connection.executed[0]
        assert "<=>" in sql and "LIMIT" in sql and sql.lstrip().startswith("SELECT")
        assert params == ("dept", "STAFF", "INTERNAL", False, None, None, None, None, None, "[0.5,1.0]", 3)
        assert connect.url == "postgresql://db"
        assert connect.kwargs["connect_timeout"] == stores.CONNECT_TIMEOUT_SECONDS
        assert connect.kwargs["options"] == f"-c statement_timeout={stores.SEARCH_STATEMENT_TIMEOUT_MS}"

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
        assert connection.executed[0][1] == (
            *SCOPE, ["pdf"], ["REPORT"], ["LOCAL"], date(2026, 1, 1), date(2026, 12, 31), "[1.0]", 2,
        )

    def test_filter_accessible_with_filters_uses_filtered_function(self, connect):
        connection = FakeConnection(rows=[{"id": "a"}])
        connect(connection)
        filters = SearchFilters(extensions=["pdf"])
        result = asyncio.run(stores.PostgresVectorStore("x").filter_accessible(["a", "b"], SCOPE, filters))
        assert result == {"a"}
        sql, params = connection.executed[0]
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
        assert not any(word in connection.executed[0][0].upper() for word in ["INSERT", "DELETE", "UPDATE", "DROP"])

    def test_values_are_parameterized(self):
        assert "%s" in stores.VECTOR_SEARCH_SQL and "{" not in stores.VECTOR_SEARCH_SQL

    def test_filter_accessible_query_and_params(self, connect):
        connection = FakeConnection(rows=[{"id": "a"}])
        connect(connection)
        result = asyncio.run(stores.PostgresVectorStore("x").filter_accessible(["a", "b"], SCOPE))
        assert result == {"a"}
        sql, params = connection.executed[0]
        assert "FROM filter_accessible_chunk_ids(" in sql
        assert params == ("dept", "STAFF", "INTERNAL", False, ["a", "b"])

    @pytest.mark.parametrize("ids, scope", [(["a"], None), ([], SCOPE)])
    def test_filter_accessible_without_input_does_not_connect(self, ids, scope):
        assert asyncio.run(stores.PostgresVectorStore("x").filter_accessible(ids, scope)) == set()

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
            ("a", "ver", 0, "A", 0, 1, "hash-a", "[0.1]"),
            ("b", "ver", 1, "B", 0, 1, "hash-b", "[0.2]"),
        ]
        assert "document_version_id = %s" in connection.executed[0][0]
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


class TestFulltext:
    @pytest.fixture
    def read(self, monkeypatch):
        calls = []

        async def fake(self, sql, params):
            calls.append((sql, params))
            return [chunk("a")]

        monkeypatch.setattr(stores.PostgresVectorStore, "_read", fake)
        return calls

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("제12조 연차", "'제12조':* | '연차':*"),
            ("a' | b:* & !c", "'a':* | 'b':* | 'c':*"),
            ("!!! ???", ""),
        ],
    )
    def test_query_escapes_operators(self, text, expected):
        assert stores.build_fulltext_query(text) == expected

    def test_query_limits_terms(self):
        query = stores.build_fulltext_query(" ".join(str(i) for i in range(50)))
        assert query.count(" | ") == stores.MAX_FULLTEXT_TERMS - 1

    def test_search_goes_through_permission_function(self):
        assert "FROM accessible_chunks(" in stores.FULLTEXT_SEARCH_SQL
        assert "FROM chunks" not in stores.FULLTEXT_SEARCH_SQL

    def test_params_order(self, read):
        filters = SearchFilters(extensions=["pdf"])
        result = asyncio.run(stores.PostgresVectorStore("x").fulltext("연차", 3, SCOPE, filters))
        assert result == [chunk("a")]
        expression = "'연차':*"
        assert read[0][1] == (*SCOPE, ["pdf"], None, None, None, None, expression, expression, 3)

    def test_returns_same_fields_as_vector_search(self):
        for alias in ["AS id", "AS text", "AS source", "AS document_id", "AS version_id", "char_start", "char_end"]:
            assert alias in stores.FULLTEXT_SEARCH_SQL

    @pytest.mark.parametrize("query, scope", [("연차", None), ("!!!", SCOPE)])
    def test_no_scope_or_terms_makes_no_query(self, read, query, scope):
        assert asyncio.run(stores.PostgresVectorStore("x").fulltext(query, 3, scope)) == []
        assert read == []

    def test_sample_store_returns_nothing(self):
        assert asyncio.run(stores.SampleVectorStore().fulltext("재택근무", 3, None)) == []
