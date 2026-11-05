"""Disposable, real PostgreSQL + Neo4j test through the application's chat API."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import psycopg

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = Path(__file__).with_name("compose.yml")
PROJECT = "docs199-graph-integration"
sys.path.insert(0, str(ROOT))
PG_URL = "postgresql://graph_test:graph_test_only_password@127.0.0.1:15432/docs199_graph_test"
NEO4J_URI = "bolt://127.0.0.1:17687"
NEO4J_PASSWORD = "graph_test_only_password"
SCOPE = {
    "departmentId": "11111111-1111-4111-8111-111111111111",
    "position": "STAFF",
    "clearance": "INTERNAL",
    "isAdmin": True,
}
QUESTION = "그래프 검색 시작 문서의 일정 정보는 몇 시야?"
DIRECT_TITLE_QUESTION = "정기 점검 안내에는 일정이 몇 시라고 적혀 있어?"
UNRELATED_QUESTION = "화성 기지의 물 재활용 절차는 어떻게 돼?"
EXPECTED_QUOTE = "정기 점검은 매주 월요일 오전 9시에 진행합니다."


def compose(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "compose", "-p", PROJECT, "-f", str(COMPOSE), *args],
        cwd=ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def wait_for_postgres() -> None:
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            with psycopg.connect(PG_URL, connect_timeout=2):
                return
        except psycopg.Error:
            time.sleep(1)
    raise RuntimeError("격리 PostgreSQL이 제한 시간 안에 준비되지 않았습니다.")


def prepare_postgres() -> tuple[str, str, str, str, str]:
    document_id, version_id = str(uuid4()), str(uuid4())
    stale_version_id, stale_chunk_id = str(uuid4()), str(uuid4())
    related_document_id, related_version_id = str(uuid4()), str(uuid4())
    with psycopg.connect(PG_URL) as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        connection.execute(
            """
            CREATE TABLE documents (
                id uuid PRIMARY KEY,
                title text NOT NULL,
                current_version integer NOT NULL,
                deleted_at timestamptz
            );
            CREATE TABLE document_versions (
                id uuid PRIMARY KEY,
                document_id uuid NOT NULL REFERENCES documents(id),
                version_number integer NOT NULL
            );
            CREATE TABLE chunks (
                id uuid PRIMARY KEY,
                document_version_id uuid NOT NULL REFERENCES document_versions(id),
                chunk_index integer NOT NULL,
                content text NOT NULL,
                char_start integer NOT NULL,
                char_end integer NOT NULL,
                content_hash text NOT NULL,
                embedding vector(1024)
            );
            CREATE FUNCTION accessible_chunks(
                uuid, text, text, boolean, text[], text[], text[], date, date
            ) RETURNS SETOF chunks LANGUAGE sql STABLE AS $$
                SELECT c.*
                FROM chunks c
                JOIN document_versions dv ON dv.id = c.document_version_id
                JOIN documents d ON d.id = dv.document_id
                WHERE dv.version_number = d.current_version
                  AND d.deleted_at IS NULL
                  AND $4 = TRUE
            $$;
            CREATE FUNCTION filter_accessible_chunk_ids(uuid, text, text, boolean, uuid[])
            RETURNS TABLE(chunk_id uuid) LANGUAGE sql STABLE AS $$
                SELECT c.id
                FROM accessible_chunks($1, $2, $3, $4, NULL, NULL, NULL, NULL, NULL) c
                WHERE c.id = ANY($5)
            $$;
            """
        )
        connection.execute(
            "INSERT INTO documents (id, title, current_version) VALUES (%s, %s, 1), (%s, %s, 1)",
            (document_id, "그래프 검색 시작 문서", related_document_id, "정기 점검 안내"),
        )
        connection.execute(
            """INSERT INTO document_versions (id, document_id, version_number)
               VALUES (%s, %s, 1), (%s, %s, 1), (%s, %s, 0)""",
            (version_id, document_id, related_version_id, related_document_id, stale_version_id, document_id),
        )
        connection.execute(
            """INSERT INTO chunks
               (id, document_version_id, chunk_index, content, char_start, char_end, content_hash, embedding)
               VALUES (%s, %s, 99, %s, 0, 35, %s, %s::vector)""",
            (
                stale_chunk_id,
                stale_version_id,
                "폐기된 버전의 오래된 안내입니다.",
                "stale-version",
                "[1," + ",".join(["0"] * 1023) + "]",
            ),
        )
    return document_id, version_id, related_document_id, related_version_id, stale_chunk_id


def main() -> int:
    compose("up", "-d", "--wait")
    try:
        wait_for_postgres()
        ids = prepare_postgres()
        document_id, version_id, related_document_id, related_version_id, stale_chunk_id = ids

        # This process uses only the disposable services declared above.
        os.environ.update(
            {
                "APP_ENV": "local",
                "USE_SAMPLE_DATA": "false",
                "DATABASE_URL": PG_URL,
                "NEO4J_URI": NEO4J_URI,
                "NEO4J_USER": "neo4j",
                "NEO4J_PASSWORD": NEO4J_PASSWORD,
                "NEO4J_DATABASE": "neo4j",
                "EMBEDDING_DIMENSIONS": "1024",
                "ACCESS_IS_ADMIN": "true",
            }
        )

        from fastapi.testclient import TestClient

        from app.agent import graph as agent_graph
        from app.retrieval import indexing, service
        from app.retrieval.fusion import rrf_fuse as original_rrf_fuse
        from app.retrieval.neo4j_store import Neo4jGraphStore
        from app.retrieval.stores import PostgresVectorStore

        def new_trace() -> dict[str, object]:
            return {
                "graph_seed_document_ids": [],
                "postgres_hybrid_seed_document_ids": [],
                "graph_candidate_chunk_ids": [],
                "postgres_accessible_document_ids": [],
                "postgres_fetch_requested_chunk_ids": [],
                "postgres_hydrated_chunk_ids": [],
                "postgres_hydrated_document_ids": [],
                "postgres_hydrated_version_ids": [],
                "postgres_unhydrated_candidate_chunk_ids": [],
                "postgres_vector_chunk_ids": [],
                "postgres_fulltext_chunk_ids": [],
                "rrf_input_rankings": [],
                "rrf_output_chunk_ids": [],
            }

        trace = new_trace()
        capture = {"enabled": False}

        async def deterministic_document_embeddings(texts: list[str]) -> list[list[float]]:
            return [[1.0] + [0.0] * 1023 for _ in texts]

        async def unavailable_query_embedding(_query: str) -> list[float]:
            # Graph 경로의 임베딩 비의존 동작을 검증하기 위해 질문 임베딩만 차단한다.
            raise RuntimeError("test-only query embedding unavailable")

        async def deterministic_query_embedding(query: str) -> list[float]:
            if "화성" in query:
                return [0.0, 1.0] + [0.0] * 1022
            return [1.0] + [0.0] * 1023

        async def grounded_test_answer(_question, context, _history) -> str:
            for index, chunk in enumerate(context, start=1):
                if EXPECTED_QUOTE in chunk["text"]:
                    return json.dumps(
                        {
                            "answer": "격리 테스트의 정기 점검은 매주 월요일 오전 9시에 진행됩니다.",
                            "evidence": [{"chunk": index, "quote": EXPECTED_QUOTE}],
                        },
                        ensure_ascii=False,
                    )
            return json.dumps({"answer": "관련 문서를 찾지 못했습니다.", "evidence": []}, ensure_ascii=False)

        indexing.embed_documents = deterministic_document_embeddings
        service.embed_query = unavailable_query_embedding
        agent_graph.generate_answer = grounded_test_answer

        with tempfile.TemporaryDirectory(prefix="docs199-graph-test-") as temp_dir:
            fixture_path = Path(temp_dir) / "documents.json"
            fixture_path.write_text(
                json.dumps(
                    [
                        {
                            "doc_id": document_id,
                            "version_id": version_id,
                            "title": "그래프 검색 시작 문서",
                            "text": "그래프 검색 시작 문서는 정기 점검 안내를 《정기 점검 안내》에서 참조합니다.",
                        },
                        {
                            "doc_id": related_document_id,
                            "version_id": related_version_id,
                            "title": "정기 점검 안내",
                            "text": EXPECTED_QUOTE,
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            indexed_chunks = asyncio.run(indexing.index_file(fixture_path))
        if indexed_chunks < 2:
            raise RuntimeError("가상 문서 색인이 완료되지 않았습니다.")

        async def add_stale_graph_candidate() -> None:
            graph_store = Neo4jGraphStore.from_uri(
                NEO4J_URI, "neo4j", NEO4J_PASSWORD, "neo4j"
            )
            try:
                async with graph_store._driver.session(database="neo4j") as session:
                    await (
                        await session.run(
                            """MATCH (d:Document {document_id: $document_id})
                               CREATE (c:Chunk {id: $chunk_id, chunk_index: 99})
                               CREATE (d)-[:HAS_CHUNK]->(c)""",
                            document_id=document_id,
                            chunk_id=stale_chunk_id,
                        )
                    ).consume()
            finally:
                await graph_store.close()

        asyncio.run(add_stale_graph_candidate())

        original_graph = Neo4jGraphStore.search_from_documents
        original_accessible_docs = PostgresVectorStore.accessible_document_ids
        original_fetch = PostgresVectorStore.fetch_accessible
        original_vector = PostgresVectorStore.search
        original_fulltext = PostgresVectorStore.fulltext

        async def traced_graph(self, document_ids, limit):
            if capture["enabled"]:
                trace["graph_seed_document_ids"] = list(document_ids)
            result = await original_graph(self, document_ids, limit)
            if capture["enabled"]:
                trace["graph_candidate_chunk_ids"] = result
            return result

        async def traced_accessible_docs(self, document_ids, scope, filters=None):
            result = await original_accessible_docs(self, document_ids, scope, filters)
            if capture["enabled"]:
                trace["postgres_hybrid_seed_document_ids"] = list(document_ids)
                trace["postgres_accessible_document_ids"] = sorted(result)
            return result

        async def traced_fetch(self, chunk_ids, scope, filters=None, embedding=None):
            result = await original_fetch(self, chunk_ids, scope, filters, embedding)
            if capture["enabled"]:
                requested_ids = list(dict.fromkeys([*trace["postgres_fetch_requested_chunk_ids"], *chunk_ids]))
                trace["postgres_fetch_requested_chunk_ids"] = requested_ids
                trace["postgres_hydrated_chunk_ids"] = list(dict.fromkeys([
                    *trace["postgres_hydrated_chunk_ids"], *(item["id"] for item in result)
                ]))
                trace["postgres_hydrated_document_ids"] = list(dict.fromkeys([
                    *trace["postgres_hydrated_document_ids"], *(item["document_id"] for item in result)
                ]))
                trace["postgres_hydrated_version_ids"] = list(dict.fromkeys([
                    *trace["postgres_hydrated_version_ids"], *(item["version_id"] for item in result)
                ]))
                hydrated_ids = set(trace["postgres_hydrated_chunk_ids"])
                trace["postgres_unhydrated_candidate_chunk_ids"] = sorted(set(requested_ids) - hydrated_ids)
            return result

        async def traced_vector(self, query_embedding, limit, scope, filters=None):
            result = await original_vector(self, query_embedding, limit, scope, filters)
            if capture["enabled"]:
                trace["postgres_vector_chunk_ids"] = [item["id"] for item in result]
            return result

        async def traced_fulltext(self, query, limit, scope, filters=None):
            result = await original_fulltext(self, query, limit, scope, filters)
            if capture["enabled"]:
                trace["postgres_fulltext_chunk_ids"] = [item["id"] for item in result]
            return result

        def traced_rrf(rankings, limit):
            result = original_rrf_fuse(rankings, limit)
            if capture["enabled"]:
                trace["rrf_input_rankings"] = [[item["id"] for item in ranking] for ranking in rankings]
                trace["rrf_output_chunk_ids"] = [item["id"] for item in result]
            return result

        Neo4jGraphStore.search_from_documents = traced_graph
        PostgresVectorStore.accessible_document_ids = traced_accessible_docs
        PostgresVectorStore.fetch_accessible = traced_fetch
        PostgresVectorStore.search = traced_vector
        PostgresVectorStore.fulltext = traced_fulltext
        service.rrf_fuse = traced_rrf

        import main as app_module

        capture["enabled"] = True
        payload = {"message": QUESTION, "accessScope": SCOPE}
        with TestClient(app_module.app) as client:
            no_embedding_response = client.post("/api/v1/agent/chat", json=payload)
            no_embedding_trace = json.loads(json.dumps(trace))
            no_embedding_body = no_embedding_response.json()

            trace.clear()
            trace.update(new_trace())
            service.embed_query = deterministic_query_embedding
            hybrid_response = client.post(
                "/api/v1/agent/chat",
                json={**payload, "threadId": "isolated-hybrid-test"},
            )
            hybrid_trace = json.loads(json.dumps(trace))
            hybrid_body = hybrid_response.json()

            extra_cases = []
            for case_name, question, access_scope, expected_document_id in (
                ("vector-only seed without title or keyword", "언제 진행돼?", SCOPE, related_document_id),
                ("hybrid seed without title", "월요일 오전 정기 점검은 몇 시에 진행해?", SCOPE, related_document_id),
                ("direct title match", DIRECT_TITLE_QUESTION, SCOPE, related_document_id),
                ("unrelated question", UNRELATED_QUESTION, SCOPE, None),
                ("PostgreSQL access denied", QUESTION, {**SCOPE, "isAdmin": False}, None),
            ):
                trace.clear()
                trace.update(new_trace())
                response = client.post(
                    "/api/v1/agent/chat",
                    json={"message": question, "accessScope": access_scope, "threadId": f"isolated-{case_name}"},
                )
                case_trace = json.loads(json.dumps(trace))
                body = response.json()
                if expected_document_id:
                    passed = (
                        response.status_code == 200
                        and expected_document_id in case_trace["graph_seed_document_ids"]
                        and expected_document_id in case_trace["postgres_hydrated_document_ids"]
                        and any(source.get("documentId") == expected_document_id for source in body.get("sources", []))
                        and any(item.get("documentId") == expected_document_id for item in body.get("evidence", []))
                    )
                    if case_name.startswith("vector-only"):
                        passed = (
                            passed
                            and bool(case_trace["postgres_vector_chunk_ids"])
                            and not case_trace["postgres_fulltext_chunk_ids"]
                        )
                else:
                    passed = (
                        response.status_code == 200
                        and not body.get("sources")
                        and not body.get("evidence")
                        and body.get("answer") == "관련 문서를 찾지 못했습니다."
                    )
                extra_cases.append(
                    {
                        "name": case_name,
                        "result": "PASS" if passed else "FAIL",
                        "question": question,
                        "access_scope": access_scope,
                        "trace": case_trace,
                        "final_source_chunk_ids": [item["id"] for item in body.get("sources", [])],
                        "sources": [
                            {key: item.get(key) for key in ("id", "source", "documentId", "versionId")}
                            for item in body.get("sources", [])
                        ],
                        "answer": body.get("answer"),
                        "evidence": body.get("evidence"),
                        "failed_sources": body.get("failedSources"),
                        "http_status": response.status_code,
                    }
                )

        def case_passed(response, body, case_trace, require_vector: bool) -> bool:
            related_source_ids = [
                source["id"]
                for source in body.get("sources", [])
                if source.get("documentId") == related_document_id
            ]
            return (
                response.status_code == 200
                and document_id in case_trace["graph_seed_document_ids"]
                and bool(case_trace["graph_candidate_chunk_ids"])
                and stale_chunk_id in case_trace["graph_candidate_chunk_ids"]
                and stale_chunk_id in case_trace["postgres_unhydrated_candidate_chunk_ids"]
                and related_document_id in case_trace["postgres_hydrated_document_ids"]
                and any(item in related_source_ids for item in case_trace["graph_candidate_chunk_ids"])
                and bool(related_source_ids)
                and any(
                    item.get("documentId") == related_document_id and item.get("quote") == EXPECTED_QUOTE
                    for item in body.get("evidence", [])
                )
                and case_trace["rrf_output_chunk_ids"] == [item["id"] for item in body.get("sources", [])]
                and (not require_vector or bool(case_trace["postgres_vector_chunk_ids"]))
            )

        no_embedding_passed = case_passed(
            no_embedding_response, no_embedding_body, no_embedding_trace, require_vector=False
        )
        hybrid_passed = case_passed(hybrid_response, hybrid_body, hybrid_trace, require_vector=True)
        extra_passed = all(item["result"] == "PASS" for item in extra_cases)
        report = {
            "result": "PASS" if no_embedding_passed and hybrid_passed and extra_passed else "FAIL",
            "execution_path": "in-process FastAPI POST /api/v1/agent/chat, real isolated PostgreSQL and Neo4j",
            "question": QUESTION,
            "access_scope": SCOPE,
            "filters": None,
            "graph_seed_strategy": (
                "document IDs from PostgreSQL vector and full-text hits, "
                "ACL checked before Neo4j expansion"
            ),
            "cases": [
                {
                    "name": "graph traversal continues when query embedding fails",
                    "result": "PASS" if no_embedding_passed else "FAIL",
                    "query_embedding": "test stub raises; graph path continues without vector distance filtering",
                    "trace": no_embedding_trace,
                    "final_source_chunk_ids": [item["id"] for item in no_embedding_body.get("sources", [])],
                    "sources": [
                        {key: item.get(key) for key in ("id", "source", "documentId", "versionId")}
                        for item in no_embedding_body.get("sources", [])
                    ],
                    "answer": no_embedding_body.get("answer"),
                    "evidence": no_embedding_body.get("evidence"),
                    "failed_sources": no_embedding_body.get("failedSources"),
                    "http_status": no_embedding_response.status_code,
                },
                {
                    "name": "vector + graph + PostgreSQL full-text RRF",
                    "result": "PASS" if hybrid_passed else "FAIL",
                    "query_embedding": "deterministic test stub; no external embedding service",
                    "trace": hybrid_trace,
                    "final_source_chunk_ids": [item["id"] for item in hybrid_body.get("sources", [])],
                    "sources": [
                        {key: item.get(key) for key in ("id", "source", "documentId", "versionId")}
                        for item in hybrid_body.get("sources", [])
                    ],
                    "answer": hybrid_body.get("answer"),
                    "evidence": hybrid_body.get("evidence"),
                    "failed_sources": hybrid_body.get("failedSources"),
                    "http_status": hybrid_response.status_code,
                },
                *extra_cases,
            ],
            "postgres_validation": (
                "The disposable accessible_chunks fixture permits only admin scope and current, non-deleted versions; "
                "it is not the Spring server ACL implementation."
            ),
            "stale_version_candidate_chunk_id": stale_chunk_id,
            "document_embedding": "deterministic 1024-dimension test stub; no external embedding service",
            "answer_model": "test stub; no Claude/Bedrock call",
            "graph_traversal_depth": 1,
            "graph_path": [
                {"from_document_id": document_id, "relationship": "REFERENCES", "to_document_id": related_document_id}
            ],
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if no_embedding_passed and hybrid_passed and extra_passed else 1
    finally:
        down = compose("down", "-v", "--remove-orphans", check=False)
        if down.returncode:
            print("격리 컨테이너 정리 명령이 실패했습니다.", file=sys.stderr)
            return_code = 1
        else:
            return_code = 0
        if return_code:
            raise RuntimeError("격리 GraphDB 테스트 컨테이너 정리에 실패했습니다.")


if __name__ == "__main__":
    raise SystemExit(main())
