"""임시 PostgreSQL, Neo4j, MinIO로 문서 저장 뒤 재적재를 확인한다.

Spring 콜백은 가짜 서버로 받고 임베딩은 대역이라 외부 API를 호출하지 않는다.
"""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from uuid import uuid4

import boto3
import psycopg

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
PROJECT = "trace-ingest-integration"
PASSWORD = "ingest_test_only_password"
OWNER_URL = f"postgresql://ingest_test:{PASSWORD}@127.0.0.1:15433/trace_ingest_test"
AI_URL = f"postgresql://ai_app:{PASSWORD}@127.0.0.1:15433/trace_ingest_test"
NEO4J_URI = "bolt://127.0.0.1:17688"
MINIO_URL = "http://127.0.0.1:19000"
BUCKET = "documents"
CALLBACK_PORT = 18081
SECRET = "ingest_test_only_secret"
DEPARTMENT = "00000000-0000-0000-0000-000000000000"
SCOPE = {"departmentId": DEPARTMENT, "position": "DIRECTOR", "clearance": "CONFIDENTIAL", "isAdmin": True}
# 권한은 서버 문서(docs/ai-integration-setup.md)의 ai_app 범위와 같게 준다
SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE ROLE ai_app LOGIN PASSWORD '{PASSWORD}';
CREATE TABLE documents (
    id uuid PRIMARY KEY, title text NOT NULL, current_version integer NOT NULL, deleted_at timestamptz
);
CREATE TABLE document_versions (
    id uuid PRIMARY KEY, document_id uuid NOT NULL REFERENCES documents(id), version_number integer NOT NULL
);
CREATE TABLE chunks (
    id uuid PRIMARY KEY, document_version_id uuid NOT NULL REFERENCES document_versions(id),
    chunk_index integer NOT NULL, content text NOT NULL, char_start integer NOT NULL, char_end integer NOT NULL,
    content_hash text NOT NULL, embedding vector(1024)
);
CREATE FUNCTION accessible_chunks(uuid, text, text, boolean, text[], text[], text[], date, date)
RETURNS SETOF chunks LANGUAGE sql STABLE AS $$
    SELECT c.* FROM chunks c
    JOIN document_versions dv ON dv.id = c.document_version_id
    JOIN documents d ON d.id = dv.document_id
    WHERE dv.version_number = d.current_version AND d.deleted_at IS NULL
$$;
CREATE FUNCTION filter_accessible_chunk_ids(uuid, text, text, boolean, uuid[])
RETURNS TABLE(chunk_id uuid) LANGUAGE sql STABLE AS $$
    SELECT c.id FROM accessible_chunks($1, $2, $3, $4, NULL, NULL, NULL, NULL, NULL) c WHERE c.id = ANY($5)
$$;
GRANT SELECT ON documents, document_versions TO ai_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON chunks TO ai_app;
"""
SAME = "변경되지 않는 문단입니다. " * 45
CHANGED_OLD = "수정 전 문단입니다. 연차는 오일입니다. " * 25
CHANGED_NEW = "수정 후 문단입니다. 연차는 칠일입니다. " * 25


def compose(*args, check=True):
    return subprocess.run(
        ["docker", "compose", "-p", PROJECT, "-f", str(HERE / "compose.yml"), *args],
        cwd=ROOT, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )


class Callbacks(BaseHTTPRequestHandler):
    received: dict[str, list[dict]] = {}
    secrets: list[str | None] = []

    def do_POST(self):
        job_id = self.path.split("/")[-2]
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Callbacks.received.setdefault(job_id, []).append(body)
        Callbacks.secrets.append(self.headers.get("X-Internal-Secret"))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


def wait_finished(job_id, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        statuses = [item["status"] for item in Callbacks.received.get(job_id, [])]
        if statuses and statuses[-1] in {"DONE", "FAILED"}:
            return Callbacks.received[job_id]
        time.sleep(0.1)
    raise TimeoutError(job_id)


def main():
    started = compose("up", "-d", "--wait", check=False)
    if started.returncode:
        # 준비 실패 원인이 보이도록 출력을 남긴다
        print("격리 컨테이너 시작 실패 Error:", started.stdout[-1500:], file=sys.stderr)
        compose("down", "-v", "--remove-orphans", check=False)
        return 1
    results = []

    def check(name, passed, detail=""):
        results.append(bool(passed))
        print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}")

    try:
        with psycopg.connect(OWNER_URL, autocommit=True) as connection:
            connection.execute(SCHEMA)
        s3 = boto3.client(
            "s3", endpoint_url=MINIO_URL, aws_access_key_id="ingesttest", aws_secret_access_key=PASSWORD,
            region_name="us-east-1",
        )
        s3.create_bucket(Bucket=BUCKET)
        os.environ.update(
            {
                "APP_ENV": "production",
                "INTERNAL_SECRET": SECRET,
                "USE_SAMPLE_DATA": "",
                "DATABASE_URL": AI_URL,
                "NEO4J_URI": NEO4J_URI,
                "NEO4J_USER": "neo4j",
                "NEO4J_PASSWORD": PASSWORD,
                "NEO4J_DATABASE": "neo4j",
                "EMBEDDING_DIMENSIONS": "1024",
                "JEV_API_KEY": "",
                "DOCUMENT_TOPICS_ENABLED": "false",
                "ENTITY_EXTRACTION_ENABLED": "false",
                "SPRING_INTERNAL_URL": f"http://127.0.0.1:{CALLBACK_PORT}",
                "STORAGE_BUCKET": BUCKET,
                "STORAGE_PREFIX": "",
                "STORAGE_ENDPOINT_URL": MINIO_URL,
                "STORAGE_ACCESS_KEY": "ingesttest",
                "STORAGE_SECRET_KEY": PASSWORD,
                "STORAGE_REGION": "us-east-1",
            }
        )
        server = HTTPServer(("127.0.0.1", CALLBACK_PORT), Callbacks)
        threading.Thread(target=server.serve_forever, daemon=True).start()

        from fastapi.testclient import TestClient

        from app.core.schemas import AccessScope
        from app.retrieval import indexing, service
        from app.retrieval.neo4j_store import Neo4jGraphStore

        embedded: list[str] = []

        def vector(text):
            digest = hashlib.sha256(text.encode()).digest()
            values = [0.0] * 1024
            values[int.from_bytes(digest[:2], "big") % 1024] = 1.0
            return values

        async def embed(texts):
            embedded.extend(texts)
            return [vector(text) for text in texts]

        async def embed_query(query):
            return vector(query)

        indexing.embed_documents = embed
        service.embed_query = embed_query

        import main as app_module

        headers = {"X-Internal-Secret": SECRET}

        def create_document(title):
            document_id = str(uuid4())
            with psycopg.connect(OWNER_URL, autocommit=True) as connection:
                connection.execute(
                    "INSERT INTO documents (id, title, current_version) VALUES (%s, %s, 0)", (document_id, title)
                )
            return document_id

        def save_version(document_id, number, text, extension="md", upload=True):
            version_id, key = str(uuid4()), f"{document_id}/{number}.{extension}"
            with psycopg.connect(OWNER_URL, autocommit=True) as connection:
                connection.execute(
                    "INSERT INTO document_versions (id, document_id, version_number) VALUES (%s, %s, %s)",
                    (version_id, document_id, number),
                )
                connection.execute("UPDATE documents SET current_version = %s WHERE id = %s", (number, document_id))
            if upload:
                s3.put_object(Bucket=BUCKET, Key=key, Body=text.encode())
            return version_id, key

        def ingest(client, version_id, key, extension="md", job_id=None):
            job_id = job_id or str(uuid4())
            body = {"jobId": job_id, "versionId": version_id, "storageKey": key, "extension": extension}
            response = client.post("/ingest", json=body, headers=headers)
            assert response.status_code == 202, response.status_code
            return wait_finished(job_id)

        def rows(sql, *params):
            with psycopg.connect(OWNER_URL) as connection:
                return connection.execute(sql, params).fetchall()

        async def graph(document_id):
            store = Neo4jGraphStore.from_uri(NEO4J_URI, "neo4j", PASSWORD, "neo4j")
            try:
                references = await store._read(
                    "MATCH (:Document {document_id: $id})-[:REFERENCES]->(t:Document) RETURN t.source AS title",
                    {"id": document_id},
                )
                versions = await store._read(
                    "MATCH (:Document {document_id: $id})-[:HAS_CHUNK]->(c:Chunk) RETURN c.version_id AS version",
                    {"id": document_id},
                )
                return sorted(r["title"] for r in references), {v["version"] for v in versions}
            finally:
                await store.close()

        scope = AccessScope(DEPARTMENT, "DIRECTOR", "CONFIDENTIAL", True)

        with TestClient(app_module.app) as client:
            refused = client.post(
                "/ingest",
                json={"jobId": str(uuid4()), "versionId": str(uuid4()), "storageKey": "x", "extension": "md"},
            )
            check("비밀 헤더 없으면 거부", refused.status_code == 401, str(refused.status_code))

            target_b = create_document("참조 대상 비")
            target_c = create_document("참조 대상 시")
            main_document = create_document("연차 규정")
            for document_id, title in ((target_b, "참조 대상 비"), (target_c, "참조 대상 시")):
                version, key = save_version(document_id, 1, f"{title} 문서입니다.")
                ingest(client, version, key)

            embedded.clear()
            v1_text = f"{SAME}\n\n{CHANGED_OLD}\n\n세부는 《참조 대상 비》를 따른다."
            v1, v1_key = save_version(main_document, 1, v1_text)
            job = ingest(client, v1, v1_key)
            first_embedded = len(embedded)
            statuses = [item["status"] for item in job]
            check("첫 저장: 처리 상태 콜백", statuses == ["PROCESSING", "DONE"], str(statuses))
            check("콜백에 비밀 헤더 포함", set(Callbacks.secrets) == {SECRET})
            count = rows("SELECT count(*) FROM chunks WHERE document_version_id = %s", v1)[0][0]
            check("첫 저장: 청크 저장(ai_app 권한)", count == first_embedded and count >= 2, f"청크 {count}개")
            references, versions = asyncio.run(graph(main_document))
            check("첫 저장: Neo4j 참조", references == ["참조 대상 비"] and versions == {v1}, str(references))

            embedded.clear()
            v2_text = f"{SAME}\n\n{CHANGED_NEW}\n\n세부는 《참조 대상 시》를 따른다."
            v2, v2_key = save_version(main_document, 2, v2_text)
            job = ingest(client, v2, v2_key)
            v2_chunks = rows("SELECT content_hash, embedding::text FROM chunks WHERE document_version_id = %s", v2)
            v1_hashes = {
                row[0]: row[1] for row in rows(
                    "SELECT content_hash, embedding::text FROM chunks WHERE document_version_id = %s", v1
                )
            }
            reused = [row for row in v2_chunks if v1_hashes.get(row[0]) == row[1]]
            check(
                "수정 저장: 바뀐 청크만 임베딩",
                job[-1]["status"] == "DONE" and 0 < len(embedded) < len(v2_chunks) and reused,
                f"전체 {len(v2_chunks)}개, 새 임베딩 {len(embedded)}개, 재사용 {len(reused)}개",
            )
            references, versions = asyncio.run(graph(main_document))
            check("수정 저장: 참조 교체", references == ["참조 대상 시"], str(references))
            check("수정 저장: Neo4j 청크가 새 버전만", versions == {v2})
            # 저장소 연결이 앱의 이벤트 루프에 묶여 있으므로 같은 루프에서 검색한다
            result = client.portal.call(service.search, "칠일", None, scope)
            found = result["chunks"]
            check(
                "수정 저장: 검색에 새 내용만",
                any("칠일" in chunk["text"] for chunk in found)
                and not any("오일" in chunk["text"] for chunk in found)
                and not result["failed_sources"],
                f"결과 {len(found)}개, 실패 {result['failed_sources']}",
            )

            embedded.clear()
            old_job = ingest(client, v1, v1_key)
            references, versions = asyncio.run(graph(main_document))
            check(
                "늦게 온 구버전 작업은 건너뜀",
                old_job[-1]["status"] == "DONE" and not embedded and versions == {v2},
            )

            embedded.clear()
            duplicate = str(uuid4())
            v3, v3_key = save_version(main_document, 3, f"{SAME}\n\n세 번째 수정입니다.")
            body = {"jobId": duplicate, "versionId": v3, "storageKey": v3_key, "extension": "md"}
            client.post("/ingest", json=body, headers=headers)
            client.post("/ingest", json=body, headers=headers)
            job = wait_finished(duplicate)
            statuses = [item["status"] for item in job]
            check("같은 jobId 동시 요청은 한 번만", statuses == ["PROCESSING", "DONE"], str(statuses))

            missing_version, _ = save_version(create_document("파일 없음"), 1, "", upload=False)
            job = ingest(client, missing_version, "없는/파일.md")
            failure = {"status": "FAILED", "errorMessage": "원본 파일을 읽지 못했습니다."}
            check("원본 없음은 고정 문구로 실패", job[-1] == failure)

            empty_version, empty_key = save_version(create_document("빈 문서"), 1, "   ")
            job = ingest(client, empty_version, empty_key)
            check("빈 문서는 실패", job[-1]["status"] == "FAILED" and "비어" in job[-1]["errorMessage"])
        server.shutdown()
    finally:
        down = compose("down", "-v", "--remove-orphans", check=False)
        if down.returncode:
            print("격리 컨테이너 정리에 실패했습니다.", file=sys.stderr)
    print(f"\n{sum(results)}/{len(results)} 통과")
    return 0 if results and all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
