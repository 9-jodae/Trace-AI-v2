from pathlib import Path
from typing import Any
from uuid import uuid5, NAMESPACE_URL

from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from langchain_ollama import OllamaEmbeddings

from .config import settings
from .chunking import split_document

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "demo_documents.json"

# Only these explicit demo relations are loaded. The claim is tied to a source chunk.
RELATIONS = [
    ("GUIDE-TRAVEL-001", "PROC-EXP-001", "출장 경비 정산은 경비 처리 절차를 따른다.", "출장 종료 후 7일 이내에 경비 정산을 제출합니다."),
    ("PROC-ONBOARD-001", "FAQ-IT-001", "신규 입사자의 계정과 장비 준비는 IT 안내 절차와 연결된다.", "IT팀은 노트북과 업무 계정을 준비합니다."),
    ("POL-HR-001", "POL-SEC-001", "휴가 예외 문의 시 문서 접근 및 문의 기준을 참고할 수 있다.", "예외적인 휴가 처리는 인사팀에 문의합니다."),
]


class LocalStores:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        self.client = QdrantClient(url=settings.qdrant_url, timeout=5)
        self.embeddings = OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )

    def close(self):
        self.driver.close()
        self.client.close()

    def check_connections(self) -> dict[str, str]:
        results = {}
        try:
            self.driver.verify_connectivity()
            results["Neo4j"] = "연결됨"
        except Exception as exc:
            results["Neo4j"] = f"연결 실패: {exc}"
        try:
            self.client.get_collections()
            results["Qdrant"] = "연결됨"
        except Exception as exc:
            results["Qdrant"] = f"연결 실패: {exc}"
        try:
            self.embeddings.embed_query("연결 확인")
            results["Ollama 임베딩"] = f"연결됨 ({settings.embedding_model})"
        except Exception as exc:
            results["Ollama 임베딩"] = f"연결 실패: {exc}"
        return results

    def ingest_demo_documents(self) -> int:
        """Upsert docs/chunks into both local stores; safe to run repeatedly."""
        import json

        docs = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        chunks = [chunk for doc in docs for chunk in split_document(doc)]
        vectors = self.embeddings.embed_documents([c["text"] for c in chunks])
        vector_size = len(vectors[0]) if vectors else 0

        if self.client.collection_exists(settings.qdrant_collection):
            info = self.client.get_collection(settings.qdrant_collection)
            existing_size = info.config.params.vectors.size
            if existing_size != vector_size:
                self.client.delete_collection(settings.qdrant_collection)
        if not self.client.collection_exists(settings.qdrant_collection):
            self.client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )

        points = []
        for chunk, vector in zip(chunks, vectors):
            payload = dict(chunk)
            points.append(models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, chunk["chunk_id"])),
                vector=vector,
                payload=payload,
            ))
        self.client.upsert(settings.qdrant_collection, points=points, wait=True)

        with self.driver.session(database=settings.neo4j_database) as session:
            session.run("CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE")
            session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")
            session.run("CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text, c.title]")
            session.run("CALL db.awaitIndex('chunk_text', 300)")
            session.run("""
                UNWIND $docs AS item
                MERGE (d:Document {doc_id: item.doc_id})
                SET d.title=item.title, d.doc_type=item.doc_type, d.source=item.source,
                    d.created_at=item.created_at, d.status=item.status,
                    d.allowed_group_ids=item.allowed_group_ids, d.text=item.text
            """, docs=docs)
            demo_ids = [doc["doc_id"] for doc in docs]
            session.run("MATCH (a:Document)-[r:RELATED_TO]->(b:Document) WHERE a.doc_id IN $ids AND b.doc_id IN $ids DELETE r", ids=demo_ids)
            session.run("MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk) WHERE d.doc_id IN $ids DETACH DELETE c", ids=demo_ids)
            session.run("""
                UNWIND $chunks AS item
                MATCH (d:Document {doc_id: item.doc_id})
                CREATE (c:Chunk {chunk_id:item.chunk_id, title:item.title, text:item.text,
                    doc_type:item.doc_type, source:item.source, created_at:item.created_at,
                    status:item.status, allowed_group_ids:item.allowed_group_ids,
                    char_start:item.char_start, char_end:item.char_end})
                CREATE (d)-[:HAS_CHUNK]->(c)
            """, chunks=chunks)
            for source_id, target_id, claim, anchor in RELATIONS:
                source_chunk = next((c for c in chunks if c["doc_id"] == source_id and anchor in c["text"]), None)
                if source_chunk:
                    session.run("""
                        MATCH (a:Document {doc_id:$source}), (b:Document {doc_id:$target})
                        MERGE (a)-[r:RELATED_TO {claim:$claim}]->(b)
                        SET r.evidence_chunk_id=$chunk_id
                    """, source=source_id, target=target_id, claim=claim, chunk_id=source_chunk["chunk_id"])
        return len(chunks)

    @staticmethod
    def _neo_filter(where_alias: str, plan: dict, params: dict, prefix: str = "") -> str:
        clauses = [f"{where_alias}.status = '유효'", f"any(g IN coalesce({where_alias}.allowed_group_ids, []) WHERE g IN $allowed_groups)"]
        params["allowed_groups"] = plan.get("allowed_group_ids", [])
        if not params["allowed_groups"]:
            clauses.append("false")
        filters = plan.get("filters", {}) or {}
        for key in ("doc_type", "source", "doc_id"):
            values = filters.get(key) or []
            if values:
                name = f"{prefix}{key}"
                prop = "doc_id" if key == "doc_id" else key
                clauses.append(f"{where_alias}.{prop} IN ${name}")
                params[name] = values
        return " AND ".join(clauses)

    def _qdrant_filter(self, plan: dict) -> models.Filter:
        must = [models.FieldCondition(key="status", match=models.MatchValue(value="유효"))]
        allowed = plan.get("allowed_group_ids", [])
        if not allowed:
            must.append(models.FieldCondition(key="doc_id", match=models.MatchValue(value="__no_access__")))
        else:
            must.append(models.FieldCondition(key="allowed_group_ids", match=models.MatchAny(any=allowed)))
        filters = plan.get("filters", {}) or {}
        for field in ("doc_type", "source", "doc_id"):
            values = filters.get(field) or []
            if values:
                must.append(models.FieldCondition(key=field, match=models.MatchAny(any=values)))
        return models.Filter(must=must)

    def vector_search(self, query: str, plan: dict) -> list[dict]:
        vector = self.embeddings.embed_query(query)
        response = self.client.query_points(
            collection_name=settings.qdrant_collection,
            query=vector,
            query_filter=self._qdrant_filter(plan),
            limit=int(plan.get("top_k", 5)),
            with_payload=True,
        )
        return [self._from_payload(point.payload, point.score, "vector") for point in response.points]

    def fulltext_search(self, query: str, plan: dict) -> list[dict]:
        # Lucene metacharacters are removed; each whitespace-separated term is an OR clause.
        import re
        terms = re.findall(r"[\w가-힣]+", query, flags=re.UNICODE)[:12]
        if not terms:
            return []
        lucene_query = " OR ".join(f'"{term}"' for term in terms)
        params: dict[str, Any] = {"query": lucene_query, "limit": int(plan.get("top_k", 5))}
        where = self._neo_filter("d", plan, params)
        with self.driver.session(database=settings.neo4j_database) as session:
            rows = session.run(f"""
                CALL db.index.fulltext.queryNodes('chunk_text', $query) YIELD node AS c, score
                MATCH (d:Document)-[:HAS_CHUNK]->(c)
                WHERE {where}
                RETURN c.chunk_id AS chunk_id, d.doc_id AS doc_id, d.title AS title, d.source_url AS source_url,
                    c.text AS text, c.char_start AS char_start, c.char_end AS char_end,
                    d.doc_type AS doc_type, d.source AS source, d.created_at AS created_at,
                    d.status AS status, d.allowed_group_ids AS allowed_group_ids, score
                ORDER BY score DESC LIMIT $limit
            """, params)
            return [self._from_payload(dict(row), row["score"], "fulltext") for row in rows]

    def related_relations(self, hits: list[dict], plan: dict) -> list[dict]:
        """Show graph edges only when their claim's evidence chunk is in final hits."""
        chunk_ids = [hit["chunk_id"] for hit in hits]
        doc_ids = list({hit["doc_id"] for hit in hits})
        if not chunk_ids or not doc_ids:
            return []
        params: dict[str, Any] = {"chunk_ids": chunk_ids, "doc_ids": doc_ids}
        source_where = self._neo_filter("sd", plan, params, "s_")
        target_where = self._neo_filter("td", plan, params, "t_")
        with self.driver.session(database=settings.neo4j_database) as session:
            rows = session.run(f"""
                MATCH (sd:Document)-[rel:RELATED_TO]->(td:Document)
                WHERE rel.evidence_chunk_id IN $chunk_ids
                    AND (sd.doc_id IN $doc_ids OR td.doc_id IN $doc_ids)
                    AND {source_where} AND {target_where}
                RETURN DISTINCT sd.title AS source_title, td.title AS target_title,
                    rel.claim AS claim, rel.evidence_chunk_id AS evidence_chunk_id
                LIMIT 20
            """, params)
            return [dict(row) for row in rows]

    @staticmethod
    def _from_payload(payload: dict, score: float, method: str) -> dict:
        return {
            "doc_id": payload["doc_id"], "chunk_id": payload["chunk_id"],
            "title": payload["title"], "text": payload["text"],
            "source_url": payload.get("source_url"),
            "doc_type": payload["doc_type"], "source": payload["source"],
            "created_at": payload["created_at"], "status": payload["status"],
            "allowed_group_ids": payload.get("allowed_group_ids", []),
            "char_start": payload["char_start"], "char_end": payload["char_end"],
            "score": float(score), "methods": [method],
        }


def combine_rrf(result_sets: list[list[dict]], top_k: int) -> list[dict]:
    """Fuse ranked result lists with reciprocal-rank fusion and dedupe by chunk."""
    merged = {}
    for result_set in result_sets:
        for rank, hit in enumerate(result_set, start=1):
            item = merged.setdefault(hit["chunk_id"], {**hit, "score": 0.0, "methods": []})
            item["score"] += 1.0 / (60 + rank)
            item["methods"] = sorted(set(item["methods"] + hit["methods"]))
    return sorted(merged.values(), key=lambda hit: hit["score"], reverse=True)[:top_k]
