"""Index the captured Notion planning page without deleting other documents."""
import json
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import models

from .chunking import split_document
from .config import settings
from .storage import DATA_PATH, LocalStores


def main():
    doc = json.loads((DATA_PATH.parent / "notion_planning.json").read_text(encoding="utf-8"))
    chunks = split_document(doc)
    stores = LocalStores()
    try:
        stores.driver.verify_connectivity()
        vectors = stores.embeddings.embed_documents([chunk["text"] for chunk in chunks])
        if stores.client.collection_exists(settings.qdrant_collection):
            existing = stores.client.get_collection(settings.qdrant_collection).config.params.vectors
            if existing.size != len(vectors[0]):
                raise ValueError("기존 Qdrant 컬렉션의 임베딩 차원이 다릅니다. 별도 컬렉션을 지정하세요.")
        else:
            stores.client.create_collection(
                settings.qdrant_collection,
                vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
            )
        with stores.driver.session(database=settings.neo4j_database) as session:
            session.run("CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE").consume()
            session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE").consume()
            session.run("CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text, c.title]").consume()
            session.run("CALL db.awaitIndex('chunk_text', 300)").consume()
            session.run("MERGE (d:Document {doc_id:$doc.doc_id}) SET d += $doc", doc=doc).consume()
            session.run("""
                UNWIND $chunks AS item
                MATCH (d:Document {doc_id:item.doc_id})
                MERGE (c:Chunk {chunk_id:item.chunk_id}) SET c += item
                MERGE (d)-[:HAS_CHUNK]->(c)
            """, chunks=chunks).consume()
        stores.client.upsert(settings.qdrant_collection, points=[
            models.PointStruct(id=str(uuid5(NAMESPACE_URL, chunk["chunk_id"])), vector=vector, payload=chunk)
            for chunk, vector in zip(chunks, vectors)
        ], wait=True)
        print(f"적재 완료: {doc['title']} / 청크 {len(chunks)}개")
    finally:
        stores.close()


if __name__ == "__main__":
    main()
