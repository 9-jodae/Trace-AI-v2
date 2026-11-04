from app.retrieval.chunking import CHUNK_SIZE, chunk_document


def test_chunks_have_sequential_ids_and_source():
    chunks = chunk_document("doc", "가나다라. " * 400, "문서")

    assert len(chunks) > 1
    assert [chunk["id"] for chunk in chunks] == [f"doc-{index}" for index in range(1, len(chunks) + 1)]
    assert all(chunk["source"] == "문서" for chunk in chunks)
    assert all(len(chunk["text"]) <= CHUNK_SIZE for chunk in chunks)
