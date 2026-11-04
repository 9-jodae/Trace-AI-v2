from app.retrieval.fusion import rrf_fuse


def _chunk(chunk_id: str) -> dict:
    return {"id": chunk_id, "text": chunk_id, "source": "문서"}


def test_chunk_found_by_both_sources_ranks_first():
    fused = rrf_fuse([[_chunk("a"), _chunk("b")], [_chunk("c"), _chunk("b")]], limit=3)

    assert fused[0]["id"] == "b"


def test_limit_is_applied():
    fused = rrf_fuse([[_chunk("a"), _chunk("b"), _chunk("c")]], limit=2)

    assert [chunk["id"] for chunk in fused] == ["a", "b"]
