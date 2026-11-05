from conftest import chunk

from app.retrieval.fusion import RRF_K, distinct_documents_first, rrf_fuse

A, B, C, D = chunk("a"), chunk("b"), chunk("c"), chunk("d")


def test_empty():
    assert rrf_fuse([], 5) == []
    assert rrf_fuse([[], []], 5) == []


def test_single_ranking_keeps_order():
    assert rrf_fuse([[A, B, C]], 5) == [A, B, C]


def test_item_in_both_rankings_rises():
    assert rrf_fuse([[A, B], [B, C]], 5) == [B, A, C]


def test_deduplicates_across_rankings():
    assert [c["id"] for c in rrf_fuse([[A, B], [A, B]], 5)] == ["a", "b"]


def test_limit():
    assert rrf_fuse([[A, B, C, D]], 2) == [A, B]


def test_limit_zero():
    assert rrf_fuse([[A]], 0) == []


def test_score_formula_prefers_low_rank_sum():
    # c는 양쪽 3위(2/(k+3))라서 한쪽 1위(1/(k+1))보다 높아야 한다
    assert 2 / (RRF_K + 3) > 1 / (RRF_K + 1)
    assert rrf_fuse([[A, B, C], [D, B, C]], 2)[0] == B


def test_tie_is_stable():
    assert rrf_fuse([[A], [B]], 2) == [A, B]


def test_same_chunk_twice_in_one_ranking_counts_once():
    assert rrf_fuse([[A, A, B], [B, C]], 1) == [B]


def test_does_not_modify_input():
    ranking = [A, B]
    rrf_fuse([ranking], 1)
    assert ranking == [A, B]


def _chunk(chunk_id: str) -> dict:
    return {"id": chunk_id, "text": chunk_id, "source": "문서"}


def test_chunk_found_by_both_sources_ranks_first():
    fused = rrf_fuse([[_chunk("a"), _chunk("b")], [_chunk("c"), _chunk("b")]], limit=3)

    assert fused[0]["id"] == "b"


def test_limit_is_applied():
    fused = rrf_fuse([[_chunk("a"), _chunk("b"), _chunk("c")]], limit=2)

    assert [chunk["id"] for chunk in fused] == ["a", "b"]


def test_distinct_documents_fill_first():
    one = {"id": "1a", "document_id": "d1"}
    two = {"id": "1b", "document_id": "d1"}
    other = {"id": "2a", "document_id": "d2"}
    assert distinct_documents_first([one, two, other], 2) == [one, other]
    assert distinct_documents_first([one, two, other], 5) == [one, other, two]


def test_distinct_documents_without_document_id_use_chunk_id():
    assert distinct_documents_first([A, B], 5) == [A, B]
