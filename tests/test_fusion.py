import pytest
from conftest import chunk

from app.retrieval.fusion import RRF_K, rrf_fuse

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


@pytest.mark.xfail(strict=True, reason="결함 9: 한 검색 결과 안의 중복 청크가 RRF 점수를 두 번 받음")
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
