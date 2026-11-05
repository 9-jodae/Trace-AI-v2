from app.retrieval.schemas import Chunk

RRF_K = 60


def rrf_fuse(rankings: list[list[Chunk]], limit: int) -> list[Chunk]:
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, chunk in enumerate(ranking, start=1):
            if chunk["id"] in seen:
                continue
            seen.add(chunk["id"])
            scores[chunk["id"]] = scores.get(chunk["id"], 0.0) + 1 / (RRF_K + rank)
            chunks.setdefault(chunk["id"], chunk)
    ordered = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [chunks[chunk_id] for chunk_id in ordered[:limit]]


def distinct_documents_first(chunks: list[Chunk], limit: int) -> list[Chunk]:
    """같은 문서의 두 번째 청크가 다른 문서를 밀어내지 않게 문서마다 하나씩 먼저 채운다. 순서는 유지한다."""
    seen: set[str] = set()
    first: list[Chunk] = []
    rest: list[Chunk] = []
    for chunk in chunks:
        key = chunk.get("document_id") or chunk["id"]
        (rest if key in seen else first).append(chunk)
        seen.add(key)
    return [*first, *rest][:limit]
