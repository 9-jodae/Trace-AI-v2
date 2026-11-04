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
