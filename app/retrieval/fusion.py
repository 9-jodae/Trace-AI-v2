from app.retrieval.schemas import Chunk

RRF_K = 60


def rrf_fuse_with_trace(rankings: list[list[Chunk]], limit: int) -> tuple[list[Chunk], list[dict]]:
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    source_ranks: dict[str, dict[int, int]] = {}
    for source_index, ranking in enumerate(rankings):
        seen: set[str] = set()
        for rank, chunk in enumerate(ranking, start=1):
            if chunk["id"] in seen:
                continue
            seen.add(chunk["id"])
            scores[chunk["id"]] = scores.get(chunk["id"], 0.0) + 1 / (RRF_K + rank)
            source_ranks.setdefault(chunk["id"], {})[source_index] = rank
            chunks.setdefault(chunk["id"], chunk)
    ordered = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    selected = ordered[:limit]
    trace = [
        {"chunk_id": chunk_id, "rank": rank, "score": scores[chunk_id], "source_ranks": source_ranks[chunk_id]}
        for rank, chunk_id in enumerate(selected, start=1)
    ]
    return [chunks[chunk_id] for chunk_id in selected], trace


def rrf_fuse(rankings: list[list[Chunk]], limit: int) -> list[Chunk]:
    return rrf_fuse_with_trace(rankings, limit)[0]
