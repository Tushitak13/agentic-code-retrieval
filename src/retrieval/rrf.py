from collections import defaultdict

from ..models import SearchResult


def reciprocal_rank_fusion(
    result_lists,
    k=60,
    limit=50,
):

    scores = defaultdict(float)
    objects = {}

    for results in result_lists:

        for rank, result in enumerate(
            results,
            start=1,
        ):

            chunk_id = result.chunk.chunk_id

            scores[chunk_id] += (
                1.0 /
                (k + rank)
            )

            objects[chunk_id] = result

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    output = []

    for chunk_id, score in ranked[:limit]:

        result = objects[chunk_id]

        output.append(
            SearchResult(
                chunk=result.chunk,
                score=score,
                source="rrf",
            )
        )

    return output