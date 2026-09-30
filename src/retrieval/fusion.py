from ..models import SearchResult


def apply_structural_verification(
    results,
    verifier,
    query,
    structural_weight=0.20,
):

    for result in results:

        structural_score = (
            verifier.verify(
                query,
                result,
            )
        )

        original = result.score

        result.evidence[
            "retrieval_score"
        ] = original

        result.evidence[
            "structural"
        ] = structural_score

        result.score = (
            (1 - structural_weight)
            * original
            +
            structural_weight
            * structural_score
        )

    results.sort(
        key=lambda x: x.score,
        reverse=True,
    )

    return results