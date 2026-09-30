from sentence_transformers import CrossEncoder


class CrossEncoderReranker:

    def __init__(self, model_name):

        self.model = CrossEncoder(
            model_name
        )

    def rerank(
        self,
        query,
        candidates,
        top_k=10,
    ):

        if not candidates:
            return []

        pairs = [
            (
                query,
                candidate.chunk.content,
            )
            for candidate in candidates
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        reranked = []

        for candidate, score in zip(
            candidates,
            scores,
        ):

            candidate.score = float(score)

            candidate.evidence[
                "cross_encoder"
            ] = float(score)

            reranked.append(
                candidate
            )

        reranked.sort(
            key=lambda x: x.score,
            reverse=True,
        )

        return reranked[:top_k]