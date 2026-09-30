from ..query.intent import classify_query


class RetrievalAgent:

    def __init__(
        self,
        semantic,
        lexical,
        structural,
        reranker,
        verifier,
        rrf,
        max_passes=2,
    ):

        self.semantic = semantic
        self.lexical = lexical
        self.structural = structural
        self.reranker = reranker
        self.verifier = verifier
        self.rrf = rrf
        self.max_passes = max_passes

    def search(
        self,
        query,
        final_k=10,
    ):

        intent = classify_query(
            query
        )

        # PASS 1
        semantic_results = (
            self.semantic.search(
                query,
                top_k=50,
            )
        )

        lexical_results = (
            self.lexical.search(
                query,
                top_k=50,
            )
        )

        result_lists = [
            semantic_results,
            lexical_results,
        ]

        if (
            intent["structural"]
            or intent["usage"]
            or intent["ordering"]
        ):

            structural_results = (
                self.structural.search(
                    query,
                    symbols=intent[
                        "symbols"
                    ],
                    ordering=intent[
                        "ordering"
                    ],
                    top_k=30,
                )
            )

            result_lists.append(
                structural_results
            )

        candidates = self.rrf(
            result_lists
        )

        reranked = self.reranker.rerank(
            query,
            candidates[:30],
            top_k=final_k,
        )

        # PASS 2 only for structural queries
        if (
            self.max_passes > 1
            and (
                intent["structural"]
                or intent["usage"]
                or intent["ordering"]
            )
        ):

            verified = []

            for result in reranked:

                evidence = (
                    self.verifier.verify(
                        query,
                        result,
                    )
                )

                result.evidence[
                    "verification"
                ] = evidence

                if evidence > 0:
                    verified.append(
                        result
                    )

            if verified:

                reranked = (
                    self.reranker.rerank(
                        query,
                        verified,
                        top_k=final_k,
                    )
                )

        return reranked, intent