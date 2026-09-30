import re


class StructuralVerifier:

    def verify(
        self,
        query,
        result,
    ):

        text = result.chunk.content.lower()
        query_lower = query.lower()

        evidence = 0.0

        # Symbol evidence
        symbols = re.findall(
            r"`([^`]+)`",
            query,
        )

        if symbols:

            matched = 0

            for symbol in symbols:

                clean = symbol.lower()

                if clean in text:
                    matched += 1

            if symbols:
                evidence += (
                    matched /
                    len(symbols)
                ) * 0.5

        # Ordering evidence
        order_words = [
            "before",
            "after",
            "then",
            "followed by",
        ]

        if any(
            word in query_lower
            for word in order_words
        ):

            if any(
                word in text
                for word in [
                    "before",
                    "after",
                ]
            ):
                evidence += 0.2

        return min(
            evidence,
            1.0,
        )