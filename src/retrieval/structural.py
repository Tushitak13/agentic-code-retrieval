from ..models import SearchResult


class StructuralRetriever:

    def __init__(self, chunks, structural_index):

        self.chunks = chunks
        self.index = structural_index

    def search(
        self,
        query,
        symbols=None,
        ordering=False,
        top_k=30,
    ):

        symbols = symbols or []

        symbol_hits = set()

        for symbol in symbols:

            clean = symbol.replace(
                "()",
                "",
            )

            for name, entries in (
                self.index.symbols.items()
            ):

                if (
                    name == clean
                    or clean.lower() in name.lower()
                ):

                    for entry in entries:
                        symbol_hits.add(
                            entry.file_path
                        )

        result = []

        for chunk in self.chunks:

            score = 0.0

            if chunk.file_path in symbol_hits:
                score += 1.0

            content_lower = (
                chunk.content.lower()
            )

            for symbol in symbols:

                clean = symbol.replace(
                    "()",
                    "",
                ).lower()

                if clean in content_lower:
                    score += 0.5

            if score > 0:

                result.append(
                    SearchResult(
                        chunk=chunk,
                        score=score,
                        source="structural",
                    )
                )

        result.sort(
            key=lambda x: x.score,
            reverse=True,
        )

        return result[:top_k]