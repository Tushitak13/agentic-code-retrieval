import numpy as np

from ..models import SearchResult


class SemanticRetriever:

    def __init__(self, chunks, embedding_engine):

        self.chunks = chunks
        self.embedding_engine = embedding_engine

        # Build one contiguous float32 matrix.
        # This makes matrix multiplication faster and
        # avoids repeatedly constructing arrays.
        self.matrix = np.asarray(
            [chunk.embedding for chunk in chunks],
            dtype=np.float32,
        )

    def search(self, query, top_k=100):

        query_vector = (
            self.embedding_engine.encode_query(
                query
            )
        )

        # Cosine similarity because embeddings are normalized.
        scores = self.matrix @ query_vector

        # Do NOT fully sort all documents.
        # argpartition finds the top-k much faster.
        top_k = min(
            top_k,
            len(scores),
        )

        candidate_indices = np.argpartition(
            scores,
            -top_k,
        )[-top_k:]

        # Sort only the selected top-k candidates.
        candidate_indices = candidate_indices[
            np.argsort(
                scores[candidate_indices]
            )[::-1]
        ]

        return [
            SearchResult(
                chunk=self.chunks[i],
                score=float(scores[i]),
                source="semantic",
            )
            for i in candidate_indices
        ]