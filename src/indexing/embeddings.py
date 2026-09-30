import numpy as np

from sentence_transformers import SentenceTransformer


class EmbeddingEngine:

    def __init__(self, model_name: str):

        self.model = SentenceTransformer(
            model_name
        )

    def encode(
        self,
        texts,
        batch_size=64,
    ):

        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        return np.asarray(
            vectors,
            dtype=np.float32,
        )

    def encode_query(
        self,
        query,
    ):

        vector = self.model.encode(
            [query],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]

        return np.asarray(
            vector,
            dtype=np.float32,
        )