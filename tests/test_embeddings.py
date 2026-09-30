from src.indexing.embeddings import EmbeddingEngine


engine = EmbeddingEngine(
    "sentence-transformers/all-MiniLM-L6-v2"
)


texts = [
    "function authenticate user credentials",
    "function calculate weather temperature",
]


vectors = engine.encode(texts)


print("Number of vectors:", len(vectors))
print("Vector dimensions:", vectors.shape)