from src.models import CodeChunk
from src.indexing.embeddings import EmbeddingEngine
from src.retrieval.semantic import SemanticRetriever


chunks = [
    CodeChunk(
        chunk_id="1",
        file_path="auth.js",
        start_line=1,
        end_line=10,
        content="function authenticateUser(credentials) {}",
    ),

    CodeChunk(
        chunk_id="2",
        file_path="weather.js",
        start_line=1,
        end_line=10,
        content="function getWeatherTemperature() {}",
    ),

    CodeChunk(
        chunk_id="3",
        file_path="payment.js",
        start_line=1,
        end_line=10,
        content="function processPayment() {}",
    ),
]


engine = EmbeddingEngine(
    "sentence-transformers/all-MiniLM-L6-v2"
)


vectors = engine.encode(
    [chunk.content for chunk in chunks]
)


for chunk, vector in zip(chunks, vectors):
    chunk.embedding = vector


retriever = SemanticRetriever(
    chunks,
    engine,
)


results = retriever.search(
    "How does user authentication work?",
    top_k=3,
)


for result in results:

    print(
        result.score,
        result.chunk.file_path,
    )