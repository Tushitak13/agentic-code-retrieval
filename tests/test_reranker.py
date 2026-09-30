from src.models import CodeChunk, SearchResult
from src.retrieval.reranker import CrossEncoderReranker


chunks = [

    CodeChunk(
        "1",
        "auth.js",
        1,
        10,
        "function authenticateUser(credentials) { return verifyCredentials(credentials); }",
    ),

    CodeChunk(
        "2",
        "weather.js",
        1,
        10,
        "function getWeatherTemperature(city) { return weatherAPI(city); }",
    ),
]


candidates = [
    SearchResult(chunks[0], 0.50),
    SearchResult(chunks[1], 0.49),
]


reranker = CrossEncoderReranker(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


results = reranker.rerank(
    "How does user authentication work?",
    candidates,
    top_k=2,
)


for result in results:

    print(
        result.score,
        result.chunk.file_path,
    )