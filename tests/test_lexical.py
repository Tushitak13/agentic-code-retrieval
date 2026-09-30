from src.models import CodeChunk
from src.retrieval.lexical import LexicalRetriever


chunks = [
    CodeChunk(
        "1",
        "auth.js",
        1,
        10,
        "function authenticateUser(credentials) {}",
    ),

    CodeChunk(
        "2",
        "weather.js",
        1,
        10,
        "function getWeatherTemperature() {}",
    ),

    CodeChunk(
        "3",
        "payment.js",
        1,
        10,
        "function processPayment() {}",
    ),
]


retriever = LexicalRetriever(
    chunks
)


results = retriever.search(
    "authenticate user",
    top_k=3,
)


for result in results:

    print(
        result.score,
        result.chunk.file_path,
    )