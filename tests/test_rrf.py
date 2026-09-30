from src.models import CodeChunk, SearchResult
from src.retrieval.rrf import reciprocal_rank_fusion


chunk1 = CodeChunk(
    chunk_id="1",
    file_path="a.js",
    start_line=1,
    end_line=10,
    content="function authenticate() {}",
)

chunk2 = CodeChunk(
    chunk_id="2",
    file_path="b.js",
    start_line=1,
    end_line=10,
    content="function login() {}",
)

chunk3 = CodeChunk(
    chunk_id="3",
    file_path="c.js",
    start_line=1,
    end_line=10,
    content="function logout() {}",
)


semantic = [
    SearchResult(chunk1, 0.90),
    SearchResult(chunk2, 0.80),
]

lexical = [
    SearchResult(chunk1, 10),
    SearchResult(chunk3, 8),
]


results = reciprocal_rank_fusion(
    [
        semantic,
        lexical,
    ]
)


print("\nRRF RESULTS:")

for result in results:

    print(
        result.chunk.file_path,
        result.score,
    )