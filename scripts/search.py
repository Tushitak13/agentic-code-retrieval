import sys

from src.pipeline import RetrievalPipeline


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/search.py "your query"')
        return

    query = " ".join(sys.argv[1:])

    pipeline = RetrievalPipeline(
        index_dir="data/index",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )

    results, intent = pipeline.search(
        query,
        top_k=10,
    )

    print("\nQuery:")
    print(query)

    print("\nDetected intent:")
    print(intent)

    print("\nResults:")

    for rank, result in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print(f"File: {result.chunk.file_path}")
        print(
            f"Lines: {result.chunk.start_line}-{result.chunk.end_line}"
        )
        print(f"Score: {result.score}")
        print(f"Source: {result.source}")
        print("Code:")
        print(result.chunk.content)


if __name__ == "__main__":
    main()