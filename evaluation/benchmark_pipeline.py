import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import mteb

from src.models import CodeChunk
from src.indexing.embeddings import EmbeddingEngine
from src.retrieval.semantic import SemanticRetriever
from src.retrieval.lexical import LexicalRetriever


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

SEMANTIC_TOP_K = 100
LEXICAL_TOP_K = 100

RRF_K = 60

# None = evaluate ALL queries
MAX_QUERIES = None

# CPU-friendly batch size
EMBED_BATCH_SIZE = 64

# Cache
CACHE_DIR = Path(
    "evaluation/cache"
)

EMBEDDING_CACHE = (
    CACHE_DIR /
    "apps_retrieval_embeddings.npy"
)

ID_CACHE = (
    CACHE_DIR /
    "apps_retrieval_ids.json"
)

OUTPUT_FILE = Path(
    "evaluation/full_hybrid_results.json"
)


# ============================================================
# METRICS
# ============================================================

def ndcg_at_k(
    ranked_ids,
    relevant_docs,
    k=10,
):
    ranked_ids = ranked_ids[:k]

    dcg = 0.0

    for rank, doc_id in enumerate(
        ranked_ids,
        start=1,
    ):
        relevance = relevant_docs.get(
            doc_id,
            0,
        )

        if relevance > 0:
            dcg += (
                relevance
                / np.log2(rank + 1)
            )

    ideal = sorted(
        relevant_docs.values(),
        reverse=True,
    )[:k]

    idcg = 0.0

    for rank, relevance in enumerate(
        ideal,
        start=1,
    ):
        if relevance > 0:
            idcg += (
                relevance
                / np.log2(rank + 1)
            )

    if idcg == 0:
        return 0.0

    return dcg / idcg


def mrr_at_k(
    ranked_ids,
    relevant_docs,
    k=10,
):
    for rank, doc_id in enumerate(
        ranked_ids[:k],
        start=1,
    ):
        if relevant_docs.get(
            doc_id,
            0,
        ) > 0:
            return 1.0 / rank

    return 0.0


# ============================================================
# RRF
# ============================================================

def full_rrf(
    result_lists,
    k=60,
):
    scores = defaultdict(float)
    objects = {}

    for results in result_lists:

        for rank, result in enumerate(
            results,
            start=1,
        ):
            chunk_id = str(
                result.chunk.chunk_id
            )

            scores[chunk_id] += (
                1.0
                / (k + rank)
            )

            objects[chunk_id] = result

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    output = []

    for chunk_id, score in ranked:
        output.append(
            objects[chunk_id]
        )

    return output


# ============================================================
# EMBEDDING CACHE
# ============================================================

def load_or_create_embeddings(
    embedding_engine,
    documents,
):
    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    document_ids = [
        str(doc.chunk_id)
        for doc in documents
    ]

    # --------------------------------------------------------
    # Check cache
    # --------------------------------------------------------

    if (
        EMBEDDING_CACHE.exists()
        and ID_CACHE.exists()
    ):

        print()
        print(
            "Checking embedding cache..."
        )

        try:

            with open(
                ID_CACHE,
                "r",
                encoding="utf-8",
            ) as f:

                cached_ids = json.load(f)

            if cached_ids == document_ids:

                print(
                    "Embedding cache matches corpus."
                )

                embeddings = np.load(
                    EMBEDDING_CACHE
                )

                print(
                    "Loaded cached embeddings:",
                    embeddings.shape,
                )

                return embeddings

            print(
                "Cache does not match current corpus."
            )

        except Exception as e:

            print(
                "Cache loading failed:",
                e,
            )

    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    print()
    print(
        "Creating corpus embeddings..."
    )

    texts = [
        doc.content
        for doc in documents
    ]

    start = time.perf_counter()

    embeddings = embedding_engine.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print()
    print(
        f"Embedding time: "
        f"{elapsed:.2f} seconds"
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print(
        "Saving embedding cache..."
    )

    np.save(
        EMBEDDING_CACHE,
        embeddings,
    )

    with open(
        ID_CACHE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            document_ids,
            f,
        )

    print(
        "Embedding cache saved."
    )

    return embeddings


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "FULL CPU HYBRID CODE RETRIEVAL BENCHMARK"
    )
    print("=" * 70)

    print()
    print(
        "Embedding model:",
        EMBEDDING_MODEL,
    )

    print(
        "Semantic candidates:",
        SEMANTIC_TOP_K,
    )

    print(
        "Lexical candidates:",
        LEXICAL_TOP_K,
    )

    print(
        "RRF candidate pool:",
        "ALL",
    )

    print(
        "Cross encoder:",
        "DISABLED",
    )

    print(
        "Queries:",
        "ALL",
    )

    # ========================================================
    # LOAD MTEB
    # ========================================================

    print()
    print(
        "Loading MTEB AppsRetrieval..."
    )

    task = mteb.get_task(
        "AppsRetrieval"
    )

    task.load_data()

    dataset = task.dataset

    # Your installed MTEB version stores:
    #
    # dataset
    # └── default
    #     └── test
    #         ├── corpus
    #         ├── queries
    #         └── relevant_docs

    test_data = dataset[
        "default"
    ][
        "test"
    ]

    corpus = test_data[
        "corpus"
    ]

    queries = test_data[
        "queries"
    ]

    qrels = test_data[
        "relevant_docs"
    ]

    print()
    print(
        "Corpus documents:",
        len(corpus),
    )

    print(
        "Total queries:",
        len(queries),
    )

    # ========================================================
    # PREPARE DOCUMENTS
    # ========================================================

    print()
    print(
        "Preparing documents..."
    )

    documents = []

    # Hugging Face Dataset is iterable.
    # It does NOT have .items().

    for row in corpus:

        doc_id = str(
            row["id"]
        )

        documents.append(
            CodeChunk(
                chunk_id=doc_id,
                file_path=doc_id,
                start_line=1,
                end_line=1,
                content=row["text"],
            )
        )

    print(
        "Prepared documents:",
        len(documents),
    )

    # ========================================================
    # EMBEDDING ENGINE
    # ========================================================

    print()
    print(
        "Loading embedding model..."
    )

    embedding_engine = EmbeddingEngine(
        EMBEDDING_MODEL
    )

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    corpus_embeddings = (
        load_or_create_embeddings(
            embedding_engine,
            documents,
        )
    )

    # Attach embeddings
    for i, chunk in enumerate(
        documents
    ):

        chunk.embedding = (
            corpus_embeddings[i]
        )

    # ========================================================
    # RETRIEVERS
    # ========================================================

    print()
    print(
        "Building semantic retriever..."
    )

    semantic_retriever = (
        SemanticRetriever(
            documents,
            embedding_engine,
        )
    )

    print(
        "Building BM25 retriever..."
    )

    lexical_retriever = (
        LexicalRetriever(
            documents
        )
    )

    print()
    print(
        "Retrievers ready."
    )

    # ========================================================
    # QUERIES
    # ========================================================

    # Hugging Face Dataset is iterable.
    # It does NOT have .items().

    if MAX_QUERIES is None:

        query_rows = queries

    else:

        query_rows = queries.select(
            range(
                min(
                    MAX_QUERIES,
                    len(queries),
                )
            )
        )

    print()
    print(
        "Queries evaluated:",
        len(query_rows),
    )

    # ========================================================
    # EVALUATION
    # ========================================================

    ndcg_scores = []
    mrr_scores = []

    query_results = {}

    total_start = (
        time.perf_counter()
    )

    print()
    print("=" * 70)
    print(
        "STARTING FULL EVALUATION"
    )
    print("=" * 70)

    for query_number, row in enumerate(
        query_rows,
        start=1,
    ):

        query_id = str(
            row["id"]
        )

        query_text = row[
            "text"
        ]

        # ----------------------------------------------------
        # Semantic
        # ----------------------------------------------------

        semantic_results = (
            semantic_retriever.search(
                query_text,
                top_k=SEMANTIC_TOP_K,
            )
        )

        # ----------------------------------------------------
        # BM25
        # ----------------------------------------------------

        lexical_results = (
            lexical_retriever.search(
                query_text,
                top_k=LEXICAL_TOP_K,
            )
        )

        # ----------------------------------------------------
        # Full RRF
        # ----------------------------------------------------

        final_results = full_rrf(
            [
                semantic_results,
                lexical_results,
            ],
            k=RRF_K,
        )

        ranked_ids = [
            str(
                result.chunk.chunk_id
            )
            for result in final_results
        ]

        # ----------------------------------------------------
        # Relevant documents
        # ----------------------------------------------------

        relevant = qrels.get(
            query_id,
            {},
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        ndcg = ndcg_at_k(
            ranked_ids,
            relevant,
            k=10,
        )

        mrr = mrr_at_k(
            ranked_ids,
            relevant,
            k=10,
        )

        ndcg_scores.append(
            ndcg
        )

        mrr_scores.append(
            mrr
        )

        query_results[
            query_id
        ] = {
            "query": query_text,
            "ranked_documents": ranked_ids,
            "relevant_documents": relevant,
            "ndcg_at_10": ndcg,
            "mrr_at_10": mrr,
        }

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        elapsed = (
            time.perf_counter()
            - total_start
        )

        average = (
            elapsed
            / query_number
        )

        remaining = (
            average
            * (
                len(query_rows)
                - query_number
            )
        )

        if (
            query_number % 50 == 0
            or query_number
            == len(query_rows)
        ):

            print(
                f"Processed "
                f"{query_number}/"
                f"{len(query_rows)} | "
                f"Avg: "
                f"{average:.3f}s/query | "
                f"Remaining: "
                f"{remaining / 60:.1f} min"
            )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    total_time = (
        time.perf_counter()
        - total_start
    )

    mean_ndcg = float(
        np.mean(ndcg_scores)
    )

    mean_mrr = float(
        np.mean(mrr_scores)
    )

    # ========================================================
    # SAVE
    # ========================================================

    output = {

        "experiment":
            "full_cpu_hybrid_rrf",

        "embedding_model":
            EMBEDDING_MODEL,

        "semantic_top_k":
            SEMANTIC_TOP_K,

        "lexical_top_k":
            LEXICAL_TOP_K,

        "rrf_limit":
            "ALL",

        "reranker":
            None,

        "corpus_documents":
            len(documents),

        "queries_evaluated":
            len(query_rows),

        "ndcg_at_10":
            mean_ndcg,

        "mrr_at_10":
            mean_mrr,

        "total_query_time_seconds":
            total_time,

        "average_query_time_seconds":
            total_time
            / len(query_rows),

        "queries":
            query_results,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print(
        "FULL CPU HYBRID RESULTS"
    )
    print("=" * 70)

    print(
        f"NDCG@10: "
        f"{mean_ndcg:.6f}"
    )

    print(
        f"MRR@10:   "
        f"{mean_mrr:.6f}"
    )

    print(
        f"Total query time: "
        f"{total_time:.2f} seconds"
    )

    print(
        f"Average query time: "
        f"{total_time / len(query_rows):.4f}"
        f" seconds/query"
    )

    print(
        f"Corpus documents: "
        f"{len(documents)}"
    )

    print(
        f"Queries evaluated: "
        f"{len(query_rows)}"
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()