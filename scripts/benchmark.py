import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import mteb

from src.models import CodeChunk
from src.indexing.embeddings import EmbeddingEngine
from src.retrieval.lexical import LexicalRetriever


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SEMANTIC_TOP_K = 100
LEXICAL_TOP_K = 100
RRF_K = 60

EMBED_BATCH_SIZE = 128
QUERY_BATCH_SIZE = 128

CACHE_DIR = Path("evaluation/cache")
EMBEDDING_CACHE = CACHE_DIR / "apps_retrieval_embeddings.npy"
ID_CACHE = CACHE_DIR / "apps_retrieval_ids.json"

OUTPUT_FILE = Path("evaluation/full_hybrid_results.json")


def ndcg_at_k(ranked_ids, relevant_docs, k=10):
    ranked_ids = ranked_ids[:k]

    dcg = 0.0

    for rank, doc_id in enumerate(ranked_ids, start=1):
        relevance = relevant_docs.get(doc_id, 0)

        if relevance > 0:
            dcg += relevance / np.log2(rank + 1)

    ideal = sorted(relevant_docs.values(), reverse=True)[:k]

    idcg = 0.0

    for rank, relevance in enumerate(ideal, start=1):
        if relevance > 0:
            idcg += relevance / np.log2(rank + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def mrr_at_k(ranked_ids, relevant_docs, k=10):
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if relevant_docs.get(doc_id, 0) > 0:
            return 1.0 / rank

    return 0.0


def full_rrf(result_lists, k=60):
    scores = defaultdict(float)
    objects = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):

            chunk_id = str(result.chunk.chunk_id)

            scores[chunk_id] += 1.0 / (k + rank)

            objects[chunk_id] = result

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return [objects[chunk_id] for chunk_id, _ in ranked]


def load_or_create_embeddings(
    embedding_engine,
    documents,
):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    document_ids = [
        str(doc.chunk_id)
        for doc in documents
    ]

    if EMBEDDING_CACHE.exists() and ID_CACHE.exists():

        print("Checking embedding cache...")

        try:
            with open(ID_CACHE, "r", encoding="utf-8") as f:
                cached_ids = json.load(f)

            if cached_ids == document_ids:

                embeddings = np.load(
                    EMBEDDING_CACHE,
                    mmap_mode="r",
                )

                print(
                    "Loaded cached embeddings:",
                    embeddings.shape,
                )

                return np.asarray(
                    embeddings,
                    dtype=np.float32,
                )

        except Exception as e:
            print("Cache loading failed:", e)

    print("Creating corpus embeddings...")

    texts = [
        doc.content
        for doc in documents
    ]

    start = time.perf_counter()

    embeddings = embedding_engine.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
    )

    elapsed = time.perf_counter() - start

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    print(
        f"Corpus embedding time: "
        f"{elapsed:.2f}s"
    )

    np.save(
        EMBEDDING_CACHE,
        embeddings,
    )

    with open(ID_CACHE, "w", encoding="utf-8") as f:
        json.dump(document_ids, f)

    return embeddings


def semantic_search_batch(
    query_embeddings,
    corpus_embeddings,
    top_k=100,
):
    """
    Fast batched semantic retrieval.

    Instead of:

        query -> matrix multiplication

    3765 separate times,

    this processes a batch of queries together.
    """

    results = []

    n_documents = corpus_embeddings.shape[0]

    top_k = min(
        top_k,
        n_documents,
    )

    for query_vector in query_embeddings:

        scores = corpus_embeddings @ query_vector

        indices = np.argpartition(
            scores,
            -top_k,
        )[-top_k:]

        indices = indices[
            np.argsort(
                scores[indices]
            )[::-1]
        ]

        results.append(
            (
                indices,
                scores[indices],
            )
        )

    return results


def main():

    print()
    print("=" * 70)
    print("FAST CPU HYBRID CODE RETRIEVAL BENCHMARK")
    print("=" * 70)

    print()
    print("Embedding model:", EMBEDDING_MODEL)
    print("Semantic top-k:", SEMANTIC_TOP_K)
    print("Lexical top-k:", LEXICAL_TOP_K)
    print("RRF:", "ALL")
    print("Reranker:", "DISABLED")
    print("Query batch size:", QUERY_BATCH_SIZE)

    # --------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------

    print()
    print("Loading MTEB AppsRetrieval...")

    task = mteb.get_task("AppsRetrieval")
    task.load_data()

    test_data = task.dataset["default"]["test"]

    corpus = test_data["corpus"]
    queries = test_data["queries"]
    qrels = test_data["relevant_docs"]

    print()
    print("Corpus:", len(corpus))
    print("Queries:", len(queries))

    # --------------------------------------------------
    # PREPARE CORPUS
    # --------------------------------------------------

    print()
    print("Preparing corpus...")

    documents = []

    for row in corpus:

        doc_id = str(row["id"])

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
        "Prepared:",
        len(documents),
        "documents",
    )

    # --------------------------------------------------
    # EMBEDDING MODEL
    # --------------------------------------------------

    print()
    print("Loading embedding model...")

    embedding_engine = EmbeddingEngine(
        EMBEDDING_MODEL
    )

    # --------------------------------------------------
    # CORPUS EMBEDDINGS
    # --------------------------------------------------

    corpus_embeddings = load_or_create_embeddings(
        embedding_engine,
        documents,
    )

    # --------------------------------------------------
    # BM25
    # --------------------------------------------------

    print()
    print("Building BM25...")

    lexical_retriever = LexicalRetriever(
        documents
    )

    # --------------------------------------------------
    # LOAD ALL QUERIES
    # --------------------------------------------------

    query_ids = [
        str(row["id"])
        for row in queries
    ]

    query_texts = [
        row["text"]
        for row in queries
    ]

    print()
    print(
        "Encoding",
        len(query_texts),
        "queries in batches..."
    )

    # --------------------------------------------------
    # BATCH QUERY EMBEDDINGS
    # --------------------------------------------------

    start = time.perf_counter()

    query_embeddings = embedding_engine.encode(
        query_texts,
        batch_size=EMBED_BATCH_SIZE,
    )

    query_embeddings = np.asarray(
        query_embeddings,
        dtype=np.float32,
    )

    embedding_time = (
        time.perf_counter() - start
    )

    print(
        f"Query embedding time: "
        f"{embedding_time:.2f}s"
    )

    # --------------------------------------------------
    # EVALUATION
    # --------------------------------------------------

    ndcg_scores = []
    mrr_scores = []

    query_results = {}

    total_start = time.perf_counter()

    total_queries = len(query_texts)

    print()
    print("=" * 70)
    print("STARTING FAST EVALUATION")
    print("=" * 70)

    for batch_start in range(
        0,
        total_queries,
        QUERY_BATCH_SIZE,
    ):

        batch_end = min(
            batch_start + QUERY_BATCH_SIZE,
            total_queries,
        )

        batch_embeddings = query_embeddings[
            batch_start:batch_end
        ]

        batch_texts = query_texts[
            batch_start:batch_end
        ]

        batch_ids = query_ids[
            batch_start:batch_end
        ]

        # --------------------------------------------------
        # SEMANTIC RETRIEVAL
        # --------------------------------------------------

        semantic_batch = semantic_search_batch(
            batch_embeddings,
            corpus_embeddings,
            SEMANTIC_TOP_K,
        )

        # --------------------------------------------------
        # PROCESS EACH QUERY
        # --------------------------------------------------

        for local_index in range(
            len(batch_texts)
        ):

            global_index = (
                batch_start + local_index
            )

            query_id = batch_ids[
                local_index
            ]

            query_text = batch_texts[
                local_index
            ]

            semantic_indices, semantic_scores = (
                semantic_batch[local_index]
            )

            semantic_results = []

            for i, score in zip(
                semantic_indices,
                semantic_scores,
            ):

                semantic_results.append(
                    type(
                        "Result",
                        (),
                        {
                            "chunk": documents[i],
                            "score": float(score),
                            "source": "semantic",
                        },
                    )()
                )

            # --------------------------------------------------
            # BM25
            # --------------------------------------------------

            lexical_results = lexical_retriever.search(
                query_text,
                top_k=LEXICAL_TOP_K,
            )

            # --------------------------------------------------
            # RRF
            # --------------------------------------------------

            final_results = full_rrf(
                [
                    semantic_results,
                    lexical_results,
                ],
                k=RRF_K,
            )

            ranked_ids = [
                str(result.chunk.chunk_id)
                for result in final_results
            ]

            relevant = qrels.get(
                query_id,
                {},
            )

            ndcg = ndcg_at_k(
                ranked_ids,
                relevant,
                10,
            )

            mrr = mrr_at_k(
                ranked_ids,
                relevant,
                10,
            )

            ndcg_scores.append(ndcg)
            mrr_scores.append(mrr)

            query_results[query_id] = {
                "query": query_text,
                "ranked_documents": ranked_ids,
                "relevant_documents": relevant,
                "ndcg_at_10": ndcg,
                "mrr_at_10": mrr,
            }

        # --------------------------------------------------
        # PROGRESS
        # --------------------------------------------------

        processed = batch_end

        elapsed = (
            time.perf_counter()
            - total_start
        )

        avg = elapsed / processed

        remaining = (
            avg
            * (total_queries - processed)
        )

        print(
            f"Processed "
            f"{processed}/{total_queries} | "
            f"Avg: {avg:.3f}s/query | "
            f"Remaining: "
            f"{remaining / 60:.1f} min"
        )

    # --------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------

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

    output = {
        "experiment": "fast_cpu_hybrid_rrf",

        "embedding_model": EMBEDDING_MODEL,

        "semantic_top_k": SEMANTIC_TOP_K,

        "lexical_top_k": LEXICAL_TOP_K,

        "rrf_limit": "ALL",

        "reranker": None,

        "corpus_documents": len(documents),

        "queries_evaluated": total_queries,

        "ndcg_at_10": mean_ndcg,

        "mrr_at_10": mean_mrr,

        "query_embedding_time_seconds":
            embedding_time,

        "total_query_time_seconds":
            total_time,

        "average_query_time_seconds":
            total_time / total_queries,

        "queries": query_results,
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

    print()
    print("=" * 70)
    print("FAST HYBRID RESULTS")
    print("=" * 70)

    print(
        f"NDCG@10: {mean_ndcg:.6f}"
    )

    print(
        f"MRR@10:   {mean_mrr:.6f}"
    )

    print(
        f"Total query time: "
        f"{total_time:.2f}s"
    )

    print(
        f"Average query time: "
        f"{total_time / total_queries:.4f}s/query"
    )

    print(
        f"Corpus documents: "
        f"{len(documents)}"
    )

    print(
        f"Queries evaluated: "
        f"{total_queries}"
    )

    print()
    print(
        "Results saved to:"
    )
    print(OUTPUT_FILE)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()