import math
from collections import Counter, defaultdict

import numpy as np

from ..models import SearchResult


class LexicalRetriever:

    def __init__(self, chunks, k1=1.5, b=0.75):

        self.chunks = chunks
        self.k1 = k1
        self.b = b

        self.doc_count = len(chunks)

        self.documents = []
        self.postings = defaultdict(list)

        self.doc_lengths = np.zeros(
            self.doc_count,
            dtype=np.int32,
        )

        # ---------------------------------------------
        # BUILD INVERTED INDEX
        # ---------------------------------------------

        total_length = 0

        for doc_index, chunk in enumerate(chunks):

            tokens = self._tokenize(
                chunk.content
            )

            counts = Counter(tokens)

            self.documents.append(counts)

            doc_length = len(tokens)

            self.doc_lengths[
                doc_index
            ] = doc_length

            total_length += doc_length

            for term, frequency in counts.items():

                self.postings[term].append(
                    (
                        doc_index,
                        frequency,
                    )
                )

        self.avg_doc_length = (
            total_length / self.doc_count
            if self.doc_count
            else 0
        )

        # ---------------------------------------------
        # IDF CACHE
        # ---------------------------------------------

        self.idf = {}

        for term, postings in self.postings.items():

            document_frequency = len(postings)

            self.idf[term] = math.log(
                1
                + (
                    self.doc_count
                    - document_frequency
                    + 0.5
                )
                / (
                    document_frequency
                    + 0.5
                )
            )

        print(
            f"BM25 inverted index built: "
            f"{len(self.postings)} unique terms"
        )

    @staticmethod
    def _tokenize(text):

        return (
            text.lower()
            .replace("(", " ")
            .replace(")", " ")
            .replace("{", " ")
            .replace("}", " ")
            .replace("[", " ")
            .replace("]", " ")
            .replace(".", " ")
            .replace(",", " ")
            .replace(":", " ")
            .replace(";", " ")
            .replace("/", " ")
            .replace("\\", " ")
            .replace("\"", " ")
            .replace("'", " ")
            .split()
        )

    def search(self, query, top_k=100):

        tokens = self._tokenize(query)

        if not tokens:
            return []

        query_terms = Counter(tokens)

        scores = defaultdict(float)

        # ---------------------------------------------
        # ONLY SCORE DOCUMENTS THAT CONTAIN
        # QUERY TERMS
        # ---------------------------------------------

        for term, query_frequency in query_terms.items():

            postings = self.postings.get(term)

            if not postings:
                continue

            idf = self.idf[term]

            for doc_index, term_frequency in postings:

                doc_length = self.doc_lengths[
                    doc_index
                ]

                denominator = (
                    term_frequency
                    + self.k1
                    * (
                        1
                        - self.b
                        + self.b
                        * (
                            doc_length
                            / self.avg_doc_length
                        )
                    )
                )

                score = (
                    idf
                    * (
                        (
                            term_frequency
                            * (self.k1 + 1)
                        )
                        / denominator
                    )
                )

                scores[doc_index] += score

        if not scores:
            return []

        # ---------------------------------------------
        # TOP-K ONLY
        # ---------------------------------------------

        indices = np.fromiter(
            scores.keys(),
            dtype=np.int32,
        )

        values = np.fromiter(
            (
                scores[i]
                for i in indices
            ),
            dtype=np.float32,
        )

        top_k = min(
            top_k,
            len(values),
        )

        if len(values) > top_k:

            selected = np.argpartition(
                values,
                -top_k,
            )[-top_k:]

            selected = selected[
                np.argsort(
                    values[selected]
                )[::-1]
            ]

        else:

            selected = np.argsort(
                values
            )[::-1]

        # ---------------------------------------------
        # SEARCH RESULTS
        # ---------------------------------------------

        results = []

        for position in selected:

            doc_index = int(
                indices[position]
            )

            results.append(
                SearchResult(
                    chunk=self.chunks[
                        doc_index
                    ],
                    score=float(
                        values[position]
                    ),
                    source="lexical",
                )
            )

        return results