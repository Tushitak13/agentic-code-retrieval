import json
import numpy as np

from .models import CodeChunk, StructuralIndex
from .indexing.embeddings import EmbeddingEngine
from .retrieval.semantic import SemanticRetriever
from .retrieval.lexical import LexicalRetriever
from .retrieval.structural import StructuralRetriever
from .retrieval.reranker import CrossEncoderReranker
from .retrieval.rrf import reciprocal_rank_fusion
from .verification.verifier import StructuralVerifier
from .agent.retrieval_agent import RetrievalAgent


class RetrievalPipeline:

    def __init__(
        self,
        index_dir,
        embedding_model,
        reranker_model,
    ):

        self.index_dir = index_dir

        self.embedding_engine = (
            EmbeddingEngine(
                embedding_model
            )
        )

        self.chunks = self._load_chunks()

        self.structural_index = (
            self._load_structural()
        )

        self.semantic = (
            SemanticRetriever(
                self.chunks,
                self.embedding_engine,
            )
        )

        self.lexical = (
            LexicalRetriever(
                self.chunks
            )
        )

        self.structural = (
            StructuralRetriever(
                self.chunks,
                self.structural_index,
            )
        )

        self.reranker = (
            CrossEncoderReranker(
                reranker_model
            )
        )

        self.verifier = (
            StructuralVerifier()
        )

        self.agent = RetrievalAgent(
            semantic=self.semantic,
            lexical=self.lexical,
            structural=self.structural,
            reranker=self.reranker,
            verifier=self.verifier,
            rrf=reciprocal_rank_fusion,
        )

    def _load_chunks(self):

        with open(
            f"{self.index_dir}/chunks.json",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        chunks = []

        for item in data:

            chunk = CodeChunk(
                chunk_id=item["chunk_id"],
                file_path=item["file_path"],
                start_line=item["start_line"],
                end_line=item["end_line"],
                content=item["content"],
            )

            chunk.embedding = np.asarray(
                item["embedding"],
                dtype=np.float32,
            )

            chunks.append(chunk)

        return chunks

    def _load_structural(self):

        with open(
            f"{self.index_dir}/structural.json",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        index = StructuralIndex()

        from .models import Symbol, CallEdge

        for name, entries in (
            data["symbols"].items()
        ):

            for entry in entries:

                index.add_symbol(
                    Symbol(
                        name=entry["name"],
                        kind=entry["kind"],
                        file_path=entry["file_path"],
                        line=entry["line"],
                    )
                )

        for edge in data["calls"]:

            index.add_call(
                CallEdge(
                    caller=edge["caller"],
                    callee=edge["callee"],
                    file_path=edge["file_path"],
                    line=edge["line"],
                )
            )

        index.inheritance = data.get(
            "inheritance",
            {},
        )

        return index

    def search(
        self,
        query,
        top_k=10,
    ):

        return self.agent.search(
            query,
            final_k=top_k,
        )