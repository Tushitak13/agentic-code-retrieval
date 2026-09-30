import json
from pathlib import Path

from ..chunker import chunk_javascript
from ..models import StructuralIndex
from .scanner import scan_javascript_files
from .embeddings import EmbeddingEngine
from .structural import JavaScriptStructuralIndexer


class ProjectIndexer:

    def __init__(
        self,
        source_dir,
        index_dir,
        embedding_model,
        chunk_config,
    ):

        self.source_dir = Path(source_dir)
        self.index_dir = Path(index_dir)

        self.embedding_engine = EmbeddingEngine(
            embedding_model
        )

        self.max_lines = chunk_config["max_lines"]
        self.overlap_lines = chunk_config["overlap_lines"]

        self.structural_indexer = (
            JavaScriptStructuralIndexer()
        )

    def build(self):

        self.index_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        files = scan_javascript_files(
            self.source_dir
        )

        all_chunks = []
        structural_index = StructuralIndex()

        for file_path in files:

            try:
                content = file_path.read_text(
                    encoding="utf-8"
                )
            except UnicodeDecodeError:
                continue

            relative_path = str(
                file_path.relative_to(
                    self.source_dir
                )
            )

            chunks = chunk_javascript(
                relative_path,
                content,
                self.max_lines,
                self.overlap_lines,
            )

            all_chunks.extend(chunks)

            self.structural_indexer.index_file(
                relative_path,
                content,
                structural_index,
            )

        texts = [
            chunk.content
            for chunk in all_chunks
        ]

        embeddings = self.embedding_engine.encode(
            texts
        )

        for chunk, vector in zip(
            all_chunks,
            embeddings,
        ):
            chunk.embedding = vector

        self._save_chunks(all_chunks)

        self._save_structural(
            structural_index
        )

        print(
            f"Indexed {len(files)} files"
        )

        print(
            f"Created {len(all_chunks)} chunks"
        )

    def _save_chunks(self, chunks):

        data = []

        for chunk in chunks:

            data.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "content": chunk.content,
                    "embedding": chunk.embedding.tolist(),
                }
            )

        with open(
            self.index_dir / "chunks.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                data,
                f,
            )

    def _save_structural(
        self,
        structural_index,
    ):

        data = {
            "symbols": {},
            "calls": [],
            "inheritance": structural_index.inheritance,
        }

        for name, symbols in (
            structural_index.symbols.items()
        ):

            data["symbols"][name] = [
                {
                    "name": s.name,
                    "kind": s.kind,
                    "file_path": s.file_path,
                    "line": s.line,
                }
                for s in symbols
            ]

        data["calls"] = [
            {
                "caller": edge.caller,
                "callee": edge.callee,
                "file_path": edge.file_path,
                "line": edge.line,
            }
            for edge in structural_index.calls
        ]

        with open(
            self.index_dir / "structural.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                data,
                f,
                indent=2,
            )