from dataclasses import dataclass, field
from typing import List, Dict, Set


@dataclass
class CodeChunk:
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    content: str
    embedding: object = None


@dataclass
class Symbol:
    name: str
    kind: str
    file_path: str
    line: int
    parent: str = ""


@dataclass
class CallEdge:
    caller: str
    callee: str
    file_path: str
    line: int


@dataclass
class StructuralIndex:
    symbols: Dict[str, List[Symbol]] = field(default_factory=dict)
    calls: List[CallEdge] = field(default_factory=list)
    inheritance: Dict[str, List[str]] = field(default_factory=dict)

    def add_symbol(self, symbol: Symbol):
        self.symbols.setdefault(symbol.name, []).append(symbol)

    def add_call(self, edge: CallEdge):
        self.calls.append(edge)

    def add_inheritance(self, child: str, parent: str):
        self.inheritance.setdefault(parent, []).append(child)


@dataclass
class SearchResult:
    chunk: CodeChunk
    score: float
    source: str = ""
    evidence: Dict[str, float] = field(default_factory=dict)


@dataclass
class QueryIntent:
    semantic: bool = True
    structural: bool = False
    usage: bool = False
    ordering: bool = False
    symbols: List[str] = field(default_factory=list)