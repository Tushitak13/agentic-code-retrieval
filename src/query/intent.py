import re

STRUCTURAL_PATTERNS = [
    r"\bwhich\s+(files|functions|methods|classes)",
    r"\bextends\b",
    r"\binherit",
    r"\bimplements\b",
    r"\bcall(s|ed|ing)?\b",
    r"\bcaller(s)?\b",
    r"\bcallee(s)?\b",
    r"\bdepends?\s+on\b",
    r"\bdependency\b",
]

USAGE_PATTERNS = [
    r"\bwhere\s+is\b",
    r"\bwhere\s+.*\s+used\b",
    r"\busage\b",
    r"\buses?\b",
    r"\bcalled\b",
]

ORDER_PATTERNS = [
    r"\bbefore\b",
    r"\bafter\b",
    r"\bthen\b",
    r"\bprior\s+to\b",
    r"\bfollowed\s+by\b",
]


def extract_symbols(query):

    symbols = []

    symbols.extend(
        re.findall(
            r"`([^`]+)`",
            query,
        )
    )

    symbols.extend(
        re.findall(
            r"\b[A-Za-z_$][A-Za-z0-9_$]*\(\)",
            query,
        )
    )

    return list(set(symbols))


def classify_query(query):

    q = query.lower()

    structural = any(
        re.search(pattern, q)
        for pattern in STRUCTURAL_PATTERNS
    )

    usage = any(
        re.search(pattern, q)
        for pattern in USAGE_PATTERNS
    )

    ordering = any(
        re.search(pattern, q)
        for pattern in ORDER_PATTERNS
    )

    return {
        "semantic": True,
        "structural": structural,
        "usage": usage,
        "ordering": ordering,
        "symbols": extract_symbols(query),
    }