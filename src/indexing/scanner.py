from pathlib import Path
from typing import List


IGNORED_DIRS = {
    "node_modules",
    ".git",
    "dist",
    "build",
    "coverage",
    ".next",
    ".cache",
}


def scan_javascript_files(root: str) -> List[Path]:

    root_path = Path(root)

    files = []

    for path in root_path.rglob("*"):

        if not path.is_file():
            continue

        if any(part in IGNORED_DIRS for part in path.parts):
            continue

        if path.suffix in {".js", ".jsx", ".mjs", ".cjs"}:
            files.append(path)

    return sorted(files)