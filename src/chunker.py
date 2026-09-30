from typing import List
from .models import CodeChunk
import hashlib


def chunk_javascript(
    file_path: str,
    content: str,
    max_lines: int = 80,
    overlap_lines: int = 15,
) -> List[CodeChunk]:

    lines = content.splitlines()

    if not lines:
        return []

    chunks = []
    start = 0

    while start < len(lines):

        end = min(start + max_lines, len(lines))

        chunk_content = "\n".join(lines[start:end])

        chunk_hash = hashlib.sha1(
            f"{file_path}:{start}:{end}:{chunk_content}".encode()
        ).hexdigest()[:16]

        chunks.append(
            CodeChunk(
                chunk_id=chunk_hash,
                file_path=file_path,
                start_line=start + 1,
                end_line=end,
                content=chunk_content,
            )
        )

        if end >= len(lines):
            break

        start = end - overlap_lines

    return chunks