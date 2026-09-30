from src.chunker import chunk_javascript


def test_chunking():

    code = """
function hello() {
    console.log("hello");
}

function world() {
    console.log("world");
}
"""

    chunks = chunk_javascript(
        "test.js",
        code,
        max_lines=5,
        overlap_lines=1,
    )

    assert len(chunks) > 0

    print("\nChunks created:")

    for chunk in chunks:
        print(
            chunk.file_path,
            chunk.start_line,
            chunk.end_line,
        )