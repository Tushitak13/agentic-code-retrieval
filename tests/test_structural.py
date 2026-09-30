from src.indexing.structural import JavaScriptStructuralIndexer
from src.models import StructuralIndex


code = """
function authenticate() {
    return true;
}

function getUserProfile() {
    authenticate();
    return {};
}

function main() {
    getUserProfile();
}
"""


index = StructuralIndex()

indexer = JavaScriptStructuralIndexer()

indexer.index_file(
    "test.js",
    code,
    index,
)


print("\nSYMBOLS:")

for name, symbols in index.symbols.items():

    for symbol in symbols:

        print(
            name,
            symbol.kind,
            symbol.file_path,
            symbol.line,
        )


print("\nCALLS:")

for call in index.calls:

    print(
        call.caller,
        "->",
        call.callee,
        "line:",
        call.line,
    )