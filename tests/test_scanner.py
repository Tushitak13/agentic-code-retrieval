from src.indexing.scanner import scan_javascript_files


files = scan_javascript_files("data/repo")

print("\nJavaScript files found:")

for file in files:
    print(file)

print(f"\nTotal files: {len(files)}")