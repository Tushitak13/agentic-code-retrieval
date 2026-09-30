import yaml

from src.indexing.indexer import ProjectIndexer


def main():

    with open(
        "config.yaml",
        encoding="utf-8",
    ) as f:

        config = yaml.safe_load(f)

    indexer = ProjectIndexer(
        source_dir=config[
            "project"
        ]["source_dir"],

        index_dir=config[
            "project"
        ]["index_dir"],

        embedding_model=config[
            "embedding"
        ]["model"],

        chunk_config=config[
            "chunking"
        ],
    )

    indexer.build()


if __name__ == "__main__":
    main()