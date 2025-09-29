#!/usr/bin/env python

import argparse


class ProgramArgs:
    def __init__(
        self,
        test_file: str = "./assets/preprocessed-testing.csv",
        chroma_db_path: str = "/var/lib/chroma/",
        chromadb_collection_name: str = "misinformation",
    ) -> None:
        self.test_file = test_file
        self.chroma_db_path = chroma_db_path
        self.chromadb_collection_name = chromadb_collection_name


def get_args() -> ProgramArgs:
    """Parses the command line arguments. Will call sys.exit() if the -h (help) flag is passed"""
    parser = argparse.ArgumentParser(
        prog="Misinformation Dataset Testeng",
        description="Uses the test database to determine how effective the AI model is at it's job",
    )

    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        help="The testing CSV file from the preprocessor program. Default='./assets/preprocessed-testing.csv'",
        default="./assets/preprocessed-testing.csv",
    )
    parser.add_argument(
        "-c",
        "--chromadb-path",
        dest="chromadb_path",
        help="Path to the ChromaDB database. Default='/var/lib/chroma'",
        default="/var/lib/chroma/",
    )
    parser.add_argument(
        "-d",
        "--chromadb-collection-name",
        dest="chromadb_collection_name",
        help="Path to the ChromaDB database. Default='misinformation'",
        default="misinformation",
    )

    arguments = parser.parse_args()

    return ProgramArgs(
        str(arguments.input),
        str(arguments.chromadb_path),
        str(arguments.chromadb_collection_name),
    )


def main():
    args = get_args()
    print(args)


if __name__ == "__main__":
    main()
