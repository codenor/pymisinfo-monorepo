#!/usr/bin/env python

import argparse
import csv

import chromadb
import math
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from rich.progress import Progress


class ProgramArgs:
    def __init__(
        self,
        input_file: str = "./assets/preprocessed.csv",
        ollama_url: str = "http://localhost:11434",
        ollama_model_name: str = "embeddinggemma:latest",
        chroma_db_path: str = "/var/lib/chroma/",
        chromadb_collection_name: str = "misinformation",
        training_percent: float = 0.9
    ) -> None:
        self.input_file = input_file
        self.ollama_url = ollama_url
        self.ollama_model_name = ollama_model_name
        self.chroma_db_path = chroma_db_path
        self.chromadb_collection_name = chromadb_collection_name
        self.training_percent = training_percent


def get_args() -> ProgramArgs:
    """Parses the command line arguments. Will call sys.exit() if the -h (help) flag is passed"""
    parser = argparse.ArgumentParser(
        prog="Misinformation Dataset Importer",
        description="Imports data from the preprocessor into a Chroma database for analysis and machine learning",
        epilog="This program depends on ollama, so please have that installed. Install guide can be found here: https://docs.ollama.com/quickstart",
    )

    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        help="The input CSV file from the preprocessor program. Default='./assets/preprocessed.csv'",
        default="./assets/preprocessed.csv",
    )
    parser.add_argument(
        "-o",
        "--olama-url",
        dest="ollama_url",
        help="The URL to your running ollama instance. Default='http://localhost:11434'",
        default="http://localhost:11434",
    )
    parser.add_argument(
        "-m",
        "--olama-model-name",
        dest="ollama_model_name",
        help="The model name for your ollama instance for embedding generation. Default='embeddinggemma:latest'",
        default="embeddinggemma:latest",
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
    parser.add_argument(
        "-t",
        "--training-percent",
        dest="training_percent",
        help="Amount of records to use from the input files for training data (in decimal form). Default=0.9",
        default=0.9,
    )

    arguments = parser.parse_args()

    return ProgramArgs(
        str(arguments.input),
        str(arguments.ollama_url),
        str(arguments.ollama_model_name),
        str(arguments.chromadb_path),
        str(arguments.chromadb_collection_name),
        float(str(arguments.training_percent)),
    )


def chunks(lst: list, n: int) -> list:
    """
    Yield successive n-sized chunks from lst.
    https://stackoverflow.com/a/312464
    """
    new_lst = []
    for i in range(0, len(lst), n):  # Slice list in steps of n
        new_lst.append(lst[i : i + n])
    return new_lst


def data_from_preprocessed_csv(
    file: str, recordPercent: float
) -> tuple[list, list, list]:
    documents = []
    metadata = []
    ids = []
    records_c = 0

    with open(file, "r") as csvfile:
        records_c = len(csvfile.readlines()) - 1
        records_c = int(math.floor(records_c - (records_c * (1 - recordPercent))))

    print(f"taking first {recordPercent * 100}% of records ({records_c}) for training data")

    with open(file, "r") as csvfile:
        reader = csv.reader(csvfile)
        idx = 0
        with Progress() as p:
            t = p.add_task("generating documents to insert", total=records_c)
            for row in reader:
                # Skip header
                if idx == 0:
                    idx += 1
                    continue
                elif idx >= records_c:
                    p.advance(t)
                    break

                inputs = str(row[0])
                documents.append(inputs)
                metadata.append({"misinformation": str(row[4])})
                ids.append(str(idx))
                p.update(t, advance=1)
                idx += 1
    return ids, documents, metadata


def main():
    # Init chromadb
    args = get_args()

    print(f"initialising and connecting to chromadb at {args.chroma_db_path}")
    ollama_ef = OllamaEmbeddingFunction(
        url=args.ollama_url,
        model_name=args.ollama_model_name,
    )

    client = chromadb.PersistentClient(path=args.chroma_db_path)

    client.delete_collection(args.chromadb_collection_name)
    collection = client.create_collection(args.chromadb_collection_name)

    print(f"collecting data from {args.input_file}")
    ids, documents, metadatas = data_from_preprocessed_csv(args.input_file, args.training_percent)

    # Import into chromadb
    chunk_size = 100
    chunked_documents = chunks(documents, chunk_size)
    chunked_ids = chunks(ids, chunk_size)
    chunked_metadatas = chunks(metadatas, chunk_size)
    with Progress() as p:
        t = p.add_task(
            f"inserting into chromadb in batches of {chunk_size}",
            total=len(chunked_documents),
        )
        for idx in range(0, len(chunked_documents)):
            embeddings = ollama_ef(chunked_documents[idx])
            collection.add(
                documents=chunked_documents[idx],
                embeddings=embeddings,
                metadatas=chunked_metadatas[idx],
                ids=chunked_ids[idx],
            )
            p.update(t, advance=1)


if __name__ == "__main__":
    main()
