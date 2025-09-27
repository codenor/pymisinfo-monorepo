#!/usr/bin/env python

import csv

import chromadb
import argparse
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from rich.progress import Progress


class ProgramArgs:
    def __init__(self, 
         preprocessedCsvPath: str = "./preprocessed.csv", 
         ollamaUrl: str = "http://localhost:11434", 
         ollamaModelName: str = "embeddinggemma:latest", 
         chromadbPath: str = "/var/lib/chroma/"
    ) -> None:
        self.preprocessedCsvPath = preprocessedCsvPath
        self.ollamaUrl = ollamaUrl
        self.ollamaModelName = ollamaModelName
        self.chromadbPath = chromadbPath


def get_args() -> ProgramArgs:
    """Parses the command line arguments. Will call sys.exit() if the -h (help) flag is passed"""
    parser = argparse.ArgumentParser(
        prog="Misinformation Dataset Importer",
        description="Imports data from the preprocessor into a Chroma database for analysis and machine learning",
        epilog="This program depends on ollama, so please have that installed. Install guide can be found here: https://docs.ollama.com/quickstart"
    )

    parser.add_argument("-i", "--input", dest="input", help="The input CSV file from the preprocessor program", default="./preprocessed.csv")
    parser.add_argument("-o", "--olama-url", dest="ollama_url", help="The URL to your running ollama instance", default="http://localhost:11434")
    parser.add_argument("-m", "--olama-model-name", dest="ollama_model_name", help="The model name for your ollama instance for embedding generation", default="embeddinggemma:latest")
    parser.add_argument("-c", "--chromadb-path", dest="chromadb_path", help="Path to the ChromaDB database", default="/var/lib/chroma/")
    # parser.add_argument(flags=["-h", "--help"], help="Show this screen", action='store_true')

    arguments = parser.parse_args()

    return ProgramArgs(str(arguments.input), str(arguments.ollama_url), str(arguments.ollama_model_name), str(arguments.chromadb_path))


def chunks(lst: list, n: int) -> list:
    """
    Yield successive n-sized chunks from lst.
    https://stackoverflow.com/a/312464
    """
    new_lst = []
    for i in range(0, len(lst), n):  # Slice list in steps of n
        new_lst.append(lst[i : i + n])
    return new_lst


def data_from_preprocessed_csv(file: str) -> tuple[list, list]:

    documents = []
    ids = []
    records_c = 0

    with open("./preprocessed.csv", "r") as csvfile:
        records_c = len(csvfile.readlines()) - 1  # to get rid of header

    with open(file, "r") as csvfile:
        reader = csv.reader(csvfile)
        idx = 0
        with Progress() as p:
            t = p.add_task("generating documents to insert", total=records_c)
            for row in reader:
                inputs = str(row[0])
                documents.append(inputs)
                ids.append(str(idx))
                p.update(t, advance=1)
                idx += 1
    return documents, ids


def main():
    # Init chromadb
    args = get_args()
    print(vars(args))
    return
    print("initialising and connecting to chromadb")
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434",
        model_name="embeddinggemma:latest",
    )

    client = chromadb.PersistentClient(path="/var/lib/chroma/")

    collection_name = "misinformation"
    client.delete_collection(collection_name)
    collection = client.create_collection(collection_name)

    print("collecting data from ./preprocessed.csv")
    documents, ids = data_from_preprocessed_csv("./preprocessed.csv")

    # Import into chromadb
    chunk_size = 5000
    chunked_documents = chunks(documents, chunk_size)
    chunked_ids = chunks(ids, chunk_size)
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
                ids=chunked_ids[idx],
            )
            p.update(t, advance=1)


if __name__ == "__main__":
    main()
