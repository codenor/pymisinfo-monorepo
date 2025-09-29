#!/usr/bin/env python

import argparse
from typing import Generator, List

import chromadb
import numpy
from chromadb.api.types import Document
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from pandas import pandas
from rich.progress import Progress


class ProgramArgs:
    def __init__(
        self,
        test_file: str = "./assets/preprocessed-testing.csv",
        chroma_db_path: str = "/var/lib/chroma/",
        chromadb_collection_name: str = "misinformation",
        ollama_url: str = "localhost:11434",
        embedding_model: str = "embeddinggemma:latest",
        verbose_level: int = 0,
        max_tests: int = 0,
    ) -> None:
        self.test_file = test_file
        self.chroma_db_path = chroma_db_path
        self.chromadb_collection_name = chromadb_collection_name
        self.ollama_url = ollama_url
        self.ollama_embedding_model = embedding_model
        self.verbose_level = verbose_level
        self.max_tests = max_tests


class CsvRecord:
    def __init__(self, content: str, misinfo_classification: str):
        self.content = content
        self.misinfo_classification = misinfo_classification


class Result:
    def __init__(
        self,
        input: CsvRecord,
        metadata: List[List[chromadb.Metadata]] | None,
        documents: List[List[Document]] | None,
        distances: List[List[float]] | None,
    ):
        self.input = input
        self.metadata = metadata
        self.documents = documents
        self.distances = distances


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
    parser.add_argument(
        "-o",
        "--olama-url",
        dest="ollama_url",
        help="URL to Ollama instance. Default='localhost:11434'",
        default="localhost:11434",
    )
    parser.add_argument(
        "-e",
        "--olama-embedding-model",
        dest="ollama_embedding_model",
        help="The model to use to create embeddings for queries. Should be the same as what you used when importing into chromadb. Default='embeddinggemma:latest'",
        default="embeddinggemma:latest",
    )
    parser.add_argument(
        "-v",
        "--verbose-level",
        dest="verbose_level",
        help="0=nothing (default), 1=show incorrect matches, 2=show incorrect matches and distances, 3=show everything",
        default=0,
    )
    parser.add_argument(
        "-m",
        "--max-tests",
        dest="max_tests",
        help="amount of tests to do in total (used for debugging). 0 will process all records. Default=0",
        default=0,
    )

    arguments = parser.parse_args()

    return ProgramArgs(
        str(arguments.input),
        str(arguments.chromadb_path),
        str(arguments.chromadb_collection_name),
        str(arguments.ollama_url),
        str(arguments.ollama_embedding_model),
        int(arguments.verbose_level),
        int(arguments.max_tests),
    )


def get_records(csv_path: str) -> Generator[CsvRecord]:
    csv_data = pandas.read_csv(
        csv_path,
        header=1,
        engine="c",
        on_bad_lines="warn",
    )

    for record in csv_data.itertuples():
        yield CsvRecord(str(record[0]), str(record[5]))


def test_all_records(
    test_file_path: str,
    ollama_ef: OllamaEmbeddingFunction,
    chroma_collection: chromadb.Collection,
    max_records: int,
):
    idx = 0
    for record in get_records(test_file_path):
        prompt_embedding = ollama_ef.embed_query([record.content])
        result = chroma_collection.query(
            query_embeddings=prompt_embedding,
            n_results=10,
            include=["documents", "distances", "metadatas"],
        )

        result_metadata = result.get("metadatas")
        result_documents = result.get("documents")
        result_distances = result.get("distances")

        yield Result(record, result_metadata, result_documents, result_distances)

        idx += 1
        if idx >= max_records:
            break


def main():
    args = get_args()

    print("connecting to chromadb")
    client = chromadb.PersistentClient(path=args.chroma_db_path)
    collection = client.get_collection(args.chromadb_collection_name)
    if args.max_tests > 0:
        records_c = numpy.minimum(len(pandas.read_csv(args.test_file)), args.max_tests)
    else:
        records_c = len(pandas.read_csv(args.test_file))

    print(
        f"preparing ollama embedding function ({args.ollama_url} | {args.ollama_embedding_model})"
    )
    ollama_ef = OllamaEmbeddingFunction(
        url=args.ollama_url,
        model_name=args.ollama_embedding_model,
    )

    correct = 0
    incorrect = 0
    with Progress() as p:
        t = p.add_task(
            f"testing model: {records_c} records to be processed", total=records_c
        )
        for result in test_all_records(
            args.test_file, ollama_ef, collection, records_c
        ):
            if (
                not result.metadata
                or len(result.metadata) < 1
                or not result.documents
                or not result.distances
            ):
                print("no closest match found")
                incorrect += 1
            else:
                score = result.metadata[0][0].get("misinformation")
                if str(score) == result.input.misinfo_classification:
                    correct += 1
                else:
                    incorrect += 1
                    # for i in range(0, len(result.items()) - 1):
                    #     print(f"\tIs: {result_metadata[0][i]};\n\t -> Distances: {result_distances[0][i]};\n\t -> Title: {result_documents[0][i]};")
            p.advance(t)

    print(f"correct: {correct}, incorrect: {incorrect}")
    print(f"accuracy: {100 * (correct / records_c)}%")


if __name__ == "__main__":
    main()
