#!/usr/bin/env python

import chromadb
import csv
from rich.progress import Progress


def chunks(lst: list, n: int) -> list:
    """
    Yield successive n-sized chunks from lst.
    https://stackoverflow.com/a/312464
    """
    new_lst = []
    for i in range(0, len(lst), n):  # Slice list in steps of n
        new_lst.append(lst[i:i + n])
    return new_lst


def data_from_preprocessed_csv(file: str) -> tuple[list, list]:
    documents = []
    ids = []
    records_c = 0

    with open("./preprocessed.csv", "r")  as csvfile:
        records_c = len(csvfile.readlines()) - 1 # to get rid of header

    with open(file, "r")  as csvfile:
        reader = csv.reader(csvfile)
        idx = 0
        with Progress() as p:
            t = p.add_task("generating embeddings", total=records_c)
            for row in reader:
                inputs = str(row[0])
                documents.append(inputs)
                ids.append(str(idx))
                p.update(t, advance=1)
                idx += 1
    return documents, ids


def main():
    # Init chromadb
    print("initialising and connecting to chromadb")
    client = chromadb.PersistentClient(path="/var/lib/chroma/")
    collection = client.get_or_create_collection("misinformation")

    print("collecting data from ./preprocessed.csv")
    documents, ids = data_from_preprocessed_csv("./preprocessed.csv")

    # Import into chromadb
    chunk_size = 500
    chunked_documents = chunks(documents, chunk_size)
    chunked_ids = chunks(ids, chunk_size)
    with Progress() as p:
        t = p.add_task(f"inserting into chromadb in batches of {chunk_size}", total=len(chunked_documents))
        for idx in range(0, len(chunked_documents)):
            collection.upsert(
                documents=chunked_documents[idx], 
                ids=chunked_ids[idx]
            )
            p.update(t, advance=1)


if __name__ == "__main__":
    main()
