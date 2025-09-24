#!/usr/bin/env python

import chromadb
import sys

if len(sys.argv) != 2:
    print(f"invalid command. usage: {sys.argv[0]} \"prompt\"")
    exit()

prompt = sys.argv[1]

client = chromadb.PersistentClient(path="/var/lib/chroma/")
collection = client.get_or_create_collection("misinformation")
result = collection.query(query_texts=[prompt], n_results=5, include=["documents", 'distances',])

print(result)
