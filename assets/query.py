#!/usr/bin/env python

import sys

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

ollama_ef = OllamaEmbeddingFunction(
    url="http://localhost:11434",
    model_name="embeddinggemma:latest",
)

if len(sys.argv) != 2:
    print(f'invalid command. usage: {sys.argv[0]} "prompt"')
    exit()

prompt = sys.argv[1]
prompt_embedding = ollama_ef.embed_query([prompt])

client = chromadb.PersistentClient(path="/var/lib/chroma/")
collection = client.get_or_create_collection("misinformation")
result = collection.query(
    query_embeddings=prompt_embedding,
    n_results=1,
    include=[
        "documents",
        "distances",
        "metadatas"
    ],
)

result_metadata = result.get("metadatas")
if not result_metadata or len(result_metadata) < 1:
    print("no closest match found")
    exit()

print(result_metadata[0][0].get("misinformation"))
