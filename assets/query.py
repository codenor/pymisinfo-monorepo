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
    n_results=10,
    include=[
        "documents",
        "distances",
        "metadatas"
    ],
)

result_metadata = result.get("metadatas")
result_documents = result.get("documents")
result_distances = result.get("distances")
if not result_metadata or len(result_metadata) < 1 or not result_documents or not result_distances:
    print("no closest match found")
    exit()

score = result_metadata[0][0].get("misinformation")
print(f"This post is considered: {score}")
print(f"Closest Results:")

for i in range(0, len(result.items()) - 1):
    print(f"\tIs: {result_metadata[0][i]};\n\t -> Distances: {result_distances[0][i]};\n\t -> Title: {result_documents[0][i]};")
