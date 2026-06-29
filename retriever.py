# retriever.py
# Connects to pgvector, takes a user query, embeds it,
# and retrieves the most relevant chunks from the database.

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = PGVector(
        connection_string=DATABASE_URL,
        embedding_function=embeddings,
        collection_name="study_docs"
    )

    # Returns top 4 most relevant chunks for any query
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def retrieve(query: str):
    retriever = get_retriever()
    results = retriever.invoke(query)

    print(f"\nQuery: {query}")
    print(f"Found {len(results)} relevant chunks:\n")
    for i, doc in enumerate(results):
        print(f"--- Chunk {i+1} ---")
        print(doc.page_content[:300])
        print()

if __name__ == "__main__":
    retrieve("What is the main topic of this document?")