# advanced_retriever.py
# Implements sentence-window and auto-merging retrieval
# plus a query router that picks the right strategy per question.

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def get_vectorstore():
    return PGVector(
        connection_string=DATABASE_URL,
        embedding_function=get_embeddings(),
        collection_name="study_docs"
    )

# Strategy 1: Sentence-window retrieval
# Fetches the matching chunk plus its neighbors for more context
def sentence_window_retriever(query: str, k: int = 4, window: int = 1):
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search(query, k=k)

    print(f"[Sentence-Window] Found {len(results)} chunks, expanding context window...")

    expanded = []
    for doc in results:
        page = doc.metadata.get("page", 0)
        # Pull neighboring chunks from same page for context
        neighbors = vectorstore.similarity_search(
            query,
            k=k + 4,
            filter={"page": page}
        )
        combined_text = " ".join([n.page_content for n in neighbors[:window * 2 + 1]])
        expanded.append(Document(
            page_content=combined_text,
            metadata=doc.metadata
        ))

    return expanded


# Strategy 2: Auto-merging retrieval
# If multiple chunks from the same page are retrieved, merge them into one
def auto_merging_retriever(query: str, k: int = 6, merge_threshold: int = 2):
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search(query, k=k)

    print(f"[Auto-Merging] Found {len(results)} chunks, merging by page...")

    # Group chunks by page
    page_groups = {}
    for doc in results:
        page = doc.metadata.get("page", 0)
        if page not in page_groups:
            page_groups[page] = []
        page_groups[page].append(doc.page_content)

    merged = []
    for page, chunks in page_groups.items():
        if len(chunks) >= merge_threshold:
            # Merge chunks from same page into one document
            merged.append(Document(
                page_content=" ".join(chunks),
                metadata={"page": page, "merged": True, "chunk_count": len(chunks)}
            ))
        else:
            merged.append(Document(
                page_content=chunks[0],
                metadata={"page": page, "merged": False}
            ))

    print(f"[Auto-Merging] Merged into {len(merged)} documents")
    return merged


# Query Router: decides which strategy to use based on question type
def query_router(query: str) -> str:
    query_lower = query.lower()

    # Factual/specific questions → sentence window (need precise context)
    factual_keywords = ["what is", "define", "explain", "how does", "what are"]

    # Broad/summary questions → auto merging (need wider coverage)
    broad_keywords = ["summarize", "overview", "list all", "what topics", "everything about"]

    for kw in broad_keywords:
        if kw in query_lower:
            return "auto_merging"

    for kw in factual_keywords:
        if kw in query_lower:
            return "sentence_window"

    # Default to sentence window
    return "sentence_window"


def advanced_retrieve(query: str):
    strategy = query_router(query)
    print(f"\nQuery: {query}")
    print(f"Router selected: {strategy}\n")

    if strategy == "sentence_window":
        docs = sentence_window_retriever(query)
    else:
        docs = auto_merging_retriever(query)

    for i, doc in enumerate(docs):
        print(f"--- Result {i+1} (page {doc.metadata.get('page', '?')}) ---")
        print(doc.page_content[:300])
        print()

    return docs


if __name__ == "__main__":
    advanced_retrieve("what is an activation function")
    print("\n" + "="*60 + "\n")
    advanced_retrieve("summarize everything in module 6")