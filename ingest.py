# ingest.py
# Loads a PDF, splits into chunks, embeds using HuggingFace,
# and stores vectors in pgvector PostgreSQL database.

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def ingest_pdf(pdf_path: str):
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Storing in pgvector...")
    vectorstore = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection_string=DATABASE_URL,
        collection_name="study_docs"
    )
    print("Done! Documents stored in database.")
    return vectorstore

if __name__ == "__main__":
    ingest_pdf("sample.pdf")