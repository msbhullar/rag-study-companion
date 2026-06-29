# pipeline.py
# Wraps the RAG ingestion pipeline in Dagster assets.
# Gives us a visual UI, scheduling, and observability.

import os
from dagster import asset, Definitions, MaterializeResult, MetadataValue
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
PDF_PATH = "sample.pdf"


@asset(description="Loads raw pages from the PDF document")
def raw_documents():
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from {PDF_PATH}")
    return MaterializeResult(
        metadata={
            "page_count": MetadataValue.int(len(docs)),
            "source_file": MetadataValue.text(PDF_PATH),
        }
    )


@asset(
    deps=["raw_documents"],
    description="Splits documents into chunks using RecursiveCharacterTextSplitter"
)
def chunked_documents():
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    return MaterializeResult(
        metadata={
            "chunk_count": MetadataValue.int(len(chunks)),
            "chunk_size": MetadataValue.int(512),
            "chunk_overlap": MetadataValue.int(50),
        }
    )


@asset(
    deps=["chunked_documents"],
    description="Embeds chunks and stores them in pgvector"
)
def vector_embeddings():
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection_string=DATABASE_URL,
        collection_name="study_docs"
    )

    print(f"Stored {len(chunks)} chunks in pgvector")

    return MaterializeResult(
        metadata={
            "chunks_stored": MetadataValue.int(len(chunks)),
            "embedding_model": MetadataValue.text("all-MiniLM-L6-v2"),
            "collection": MetadataValue.text("study_docs"),
            "database": MetadataValue.text("pgvector"),
        }
    )


# Register all assets with Dagster
defs = Definitions(
    assets=[raw_documents, chunked_documents, vector_embeddings]
)