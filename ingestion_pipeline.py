"""
Ingestion Pipeline Module

This module handles the loading, splitting, and vectorization of documents
for the RAG system. It processes text files from a specified directory,
splits them into chunks, and stores them in a ChromaDB vector database.
"""

import os
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

DOCS_PATH = "docs"
PERSIST_DIR = "db/chroma_db"

def load_documents(docs_path: str = "docs") -> List:
    """
    Load all text files from the specified directory.

    Args:
        docs_path: Path to the directory containing text files.

    Returns:
        List of loaded documents.

    Raises:
        FileNotFoundError: If the directory doesn't exist or contains no .txt files.
    """
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"The directory '{docs_path}' does not exist. Please create it and add your text files."
        )

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
    )

    documents = loader.load()

    if not documents:
        raise FileNotFoundError(f"No .txt files found in the directory '{docs_path}'.")

    # Log first few documents for verification
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  Metadata: {doc.metadata}")

    print(f"Successfully loaded {len(documents)} documents.")
    return documents
        

def split_documents(documents: List, chunk_size: int = 1000, chunk_overlap: int = 100) -> List:
    """
    Split documents into smaller chunks with overlap.

    Args:
        documents: List of documents to split.
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Number of characters to overlap between chunks.

    Returns:
        List of document chunks.
    """
    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        # Log first few chunks for verification
        for i, chunk in enumerate(chunks[:7]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"  Source: {chunk.metadata['source']}")
            print(f"  Length: {len(chunk.page_content)} characters")
            print(f"  Content: {chunk.page_content[:200]}...")
            print("-" * 50)

        if len(chunks) > 7:
            print(f"\n... and {len(chunks) - 7} more chunks.")

    print(f"Successfully split into {len(chunks)} chunks.")
    return chunks


def create_vector_store(chunks: List, persist_directory: str = "db/chroma_db") -> Chroma:
    """
    Create and persist a ChromaDB vector store from document chunks.

    Args:
        chunks: List of document chunks to vectorize.
        persist_directory: Directory to persist the vector store.

    Returns:
        The created Chroma vector store.
    """
    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    print("--- Creating Chroma vector store ---")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print("--- Finished creating vector store ---")

    print(f"Vector store created and saved to '{persist_directory}'")
    return vector_store
def run_ingestion_pipeline() -> None:
    """
    Execute the full ingestion pipeline: load documents, split into chunks,
    and create the vector store.
    """
    print("🔄 Running ingestion pipeline...")
    try:
        documents = load_documents(DOCS_PATH)
        chunks = split_documents(documents)
        create_vector_store(chunks, PERSIST_DIR)
        print("✅ Ingestion pipeline completed successfully.")
    except Exception as e:
        print(f"❌ Ingestion pipeline failed: {e}")
        raise


       

if __name__ == "__main__":
    run_ingestion_pipeline()