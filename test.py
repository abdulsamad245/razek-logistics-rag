"""
Test Script for Vector Database

This script tests the ChromaDB vector store by retrieving and displaying
stored documents and their metadata.
"""

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Configuration
PERSIST_DIR = "db/chroma_db"
EMBEDDING_MODEL = OpenAIEmbeddings(model="text-embedding-3-small")
PREVIEW_DOCS = 5
PREVIEW_LENGTH = 300

# Initialize database
db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=EMBEDDING_MODEL
)

# Retrieve all stored documents
docs = db.get()

print(f"Number of embeddings: {len(docs['ids'])}")

# Display first few documents
for i, doc in enumerate(docs["documents"][:PREVIEW_DOCS]):
    print(f"\nDocument {i+1}:")
    print(f"  Content preview: {doc[:PREVIEW_LENGTH]}...")
    if len(doc) > PREVIEW_LENGTH:
        print("  (truncated)")
