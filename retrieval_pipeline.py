"""
Retrieval Pipeline Module

This module handles document retrieval and question answering using
the vectorized documents stored in ChromaDB and OpenAI's language models.
"""

from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

PERSISTENT_DIRECTORY = "db/chroma_db"
EMBEDDING_MODEL = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize the vector database
db = Chroma(
    persist_directory=PERSISTENT_DIRECTORY,
    embedding_function=EMBEDDING_MODEL,
    collection_metadata={"hnsw:space": "cosine"},
)


def retrieve_documents(query: str, k: int = 5) -> List:
    """
    Retrieve relevant documents for a given query.

    Args:
        query: The search query.
        k: Number of documents to retrieve.

    Returns:
        List of relevant documents.
    """
    retriever = db.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)


def generate_answer(query: str, relevant_docs: List) -> str:
    """
    Generate an answer based on the query and relevant documents.

    Args:
        query: The user's question.
        relevant_docs: List of relevant documents.

    Returns:
        Generated answer string.
    """
    combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, concise, and accurate answer using only the information from the documents above.
If the answer is not contained within the documents, respond with "I currently do not have that information, I can escalate the issue to my superior."."""

    model = ChatOpenAI(model="gpt-4o")
    messages = [
        SystemMessage(content="You are a Razek logistics internal knowledge assistant"),
        HumanMessage(content=combined_input),
    ]

    response = model.invoke(messages)
    return response.content


if __name__ == "__main__":
    query = "what is Razek's logistics procurement process?"

    print(f"User Query: {query}")
    print("--- Retrieving relevant documents ---")

    relevant_docs = retrieve_documents(query)

    print("--- Retrieved Documents ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")

    print("--- Generating Answer ---")
    answer = generate_answer(query, relevant_docs)
    print("Answer:")
    print(answer)
