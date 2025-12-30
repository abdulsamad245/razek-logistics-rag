"""
Streamlit Application for RAG System

This application provides a web interface for querying the RAG system
about Razek's internal documents. It includes chat functionality,
sample questions, and automatic ingestion pipeline execution.
"""

import os
from typing import List

from dotenv import load_dotenv
from ingestion_pipeline import PERSIST_DIR, run_ingestion_pipeline
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import streamlit as st

load_dotenv()

# Constants
PAGE_TITLE = "Razek Logistics RAG System"
LOGO_PATH = "assets/razek_logo.png"
PAGE_ICON_WIDTH = 60
CHAT_PLACEHOLDER = "Ask a question about Razek logistics..."
SAMPLE_QUESTIONS = [
    "Why should I use Razek?",
    "What is Razek's logistics procurement process?",
    "How does Razek manage supply chain operations?",
    "What documents are stored in the Logistics Document Manager?",
    "How are shipments approved and delivered through Razek?",
    "What happens to shipments at end of lifecycle?"
]
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
LLM_MODEL_NAME = "gpt-4o"
RETRIEVER_K = 7
HNSW_SPACE = "cosine"


# Streamlit UI Configuration
st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.image(LOGO_PATH, width=PAGE_ICON_WIDTH)
st.title(PAGE_TITLE)
st.write("Ask any question based on Razek's logistics operations and policies.")

# --- Auto-run ingestion pipeline if vector store is missing ---
if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
    st.info("No vector store found. Running ingestion pipeline...")
    run_ingestion_pipeline()
    st.success("Ingestion completed!")

# --- Load vector store ---
embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": HNSW_SPACE},
)

retriever = db.as_retriever(search_kwargs={"k": RETRIEVER_K})

# --- Session State (Chat Memory) ---

if "messages" not in st.session_state:
    st.session_state.messages = []

if "prefill_input" not in st.session_state:
    st.session_state.prefill_input = ""

# --- Sidebar ---
st.sidebar.image(LOGO_PATH, width=PAGE_ICON_WIDTH)
st.sidebar.title("Sample Questions")

selected_question = st.sidebar.radio(
    "Try one of these:",
    SAMPLE_QUESTIONS
)

# --- Add a button to fill the input with a sample question without auto-sending ---
if st.sidebar.button("Use this sample question"):
    st.session_state.prefill_input = selected_question

# --- Display previous chat messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# --- Chat Input ---
user_query = st.chat_input(
    CHAT_PLACEHOLDER,
    key="user_input"
)

# If user_query is empty, use the prefilled input from the sample question
if not user_query and st.session_state.prefill_input:
    user_query = st.session_state.prefill_input
    # Clear the prefill so it doesn't keep triggering
    st.session_state.prefill_input = ""

# --- Run query only when user submits ---
if user_query:
    # --- Store user message ---
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )
    with st.chat_message("user"):
        st.write(user_query)

# --- Retrieve documents ---
    relevant_docs = retriever.invoke(user_query)

    # --- Build LLM input ---
    combined_input = f"""You are a helpful assistant supporting Razek logistics operations.

### RULES ABOUT KNOWLEDGE USE
1. **If the user's question is about Razek logistics, supply chain processes, policies, operations, or anything internal**,
   ONLY use the Razek documents provided below.

2. **If the documents do not contain the answer, respond politely without guessing. Use a friendly message.**


3. **If the user's question is general :**
   You may answer using your own general knowledge.

4. **Never invent or guess details about Razek logistics.**

5. Your tone should be professional, friendly, and conversational.

6. Even if the user phrases the question differently from the document wording, try to understand the meaning and find relevant content.

### USER QUESTION:
{user_query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Guidelines:
- First determine whether the question is Razek logistics-related or general.
# - If Razek logistics-related → use ONLY the documents.
- If not Razek logistics-related → answer normally.
- If documents lack the answer → give the fallback message above.
"""

    # Run model
    model = ChatOpenAI(model=LLM_MODEL_NAME)
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=combined_input),
    ]

    with st.spinner("Generating answer..."):
        response = model.invoke(messages)
    
    with st.chat_message("assistant"):
        st.write(response.content)

    # store assistant response in session state for chat history
    st.session_state.messages.append({"role": "assistant", "content": response.content})
