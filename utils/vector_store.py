# utils/vector_store.py
from langchain.vectorstores import Chroma
from models.embedding_model import get_embedding_model
from config import CHROMA_PERSIST_DIR
import os

from utils.retriever import create_compression_retriever


def create_or_load_vector_store(chunks):
    """Create or load Chroma vector store."""
    embedding_model = get_embedding_model()
    vector_store = Chroma.from_texts(chunks, embedding_model, persist_directory=CHROMA_PERSIST_DIR)
    return vector_store

def check_chromadb_status():
    """Check if ChromaDB exists and contains data."""
    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        embedding_model = get_embedding_model()
        vector_store = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embedding_model)
        return create_compression_retriever(vector_store)
    return None