# utils/text_processing.py
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text):
    """Split text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    return chunks