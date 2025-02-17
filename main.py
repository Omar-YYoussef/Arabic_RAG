# main.py
import google.generativeai as genai
from utils.ocr_utils import perform_ocr
from utils.text_processing import chunk_text
from utils.vector_store import create_or_load_vector_store, check_chromadb_status
from utils.retriever import create_compression_retriever
from config import POPPLER_PATH, GENAI_API_KEY
from pdf2image import convert_from_path
import tempfile
import os

genai.configure(api_key=GENAI_API_KEY)

def process_image_and_create_embeddings(image_path, prompt):
    """Process image and create embeddings."""
    model = genai.GenerativeModel('gemini-2.0-flash')
    extracted_text = perform_ocr(model, image_path, prompt)
    chunks = chunk_text(extracted_text)
    vector_store = create_or_load_vector_store(chunks)
    return create_compression_retriever(vector_store)

def process_pdf_and_create_embeddings(pdf_path, prompt):
    """Process PDF and create embeddings."""
    model = genai.GenerativeModel('gemini-2.0-flash')
    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    extracted_text = ""
    for i, image in enumerate(images):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_image:
            image.save(temp_image.name, 'PNG')
            temp_image_path = temp_image.name
        extracted_text += perform_ocr(model, temp_image_path, prompt) + "\n"
        os.unlink(temp_image_path)
    chunks = chunk_text(extracted_text)
    vector_store = create_or_load_vector_store(chunks)
    return create_compression_retriever(vector_store)

def generate_response(query, compression_retriever):
    """Generate response for a query using the compression retriever."""
    retrieved_docs = compression_retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    model = genai.GenerativeModel('gemini-2.0-flash-lite-preview-02-05')
    response = model.generate_content(f"Context: {context}\n\nQuestion: {query}")
    return response.text