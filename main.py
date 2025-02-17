import google.generativeai as genai
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import os
import base64
from pdf2image import convert_from_path
from PIL import Image
import tempfile

# Configure API key
genai.configure(api_key='your_api_key')

# Initialize models and paths
emb_model_path = r"Omartificial-Intelligence-Space--GATE-AraBert-v1" #path to your Embedding

reranker_model_path = r"Omartificial-Intelligence-Space--ARA-Reranker-V1" #path to your Reranker

# Initialize models
reranker_model = HuggingFaceCrossEncoder(model_name=reranker_model_path)
embedding_model = HuggingFaceEmbeddings(model_name=emb_model_path)

CHROMA_PERSIST_DIR = "./chroma_db"

def perform_ocr(model, image_path, prompt):
    """Perform OCR on a single image."""
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()

    encoded_image = base64.b64encode(image_bytes).decode('utf-8')

    response = model.generate_content(
        [
            {'mime_type': 'image/png', 'data': encoded_image},
            prompt
        ]
    )

    return response.text

def chunk_text(text):
    """Split text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    return chunks

def create_or_load_vector_store(chunks):
    """Create or load Chroma vector store."""
    vector_store = Chroma.from_texts(chunks, embedding_model, persist_directory=CHROMA_PERSIST_DIR)
    return vector_store

def create_compression_retriever(vector_store):
    """Create a compression retriever."""
    compressor = CrossEncoderReranker(model=reranker_model, top_n=3)
    retriever = vector_store.as_retriever()
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
    return compression_retriever

def process_image_and_create_embeddings(image_path, prompt):
    """Process image and create embeddings."""
    model = genai.GenerativeModel('gemini-2.0-flash')
    # Extract text from image
    extracted_text = perform_ocr(model, image_path, prompt)

    # Create chunks
    chunks = chunk_text(extracted_text)

    # Create vector store
    vector_store = create_or_load_vector_store(chunks)

    # Create and return compression retriever
    return create_compression_retriever(vector_store)

def process_pdf_and_create_embeddings(pdf_path, prompt):
    """Process PDF and create embeddings."""
    model = genai.GenerativeModel('gemini-2.0-flash')

    poppler_path = r"Release-24.08.0-0\poppler-24.08.0\Library\bin" #path to your poppler

    # Convert PDF to images
    images = convert_from_path(pdf_path, poppler_path=poppler_path)

    extracted_text = ""
    for i, image in enumerate(images):
        # Save image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_image:
            image.save(temp_image.name, 'PNG')
            temp_image_path = temp_image.name

        # Extract text from image
        extracted_text += perform_ocr(model, temp_image_path, prompt) + "\n"

        # Clean up temporary image file
        os.unlink(temp_image_path)

    # Create chunks
    chunks = chunk_text(extracted_text)

    # Create vector store
    vector_store = create_or_load_vector_store(chunks)

    # Create and return compression retriever
    return create_compression_retriever(vector_store)


def check_chromadb_status():
    """Check if ChromaDB exists and contains data."""
    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        vector_store = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embedding_model)
        return create_compression_retriever(vector_store)
    return None



def generate_response(query, compression_retriever):
    """Generate response for a query using the compression retriever."""
    retrieved_docs = compression_retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    model = genai.GenerativeModel('gemini-2.0-flash-lite-preview-02-05')
    response = model.generate_content(f"Context: {context}\n\nQuestion: {query}")
    return response.text