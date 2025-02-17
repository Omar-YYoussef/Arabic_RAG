# utils/retriever.py
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from models.reranker_model import get_reranker_model

def create_compression_retriever(vector_store):
    """Create a compression retriever."""
    reranker_model = get_reranker_model()
    compressor = CrossEncoderReranker(model=reranker_model, top_n=3)
    retriever = vector_store.as_retriever()
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
    return compression_retriever