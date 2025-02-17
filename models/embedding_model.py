# models/embedding_model.py
from langchain.embeddings import HuggingFaceEmbeddings
from config import EMB_MODEL_PATH

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMB_MODEL_PATH)