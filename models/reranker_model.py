# models/reranker_model.py
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from config import RERANKER_MODEL_PATH

def get_reranker_model():
    return HuggingFaceCrossEncoder(model_name=RERANKER_MODEL_PATH)