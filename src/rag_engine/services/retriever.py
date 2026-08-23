from rag_engine.services.vector_repository import VectoryDbBaseClass
from rag_engine.clients.ollama_client import BaseEmbeddingModel

class RetrieverService:
    def  __init__(self, vector_repo:VectoryDbBaseClass, embedding_model: BaseEmbeddingModel):
        self.vector_repo = vector_repo
        self.embedding_model = embedding_model

    def retrieve(self, query_text:str, top_k:int):
        pass