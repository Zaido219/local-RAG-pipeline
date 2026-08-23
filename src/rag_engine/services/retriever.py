from rag_engine.services.vector_repository import VectoryDbBaseClass
from rag_engine.clients.ollama_client import BaseEmbeddingModel
from rag_engine.core.models import RetrievalResult

class RetrieverService:
    def  __init__(self, vector_repo:VectoryDbBaseClass, embedding_model: BaseEmbeddingModel):
        self.vector_repo = vector_repo
        self.embedding_model = embedding_model

    def retrieve(self, query_text:str, top_k:int=5) -> list[RetrievalResult]:
        query_embedding = self.embedding_model.embed_query(query_text)
        return self.vector_repo.similarity_search(query_embedding, top_k=top_k)