from abc import ABC, abstractmethod
from rag_engine.core.models import TextChunk, RetrievalResult

class VectoryDbBaseClass(ABC):
    @abstractmethod
    def add_chunks(self,chunks:list[TextChunk]) -> None:
        pass
    
    @abstractmethod
    def similarity_search(self, query_embedding:list[float], top_k:int = 5) -> list[RetrievalResult]:
        pass


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def embed_query(self, text:str) -> list[float]:
        pass
    @abstractmethod
    def embed_documents(self, texts:list[str]) -> list[list[float]]:
        pass


class BaseInferenceClient(ABC):
    @abstractmethod
    def generate(self,prompt:str) -> str:
        pass