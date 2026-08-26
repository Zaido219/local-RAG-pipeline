from typing import List
from rag_engine.core.interfaces import VectoryDbBaseClass, BaseEmbeddingModel
from rag_engine.core.models import RetrievalResult


class RetrieverService:
    """Service responsible for converting natural language queries into embeddings and retrieving top matches."""

    def __init__(self, vector_repo: VectoryDbBaseClass, embedding_model: BaseEmbeddingModel):
        self.vector_repo = vector_repo
        self.embedding_model = embedding_model

    def retrieve(self, query_text: str, top_k: int = 5) -> List[RetrievalResult]:
        """Embeds the search query and delegates vector search to the vector repository."""
        query_embedding = self.embedding_model.embed_query(query_text)
        return self.vector_repo.similarity_search(query_embedding, top_k=top_k)