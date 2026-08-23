from abc import ABC, abstractmethod
from rag_engine.core.interfaces import BaseEmbeddingModel
import ollama


class OllamaEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name:str="nomic-embed-text", host:str="http://localhost:11434"):
        super().__init__()
        self.model_name = model_name
        self.client = ollama.Client(host=host)

    def embed_query(self, text:str) -> list[float]:
        response = self.client.embed(model=self.model_name, input=text)
        return response["embeddings"][0] #this will be a list of vectors we will only be returning the first one

    def embed_documents(self, texts:list[str]) -> list[list[float]]:
        # Take advantage of Ollama's native batch embedding support
        response = self.client.embed(model=self.model_name, input=texts)
        return response["embeddings"]

if __name__ == "__main__":
    embedder = OllamaEmbeddingModel()
    vector = embedder.embed_query("Hello world")

    print(f"Vector dimensions: {len(vector)}")