from rag_engine.core.interfaces import BaseEmbeddingModel, BaseInferenceClient
import ollama
import os


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


class OllamaInferenceClient(BaseInferenceClient):
    def __init__(self, model_name:str="qwen2.5:7b", host:str="http://localhost:11434"):
        self.model_name = model_name
        self.client = ollama.Client(host=host)

    def generate(self, prompt:str) -> str:
        response = self.client.generate(
            model=self.model_name, 
            prompt=prompt,
            options={
                "num_ctx":2048,
                "keep_alive": 0
                })
        return response["response"]


class GeminiInferenceClient(BaseInferenceClient):
    def __init__(self, model_name:str="gemini-2.5-flash", key:str | None = None):
        self.model_name = model_name
        api_key = key or os.getenv("API_KEY")

        if not api_key:
            raise ValueError("Missing api key")
        # initialize the google genai client
        self.client = genai.Client(api_key=key)

    def generate(self, prompt:str):
        response = self.client.models.generate_content(
            model = self.model_name,
            contents = prompt,
        )

        return response.text