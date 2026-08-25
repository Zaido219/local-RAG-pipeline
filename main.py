import os
from rag_engine.services.document_processor import DocumentProcessor
from rag_engine.services.vector_repository import ChromaVectorRepository
from rag_engine.clients.ollama_client import OllamaEmbeddingModel
from rag_engine.services.pipeline import RAGPipeline
from rag_engine.services.retriever import RetrieverService
from rag_engine.services.prompt_builder import PromptBuilderService
from rag_engine.clients.ollama_client import GeminiInferenceClient, OllamaEmbeddingModel

def main():
    chroma_path = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    print("Initializing Ingestion Pipeline...")
    processor = DocumentProcessor()
    vector_repo = ChromaVectorRepository(
        persist_directory=chroma_path, 
        collection_name="constitution_docs"
    )
    embedder = OllamaEmbeddingModel(model_name="nomic-embed-text", host=ollama_url)
    retriever = RetrieverService(vector_repo=vector_repo, embedding_model=embedder)
    prompt_builder = PromptBuilderService()
    inference_client = GeminiInferenceClient()

    pipeline = RAGPipeline(
        document_processor=processor,
        vector_repo=vector_repo,
        embedding_model=embedder,
        retriever=retriever,
        prompt_builder=prompt_builder,
        inference_client=inference_client
    )

    document_path = "sample.txt"
    print(f"Ingesting document: {document_path}...")
    
    chunks_created = pipeline.ingest_document(document_path)
    print(f"Successfully stored {chunks_created} chunks into ChromaDB.")

if __name__ == "__main__":
    main()