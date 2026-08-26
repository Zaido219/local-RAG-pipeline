import os
import sys
from rag_engine.services.document_processor import DocumentProcessor
from rag_engine.services.vector_repository import ChromaVectorRepository
from rag_engine.clients.ollama_client import OllamaEmbeddingModel, GeminiInferenceClient
from rag_engine.services.retriever import RetrieverService
from rag_engine.services.prompt_builder import PromptBuilderService
from rag_engine.services.pipeline import RAGPipeline


def main():
    chroma_path = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    document_path = os.getenv("DOCUMENT_PATH", "data_privacy_act.txt")

    # Guardrail: Verify target file exists
    if not os.path.exists(document_path):
        print(f"[Error] Target document file not found at: '{document_path}'")
        sys.exit(1)

    print("Initializing Ingestion Dependencies...")
    processor = DocumentProcessor(chunk_size=1000, overlap=150)
    vector_repo = ChromaVectorRepository(
        persist_directory=chroma_path, 
        collection_name="constitution_docs"
    )
    embedder = OllamaEmbeddingModel(model_name="nomic-embed-text", host=ollama_url)
    
    # Query services (instantiated to fulfill RAGPipeline interface)
    retriever = RetrieverService(vector_repo=vector_repo, embedding_model=embedder)
    prompt_builder = PromptBuilderService()
    inference_client = GeminiInferenceClient()

    # Construct Orchestrator Pipeline
    pipeline = RAGPipeline(
        document_processor=processor,
        vector_repo=vector_repo,
        embedding_model=embedder,
        retriever=retriever,
        prompt_builder=prompt_builder,
        inference_client=inference_client
    )

    print(f"Ingesting document: '{document_path}'...")
    try:
        chunks_created = pipeline.ingest_document(document_path)
        print(f"Successfully processed and stored {chunks_created} chunks into ChromaDB.")
    except Exception as e:
        print(f"[Error] Failed to ingest document: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()