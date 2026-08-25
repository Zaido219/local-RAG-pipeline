import os
from rag_engine.services.document_processor import DocumentProcessor
from rag_engine.services.vector_repository import ChromaVectorRepository
from rag_engine.clients.ollama_client import OllamaEmbeddingModel
from rag_engine.services.pipeline import RAGPipeline

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

    pipeline = RAGPipeline(
        document_processor=processor,
        vector_repo=vector_repo,
        embedding_model=embedder,
    )

    document_path = "sample.txt"
    print(f"Ingesting document: {document_path}...")
    
    chunks_created = pipeline.ingest_document(document_path)
    print(f"Successfully stored {chunks_created} chunks into ChromaDB.")

if __name__ == "__main__":
    main()