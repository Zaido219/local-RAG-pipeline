from rag_engine.services.document_processor import DocumentProcessor
from rag_engine.services.vector_repository import ChromaVectorRepository
from rag_engine.clients.ollama_client import OllamaEmbeddingModel, OllamaInferenceClient
from rag_engine.services.retriever import RetrieverService
from rag_engine.services.prompt_builder import PromptBuilderService
from rag_engine.services.pipeline import RAGPipeline

def main():
    processor = DocumentProcessor()
    vector_repo = ChromaVectorRepository(
        persist_directory="./chroma_db", 
        collection_name="constitution_docs"
    )
    
    embedder = OllamaEmbeddingModel(model_name="nomic-embed-text")
    retriever = RetrieverService(vector_repo=vector_repo, embedding_model=embedder)
    prompt_builder = PromptBuilderService()
    inference_client = OllamaInferenceClient(model_name="qwen2.5:7b")

    pipeline = RAGPipeline(
        document_processor=processor,
        vector_repo=vector_repo,
        embedding_model=embedder,
        retriever=retriever,
        prompt_builder=prompt_builder,
        inference_client=inference_client
    )

    print("Ingesting Alice in Wonderland Chapter 1...")
    chunks_created = pipeline.ingest_document("sample.txt")
    print(f"Stored {chunks_created} chunks into ChromaDB.\n")

    # Sample query
    query_text = "Describe article 3"
    print(f"User Query: {query_text}")
    print("-" * 50)
    
    answer = pipeline.query(query_text, top_k=2)
    print("\n--- LLM ANSWER ---")
    print(answer)

if __name__ == "__main__":
    main()