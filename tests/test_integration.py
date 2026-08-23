from rag_engine.services.document_processor import DocumentProcessor
from rag_engine.clients.ollama_client import OllamaEmbeddingModel
from rag_engine.services.vector_repository import ChromaVectorRepository
from rag_engine.services.retriever import RetrieverService

def test_full_rag_integration(tmp_path):
    # 1. Initialize components (using tmp_path to keep tests clean)
    processor = DocumentProcessor()
    embedder = OllamaEmbeddingModel()
    
    # Use Pytest's tmp_path fixture so Chroma doesn't write persistent state to your real DB directory during tests
    vector_repo = ChromaVectorRepository(
        persist_directory=str(tmp_path / "chroma_db"), 
        collection_name="test_rag_docs"
    )

    # 2. Process & Chunk Document
    doc, chunks = processor.process_document("tests/sample.txt")

    # 3. Generate Embeddings for Chunks
    chunk_texts = [c.text for c in chunks]
    embeddings = embedder.embed_documents(chunk_texts)

    for chunk, emb in zip(chunks, embeddings):
        chunk.embedding = emb

    # 4. Save to Vector Store
    vector_repo.add_chunks(chunks)

    # 5. Query via RetrieverService
    retriever = RetrieverService(vector_repo=vector_repo, embedding_model=embedder)
    results = retriever.retrieve("What is this document about?", top_k=2)

    # 6. Assertions (This is what makes it a test!)
    assert len(results) > 0
    assert results[0].chunk.text is not None