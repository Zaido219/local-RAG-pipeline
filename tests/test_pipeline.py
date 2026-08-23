from unittest.mock import MagicMock
from rag_engine.services.pipeline import RAGPipeline
from rag_engine.core.models import Document, TextChunk, RetrievalResult

def test_ingest_document():
    # 1. ARRANGE
    mock_processor = MagicMock()
    mock_vector_repo = MagicMock()
    mock_embedder = MagicMock()
    mock_retriever = MagicMock()
    mock_builder = MagicMock()
    mock_inference = MagicMock()

    # Setup mock returns for ingestion
    doc = Document(id="d1", content="sample.txt")
    chunks = [
        TextChunk(chunk_id="c1", document_id="d1", text="Chunk 1 text"),
        TextChunk(chunk_id="c2", document_id="d1", text="Chunk 2 text")
    ]
    mock_processor.process_document.return_value = (doc, chunks)
    mock_embedder.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]

    pipeline = RAGPipeline(
        document_processor=mock_processor,
        vector_repo=mock_vector_repo,
        embedding_model=mock_embedder,
        retriever=mock_retriever,
        prompt_builder=mock_builder,
        inference_client=mock_inference
    )

    # 2. ACT
    chunk_count = pipeline.ingest_document("sample.txt")

    # 3. ASSERT
    mock_processor.process_document.assert_called_once_with("sample.txt")
    mock_embedder.embed_documents.assert_called_once_with(["Chunk 1 text", "Chunk 2 text"])
    mock_vector_repo.add_chunks.assert_called_once_with(chunks)
    assert chunk_count == 2
    assert chunks[0].embedding == [0.1, 0.2]


def test_query_pipeline():
    # 1. ARRANGE
    mock_processor = MagicMock()
    mock_vector_repo = MagicMock()
    mock_embedder = MagicMock()
    mock_retriever = MagicMock()
    mock_builder = MagicMock()
    mock_inference = MagicMock()

    # Setup mock returns for query execution
    mock_results = [
        RetrievalResult(
            chunk=TextChunk(chunk_id="c1", document_id="d1", text="Retrieved context"), 
            score=0.1
        )
    ]
    mock_retriever.retrieve.return_value = mock_results
    mock_builder.build_prompt.return_value = "Formatted prompt context"
    mock_inference.generate.return_value = "Generated answer from LLM"

    pipeline = RAGPipeline(
        document_processor=mock_processor,
        vector_repo=mock_vector_repo,
        embedding_model=mock_embedder,
        retriever=mock_retriever,
        prompt_builder=mock_builder,
        inference_client=mock_inference
    )

    # 2. ACT
    answer = pipeline.query("What is RAG?", top_k=2)

    # 3. ASSERT
    mock_retriever.retrieve.assert_called_once_with("What is RAG?", top_k=2)
    mock_builder.build_prompt.assert_called_once_with("What is RAG?", mock_results)
    mock_inference.generate.assert_called_once_with("Formatted prompt context")
    assert answer == "Generated answer from LLM"