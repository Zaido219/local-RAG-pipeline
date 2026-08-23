from unittest.mock import MagicMock
from rag_engine.services.retriever import RetrieverService
from rag_engine.core.models import RetrievalResult, TextChunk

def test_retriever_service():
    # 1. ARRANGE: Create mock dependencies
    mock_vector_repo = MagicMock()
    mock_embedding_model = MagicMock()

    # Define dummy returns for the mocks
    mock_embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]
    
    expected_chunk = TextChunk(
        chunk_id="c1", document_id="d1", text="RAG architecture test"
    )
    expected_result = [RetrievalResult(chunk=expected_chunk, score=0.05)]
    mock_vector_repo.similarity_search.return_value = expected_result

    retriever = RetrieverService(
        vector_repo=mock_vector_repo, 
        embedding_model=mock_embedding_model
    )

    results = retriever.retrieve(query_text="What is RAG?", top_k=3)

    # 3. ASSERT
    # Check that embed_query was called with correct string
    mock_embedding_model.embed_query.assert_called_once_with("What is RAG?")
    
    # Check that similarity_search was called with the mock's generated embedding
    mock_vector_repo.similarity_search.assert_called_once_with([0.1, 0.2, 0.3], top_k=3)
    
    # Check output matches expected domain model
    assert results == expected_result