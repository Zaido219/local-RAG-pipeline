import pytest
from rag_engine.core.models import TextChunk, RetrievalResult
from rag_engine.services.vector_repository import ChromaVectorRepository

def test_add_and_search_chunk(tmp_path):

    repo = ChromaVectorRepository(
        persist_directory=str(tmp_path),
        collection_name= "test_collection"
    ) 
    sample_chunk = TextChunk(
        chunk_id="chunk_1",
        document_id="doc_123",
        text="The quick brown fox jumps over the lazy dogs.",
        embedding=[0.1, 0.2, 0.3],
        metadata={"source": "unit_test"}
    )
    repo.add_chunks([sample_chunk])
    results = repo.similarity_search(query_embedding=[0.1, 0.2, 0.3, ], top_k=1)

    assert len(results) == 1
    assert isinstance(results[0], RetrievalResult)
    assert results[0].chunk.chunk_id == "chunk_1"
    assert results[0].chunk.text == "The quick brown fox jumps over the lazy dogs."
    assert results[0].chunk.document_id == "doc_123"