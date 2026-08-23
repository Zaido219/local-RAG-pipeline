import pytest
from rag_engine.services.document_processor import DocumentProcessor
from rag_engine.core.models import Document, TextChunk

def test_process_document(tmp_path):
    # Setup temporary text file
    test_file = tmp_path / "sample.txt"
    test_file.write_text("Hello world! This is a test document for RAG chunking.")

    processor = DocumentProcessor()
    doc, chunks = processor.process_document(
        file_path=str(test_file), 
        chunk_size=20, 
        overlap=5
    )

    # Assertions
    assert isinstance(doc, Document)
    assert len(chunks) > 0
    assert isinstance(chunks[0], TextChunk)
    assert chunks[0].document_id == doc.id