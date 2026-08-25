import uuid
import re
from rag_engine.core.models import Document, TextChunk

class DocumentProcessor:
    def __init__(self):
        pass

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
        """Splits text recursively based on natural paragraph and sentence delimiters."""
        # Clean double newlines to split by logical sections first
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_len = len(para)
            
            # If adding this paragraph exceeds chunk size, dump current buffer
            if current_length + para_len > chunk_size and current_chunk:
                combined_text = "\n\n".join(current_chunk)
                chunks.append(combined_text)
                
                # Simple overlap: keep the last element for context continuity
                current_chunk = current_chunk[-1:] if overlap > 0 else []
                current_length = sum(len(p) for p in current_chunk)

            current_chunk.append(para)
            current_length += para_len

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def process_document(self, file_path: str, chunk_size: int = 1000, overlap: int = 150):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        doc_id = str(uuid.uuid4())
        document = Document(
            id=doc_id,
            content=content,
            metadata={"file_path": file_path}
        )

        raw_chunks = self._chunk_text(content, chunk_size=chunk_size, overlap=overlap)
        text_chunks = []

        for idx, chunk_text in enumerate(raw_chunks):
            chunk = TextChunk(
                chunk_id=f"{doc_id}_{idx}",
                document_id=doc_id,
                text=chunk_text,
                metadata={"chunk_index": idx, "file_path": file_path}
            )
            text_chunks.append(chunk)

        return document, text_chunks