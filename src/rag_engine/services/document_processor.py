from rag_engine.core.models import Document,TextChunk
import uuid

class DocumentProcessor:
    def __init__(self):
        pass

    def _chunk_text(self,text:str, chunk_size:int,overlap:int) -> list[str]:
        chunks = []
        start = 0
        text_length = len(text)

        if overlap >= chunk_size :
            raise ValueError("Overlap must be strictly less than the chunk size")
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            # move window forward accounting for overlap
            start += chunk_size - overlap

        return chunks

    def process_document(self, file_path, chunk_size=500, overlap=50):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            doc_id = str(uuid.uuid4())
            document = Document(
                id=doc_id,
                content=content,
                metadata={"file_path" : file_path}
            )
        # generate the text chunk
        raw_chunks = self._chunk_text(content, chunk_size=chunk_size, overlap=overlap)
        text_chunks = []

        for idx, chunk_text in enumerate(raw_chunks):
            chunk = TextChunk(
                chunk_id=f"{doc_id}_{idx}", # or str(uuid.uuid4())
                document_id=doc_id,
                text=chunk_text,
                metadata={"chunk_index": idx, "file_path": file_path}
            )
            text_chunks.append(chunk)

        return document, text_chunks
