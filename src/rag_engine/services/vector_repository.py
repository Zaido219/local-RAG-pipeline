from typing import List, Dict, Any
import chromadb
from rag_engine.core.models import TextChunk, RetrievalResult
from rag_engine.core.interfaces import VectoryDbBaseClass


class ChromaVectorRepository(VectoryDbBaseClass):
    """Vector database implementation using ChromaDB with metadata enrichment support."""

    def __init__(self, persist_directory: str, collection_name: str):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Explicitly set distance metric
        )

    def add_chunks(self, chunks: List[TextChunk]) -> None:
        """Stores text chunks and their embeddings into ChromaDB."""
        if not chunks:
            return

        ids = [chunk.chunk_id for chunk in chunks]
        # Store enriched embed_text as the primary searchable document payload
        documents = [chunk.get_searchable_text() for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks]

        # Explicitly pack raw_text and embed_text into metadata for reconstruction
        metadatas: List[Dict[str, Any]] = []
        for chunk in chunks:
            meta = {
                **chunk.metadata,
                "document_id": chunk.document_id,
                "raw_text": chunk.text,
                "embed_text": chunk.embed_text or ""
            }
            metadatas.append(meta)

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def similarity_search(self, query_embedding: List[float], top_k: int = 5) -> List[RetrievalResult]:
        """Performs vector similarity search and maps raw distances to normalized scores."""
        raw_result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # Chroma returns nested lists corresponding to batch queries
        ids = raw_result["ids"][0] if raw_result.get("ids") else []
        metadatas = raw_result["metadatas"][0] if raw_result.get("metadatas") else []
        distances = raw_result["distances"][0] if raw_result.get("distances") else []

        retrieval_results: List[RetrievalResult] = []

        for chunk_id, metadata, distance in zip(ids, metadatas, distances):
            # Extract fields back from metadata
            doc_id = metadata.pop("document_id", "")
            raw_text = metadata.pop("raw_text", "")
            embed_text = metadata.pop("embed_text", None)

            # Reconstruct domain model
            text_chunk = TextChunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                text=raw_text,
                embed_text=embed_text,
                metadata=metadata  # Remaining clean user metadata
            )

            # Convert distance to similarity score
            similarity_score = 1.0 / (1.0 + distance)

            retrieval_results.append(
                RetrievalResult(
                    chunk=text_chunk,
                    score=similarity_score,
                    retrieval_type="vector"
                )
            )

        return retrieval_results