from rag_engine.core.models import TextChunk, RetrievalResult
from abc import ABC,abstractmethod
import chromadb 

class VectoryDbBaseClass(ABC):
    @abstractmethod
    def add_chunks(self,chunks:list[TextChunk]) -> None:
        pass
    
    @abstractmethod
    def similarity_search(self, query_embedding:list[float], top_k:int = 5) -> list[RetrievalResult]:
        pass



class ChromaVectorRepository(VectoryDbBaseClass):
    def __init__(self, persist_directory:str, collection_name:str):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(collection_name)
        

    def add_chunks(self, chunks:list[TextChunk]):
        id = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks]
        metadata = [
            {**chunk.metadata, "document_id": chunk.document_id}
            for chunk in chunks
        ]

        self.collection.add(
            ids=id,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadata,
        )

    def similarity_search(self, query_embedding, top_k = 5) -> list[RetrievalResult]:
        raw_result = self.collection.query(
            query_embeddings=[query_embedding], 
            n_results=top_k
        )
        ids = raw_result["ids"][0]
        documents = raw_result["documents"][0]
        metadatas = raw_result["metadatas"][0]
        distances = raw_result["distances"][0]

        retrieval_results = []

        for chunk_id, text, metadata, distance in zip(ids,documents, metadatas, distances):
            doc_id = metadata.get("document_id", "")
            # reconstruct text chunk
            textChunk = TextChunk(
                chunk_id= chunk_id,
                document_id=doc_id,
                text=text,
                metadata=metadata
            )
            # reconstruct retrieval result, mapping distance to score
            retrieval_result = RetrievalResult(
                chunk=textChunk,
                score=distance
            )

            # append to retrieval_result
            retrieval_results.append(retrieval_result)
        
        return retrieval_results   
    