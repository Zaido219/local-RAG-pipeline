from rag_engine.services.document_processor import DocumentProcessor
from rag_engine.services.vector_repository import ChromaVectorRepository
from rag_engine.services.retriever import RetrieverService
from rag_engine.services.prompt_builder import PromptBuilderService
from rag_engine.core.interfaces import BaseEmbeddingModel, BaseInferenceClient

class RAGPipeline:
    def __init__(
        self,
        document_processor: DocumentProcessor,
        vector_repo: ChromaVectorRepository,
        embedding_model: BaseEmbeddingModel,
        retriever: RetrieverService,
        prompt_builder: PromptBuilderService,
        inference_client: BaseInferenceClient,
    ):
        self.document_processor = document_processor
        self.vector_repo = vector_repo
        self.embedding_model = embedding_model
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.inference_client = inference_client

    def ingest_document(self, file_path: str) -> int:
        """Processes, embeds, and stores a document. Returns number of chunks created."""
        _, chunks = self.document_processor.process_document(file_path)
        
        chunk_texts = [c.text for c in chunks]
        embeddings = self.embedding_model.embed_documents(chunk_texts)

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        self.vector_repo.add_chunks(chunks)
        return len(chunks)

    def query(self, user_query: str, top_k: int = 3) -> str:
        """Retrieves relevant chunks, constructs prompt, and generates LLM answer."""
        retrieved_results = self.retriever.retrieve(user_query, top_k=top_k)
        prompt = self.prompt_builder.build_prompt(user_query, retrieved_results)
        response = self.inference_client.generate(prompt)
        return response