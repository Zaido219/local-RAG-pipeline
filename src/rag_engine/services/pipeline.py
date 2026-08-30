from typing import List
from rag_engine.core.models import Document, TextChunk
from rag_engine.core.interfaces import (
    VectoryDbBaseClass,
    BaseEmbeddingModel,
    BaseInferenceClient,
    SessionMemoryInterface
)
from rag_engine.services.document_processor import DocumentProcessor
from rag_engine.services.retriever import RetrieverService
from rag_engine.services.prompt_builder import PromptBuilderService
from rag_engine.clients.transformer_clients import QueryTransformer #!is query transformer not being actually used in here ?


class RAGPipeline:
    """Orchestrates document ingestion and contextual RAG querying pipelines."""

    def __init__(
        self,
        document_processor: DocumentProcessor,
        vector_repo: VectoryDbBaseClass,
        embedding_model: BaseEmbeddingModel,
        retriever: RetrieverService,
        prompt_builder: PromptBuilderService,
        inference_client: BaseInferenceClient,
        # query transformer will be injected as an optional dependency
        query_transformer: None,
        session_memory:SessionMemoryInterface
    ):
        self.document_processor = document_processor
        self.vector_repo = vector_repo
        self.embedding_model = embedding_model
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.inference_client = inference_client
        self.session_memory = session_memory
        self.query_transformer = query_transformer

    def ingest_document(self, file_path: str) -> int:
        """Processes, embeds (with context headers), and stores a document."""
        _, chunks = self.document_processor.process_document(file_path)
        if not chunks:
            return 0

        # CRITICAL FIX: Embed using header-enriched searchable text
        chunk_texts = [c.get_searchable_text() for c in chunks]
        embeddings = self.embedding_model.embed_documents(chunk_texts)

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        self.vector_repo.add_chunks(chunks)
        return len(chunks)

    def query(self, user_query: str, session_id:str, top_k: int = 8) -> str:
        """Retrieves relevant chunks, constructs prompt, and generates LLM response."""
        # add user query to session memory and fetch active session history
        self.session_memory.add_message(session_id, role="user", content=user_query)
        chat_history = self.session_memory.get_session_history(session_id)

        # expand  the query if query_transformer is available
        search_query = user_query
        if self.query_transformer:
            search_query = self.query_transformer.expand_query(user_query)

        retrieved_results = self.retriever.retrieve(search_query, top_k=top_k)

        prompt = self.prompt_builder.build_prompt(user_query, retrieved_results, chat_history)

        answer =  self.inference_client.generate(prompt)

        # persist assistant response to redis
        self.session_memory.add_message(session_id, role="assistant", content=answer)

        return answer