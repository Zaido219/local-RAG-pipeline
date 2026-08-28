from typing import Optional
from rag_engine.clients.ollama_client import GeminiInferenceClient
from rag_engine.core.interfaces import QueryTransformerInterface


class QueryTransformer(QueryTransformerInterface):
    """Transforms raw user queries into legally enriched search queries."""

    EXPANSION_SYSTEM_PROMPT = (
        "You are a legal AI assistant. Your task is to rephrase and expand "
        "user questions regarding statutes into domain-specific legal terminology. "
        "Identify the underlying legal concepts, relevant statutory terms, "
        "and potential section topics.\n\n"
        "Do NOT answer the question. Output ONLY a single enriched search query phrase."
    )
    def __init__(self, inference_client:Optional[GeminiInferenceClient] = None):
        self.client = inference_client or GeminiInferenceClient()

    def expand_query(self, raw_query:str) -> str:
        """Transforms natural language queries into legally enriched vector search queries."""
        prompt = (
            f"{self.EXPANSION_SYSTEM_PROMPT}\n\n"
            f"User Question: {raw_query}\n"
            f"Enriched Legal Query:"
        )
        try:
            enriched_keywords = self.client.generate(prompt).strip()
            return f"{raw_query} {enriched_keywords}"
        except Exception:
            return raw_query