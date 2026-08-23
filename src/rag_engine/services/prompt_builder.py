from rag_engine.core.models import RetrievalResult
from textwrap import dedent

class PromptBuilderService:
    def __init__(self, system_instruction:str | None = None):
        self.system_instruction = system_instruction or (
            "You are a helpful assistant. Answer the user's question using ONLY "
            "the provided context chunks below. If the information is not in the context, "
            "state clearly that you do not know based on the provided documents."
        )
    def build_prompt(self, query: str, results: list[RetrievalResult]) -> str:
        context_text = "\n\n---\n\n".join(
            [f"[Source Chunk {i+1}]:\n{res.chunk.text}" for i, res in enumerate(results)]
        )

        formatted_prompt = f"""{self.system_instruction}

    CONTEXT INFORMATION:
    {context_text}

    USER QUESTION:
    {query}

    ANSWER:"""
        return dedent(formatted_prompt).strip()