from rag_engine.core.models import RetrievalResult

class PromptBuilderService:
    def __init__(self, system_instruction:str | None = None):
        self.system_instruction = system_instruction or (
            "You are a helpful assistant. Answer the user's question using ONLY "
            "the provided context chunks below. If the information is not in the context, "
            "state clearly that you do not know based on the provided documents."
        )
    def build_prompt(self, query:str, results:list[RetrievalResult]) -> str:
        # extract text chunks from the retrieval result model
        context_text = "\n\n---\n\n".join(
            [f"[Source Chunk {i+1}]:\n{res.chunk.text}" for i, res in enumerate(results)]
        )
        # Assemble full augmented prompt
        formatted_prompt = f"""{self.system_instruction}
        CONTEXT INFORMATION:
        {context_text}
        USER QUESTION:
        {query}
        ANSWER:"""

        return formatted_prompt