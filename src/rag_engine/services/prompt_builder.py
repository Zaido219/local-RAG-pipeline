from typing import List, Optional
from textwrap import dedent
from rag_engine.core.models import RetrievalResult


class PromptBuilderService():
    """Formats retrieved context results and user queries into structured LLM prompts."""

    DEFAULT_SYSTEM_INSTRUCTION = (
        "You are an expert legal assistant specializing in statutory compliance. "
        "Answer the user's question accurately using ONLY the provided context chunks below. "
        "For every claim or sanction you mention, explicitly cite the Chapter and Section number. "
        "If the answer cannot be derived from the context, state clearly that the provided documents "
        "do not contain sufficient information."
    )

    def __init__(self, system_instruction: Optional[str] = None):
        self.system_instruction = system_instruction or self.DEFAULT_SYSTEM_INSTRUCTION

    def _format_chunk_citation(self, idx: int, result: RetrievalResult) -> str:
        """Formats an individual chunk into a structured context block with legal citations."""
        chunk = result.chunk
        meta = chunk.metadata

        # Extract structured metadata tags if available
        chap_num = meta.get("chapter_number", "N/A")
        chap_title = meta.get("chapter_title", "")
        sec_num = meta.get("section_number", "N/A")
        sec_title = meta.get("section_title", "")

        header = f"[Source {idx + 1} | Chapter {chap_num}: {chap_title} | Sec. {sec_num}: {sec_title}]"
        
        return f"{header}\n{chunk.text}"

    def build_prompt(self, query: str, results: List[RetrievalResult]) -> str:
        """Constructs the complete prompt string combining system instructions, formatted context, and query."""
        if not results:
            context_text = "No relevant document chunks were retrieved."
        else:
            formatted_chunks = [
                self._format_chunk_citation(i, res) for i, res in enumerate(results)
            ]
            context_text = "\n\n---\n\n".join(formatted_chunks)

        formatted_prompt = f"""\
{self.system_instruction}

CONTEXT INFORMATION:
{context_text}

USER QUESTION:
{query}

ANSWER:"""

        return dedent(formatted_prompt).strip()