from typing import List, Optional, Dict
from textwrap import dedent
from rag_engine.core.models import RetrievalResult


class PromptBuilderService():
    """Formats retrieved context results and user queries into structured LLM prompts."""

    DEFAULT_SYSTEM_INSTRUCTION = (
        "You are an expert legal assistant specializing in statutory compliance. "
        "Use the CONVERSATION HISTORY to maintain dialogue context and remember user details. "
        "Use the CONTEXT INFORMATION from retrieved documents to answer legal questions. "
        "For claims or sanctions derived from documents, explicitly cite Chapter and Section numbers. "
        "If a legal question cannot be derived from the document context, state that the documents "
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

    def _format_chat_history(self, chat_history:Optional[List[Dict[str, str]]]) -> str:
        """Converts raw list of message dicts into a formatted transcript."""
        if not chat_history:
            return "No previous conversation history"
        formatted_turns = []

        for msg in chat_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_turns.append(f"{role}: {msg['content']}")

        return "\n".join(formatted_turns)

    
    def build_prompt(self, query: str, results: List[RetrievalResult], chat_history:Optional[List[Dict[str,str]]] = None) -> str:
        """Constructs the complete prompt string combining system instructions, formatted context, chat history and query."""
        if not results:
            context_text = "No relevant document chunks were retrieved."
        else:
            formatted_chunks = [
                self._format_chunk_citation(i, res) for i, res in enumerate(results)
            ]
            context_text = "\n\n---\n\n".join(formatted_chunks)

        history_text = self._format_chat_history(chat_history)

        formatted_prompt = f"""\
{self.system_instruction}

CONVERSATION HISTORY:
{history_text}

RETRIEVED DOCUMENT CONTEXT:
{context_text}

USER QUESTION:
{query}

ANSWER:"""

        return dedent(formatted_prompt).strip() 