import uuid
import re
from typing import List, Dict, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_engine.core.models import Document, TextChunk


class DocumentProcessor:
    """Processes legal documents into structured, metadata-enriched chunks for RAG pipelines."""

    CHAPTER_PATTERN = re.compile(r"^CHAPTER\s+([IVXLCDM]+)\s*(.*)$", re.IGNORECASE)
    SECTION_PATTERN = re.compile(r"^(?:SEC\.|SECTION)\s+(\d+)\.\s*(.*?)(?:\s*–|\s*—|\s*-|\.|$)", re.IGNORECASE)

    def __init__(self, chunk_size: int = 1000, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Industry-standard recursive splitter handling natural boundaries
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def _extract_structural_blocks(self, content: str) -> List[Dict[str, Any]]:
        """Parses legal content line-by-line, tracking Chapter and Section state machine."""
        lines = content.split("\n")
        blocks: List[Dict[str, Any]] = []
        
        current_chapter_num = "GENERAL"
        current_chapter_title = "PREAMBLE"
        current_sec_num = "0"
        current_sec_title = "Header Info"
        
        buffer: List[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.lower() == "back to top":  # Filter out known web noise
                continue

            # Check for Chapter match
            chap_match = self.CHAPTER_PATTERN.match(stripped)
            if chap_match:
                # Flush previous accumulated text
                if buffer:
                    blocks.append({
                        "chapter_number": current_chapter_num,
                        "chapter_title": current_chapter_title,
                        "section_number": current_sec_num,
                        "section_title": current_sec_title,
                        "text": "\n".join(buffer)
                    })
                    buffer = []
                current_chapter_num = chap_match.group(1).upper()
                current_chapter_title = chap_match.group(2).strip()
                continue

            # Check for Section match
            sec_match = self.SECTION_PATTERN.match(stripped)
            if sec_match:
                if buffer:
                    blocks.append({
                        "chapter_number": current_chapter_num,
                        "chapter_title": current_chapter_title,
                        "section_number": current_sec_num,
                        "section_title": current_sec_title,
                        "text": "\n".join(buffer)
                    })
                    buffer = []
                current_sec_num = sec_match.group(1)
                current_sec_title = sec_match.group(2).strip()
                
            buffer.append(stripped)

        # Flush final block
        if buffer:
            blocks.append({
                "chapter_number": current_chapter_num,
                "chapter_title": current_chapter_title,
                "section_number": current_sec_num,
                "section_title": current_sec_title,
                "text": "\n".join(buffer)
            })

        return blocks

    def process_document(self, file_path: str) -> Tuple[Document, List[TextChunk]]:
        """Reads document, builds metadata tree, and generates enriched TextChunks."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        doc_id = str(uuid.uuid4())
        document = Document(
            id=doc_id,
            content=content,
            metadata={"file_path": file_path, "doc_type": "statute"}
        )

        blocks = self._extract_structural_blocks(content)
        text_chunks: List[TextChunk] = []
        chunk_idx = 0

        for block in blocks:
            # Standard sub-splitting via LangChain for large sections
            sub_texts = self.text_splitter.split_text(block["text"])

            for sub_text in sub_texts:
                # Context enrichment for embed_text
                header_prefix = (
                    f"[Chapter {block['chapter_number']}: {block['chapter_title']} | "
                    f"Sec. {block['section_number']}: {block['section_title']}]"
                )
                embed_text = f"{header_prefix}\n{sub_text}"

                metadata = {
                    "chunk_index": chunk_idx,
                    "file_path": file_path,
                    "chapter_number": block["chapter_number"],
                    "chapter_title": block["chapter_title"],
                    "section_number": block["section_number"],
                    "section_title": block["section_title"]
                }

                chunk = TextChunk(
                    chunk_id=f"{doc_id}_{chunk_idx}",
                    document_id=doc_id,
                    text=sub_text,
                    embed_text=embed_text,
                    metadata=metadata
                )
                text_chunks.append(chunk)
                chunk_idx += 1

        return document, text_chunks