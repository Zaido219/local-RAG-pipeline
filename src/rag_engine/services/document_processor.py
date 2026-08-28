import json
import uuid
from typing import List, Dict, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_engine.core.models import Document, TextChunk


class DocumentProcessor:
    """Processes structured JSON statute documents into metadata-enriched RAG chunks."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def _flatten_section_text(self, section: Dict[str, Any]) -> str:
        """Combines main section text and any nested subsections into a single content block."""
        text_parts = [section.get("text", "")]
        
        subsections = section.get("subsections", [])
        if subsections:
            for sub in subsections:
                identifier = sub.get("identifier", "")
                term = sub.get("term", "")
                sub_text = sub.get("text", "")
                
                # Format nested definitions cleanly for RAG retrieval context
                if term:
                    header = f"({identifier}) {term}:"
                else:
                    header = f"({identifier}):"
                
                text_parts.append(f"{header} {sub_text}")
                
        return "\n\n".join(filter(None, text_parts))

    def _extract_structural_blocks(self, raw_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Traverses the JSON schema to extract global metadata and flattened section blocks."""
        doc_metadata = raw_data.get("document_metadata", {})
        chapters = raw_data.get("chapters", [])
        blocks: List[Dict[str, Any]] = []

        for chapter in chapters:
            chap_num = chapter.get("chapter_number", "")
            chap_title = chapter.get("chapter_title", "")
            sections = chapter.get("sections", [])

            for section in sections:
                sec_num = section.get("section_number", "")
                sec_title = section.get("section_title", "")
                combined_text = self._flatten_section_text(section)

                blocks.append({
                    "chapter_number": chap_num,
                    "chapter_title": chap_title,
                    "section_number": sec_num,
                    "section_title": sec_title,
                    "text": combined_text
                })

        return doc_metadata, blocks

    def process_document(self, file_path: str) -> Tuple[Document, List[TextChunk]]:
        """Reads the JSON file, extracts structural units, and generates enriched TextChunks."""
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        doc_id = str(uuid.uuid4())
        content_str = json.dumps(raw_data)
        
        doc_metadata, blocks = self._extract_structural_blocks(raw_data)

        # Propagate global document metadata across all chunks for stronger contextual search
        base_metadata = {
            "file_path": file_path,
            "doc_type": "statute",
            **doc_metadata
        }

        document = Document(
            id=doc_id,
            content=content_str,
            metadata=base_metadata
        )

        text_chunks: List[TextChunk] = []
        chunk_idx = 0

        for block in blocks:
            sub_texts = self.text_splitter.split_text(block["text"])

            for sub_text in sub_texts:
                header_prefix = (
                    f"[Chapter {block['chapter_number']}: {block['chapter_title']} | "
                    f"Sec. {block['section_number']}: {block['section_title']}]"
                )
                embed_text = f"{header_prefix}\n{sub_text}"

                chunk_metadata = {
                    "chunk_index": chunk_idx,
                    **base_metadata,
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
                    metadata=chunk_metadata
                )
                text_chunks.append(chunk)
                chunk_idx += 1

        return document, text_chunks