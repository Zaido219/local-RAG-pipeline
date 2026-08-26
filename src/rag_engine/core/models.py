from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class Document:
    id:str
    content:str
    doc_type:str = "legal_act"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TextChunk:
    chunk_id:str
    document_id:str
    text:str
    # Search models perform better when embedded text includes structural context headers
    embed_text:Optional[str] = None
    embedding:Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    def get_searchable_text(self) -> str:
        return self.embed_text if self.embed_text else self.text


@dataclass
class RetrievalResult:
    #will reference text chunk to preserve document_id and metadata for citations
    chunk : TextChunk
    score:float
    retrieval_type:str = "vector"
