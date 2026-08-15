from dataclasses import dataclass, field
from typing import List, Optional, Any

@dataclass
class Document:
    id:str
    content:str
    metadata: dict = field(default_factory=dict)


@dataclass
class TextChunk:
    chunk_id:str
    document_id:str
    text:str
    embedding:Optional[List[float]] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    #will reference text chunk to preserve document_id and metadata for citations
    chunk : TextChunk
    score:float
