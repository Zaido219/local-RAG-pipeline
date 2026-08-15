# Services Module (`rag_engine.services`)

The `services` directory forms the **Application Logic Layer** of the RAG pipeline. It contains the operational services responsible for text processing, vector database interactions, context retrieval, and prompt augmentation.

---

## 🎯 Service Overview

* **`document_processor.py`**: Handles document loading, text extraction, and text chunking strategies (converting raw `Document` objects into vectorized `TextChunk` units).
* **`vector_repository.py`**: Interacts directly with the vector database (ChromaDB) to manage embedding persistence, collection initialization, and similarity search queries.
* **`retriever.py`**: Orchestrates semantic search requests by coordinating with `vector_repository.py` and returning structured `RetrievalResult` objects to downstream callers.
* **`prompt_builder.py`**: Assembles user prompts and retrieved text chunks into formatted, context-augmented prompts for LLM inference.

---

## 🛡️ Architectural Rules

1. **Depend on `core` Entities:** All services must import shared domain models (`Document`, `TextChunk`, `RetrievalResult`) from `rag_engine.core.models`.
2. **Encapsulate External Libraries:** Vendor-specific code (e.g., ChromaDB calls) should stay contained within its dedicated service rather than leaking into application interfaces.
3. **Stateless Processing:** Except for persistent database connections, services should remain stateless to simplify testing and async execution.

---

## 📁 Expected File Structure

```text
src/rag_engine/services/
├── README.md               <-- Module overview
├── __init__.py             <-- Package exports for service instances
├── document_processor.py   <-- Ingestion & chunking service
├── prompt_builder.py       <-- Context formatting service
├── retriever.py            <-- Retrieval orchestration service
└── vector_repository.py    <-- Vector DB interface (ChromaDB)