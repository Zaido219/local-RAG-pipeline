# Core Module (`rag_engine.core`)

The `core` directory serves as the **Domain Layer** of the RAG engine architecture. It contains the central domain entities, dataclasses, interfaces, and custom exceptions that define data representation across the system.

---

## 🎯 Primary Responsibilities

* **Domain Models (`models.py`):** Holds shared dataclasses/Pydantic schemas (e.g., `Document`, `TextChunk`, `QueryResult`) that represent core data structures.
* **Abstract Interfaces:** Defines base contracts (e.g., `BaseVectorStore`, `BaseLLMClient`) to enforce the **Dependency Inversion Principle (DIP)**.
* **Domain Exceptions:** Defines custom error types (e.g., `ChunkingError`, `VectorSearchError`) for uniform error handling.
* **Constants & Type Definitions:** Standardizes system-wide constants, enums (e.g., file types, distance metrics), and custom type aliases.

---

## 🛡️ Architectural Rules

1. **Zero External Dependencies:** `core` should rely on standard Python libraries (e.g., `dataclasses`, `typing`, `abc`) or lightweight validation primitives. It must **never** depend on vendor SDKs like ChromaDB, Ollama, or web frameworks.
2. **Inward Dependency Direction:** Services (`services/`) and clients (`clients/`) depend on `core`. **`core` must never import from `services` or `clients`.**
3. **Decoupling Rule:** Changes to third-party integrations (e.g., swapping vector databases or switching local LLMs) must **never** force modifications inside `core`.

---

## 📁 Expected File Structure

```text
src/rag_engine/core/
├── README.md         <-- Module overview (this file)
├── __init__.py       <-- Package exports for core types
├── models.py         <-- Shared data structures (Document, TextChunk, etc.)
├── exceptions.py     <-- Domain-specific exception hierarchy
└── interfaces.py     <-- Abstract base classes and contracts