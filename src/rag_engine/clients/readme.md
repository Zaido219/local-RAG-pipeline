# Clients Module (`rag_engine.clients`)

The `clients` directory forms the **Infrastructure Layer** of the RAG pipeline. It contains concrete adapters responsible for direct communication with external APIs, SDKs, and local modeling servers (such as Ollama and Google's Gemini models).

---

## 🎯 Integration Overview

* **`ollama_client.py`**:
  - **`OllamaEmbeddingModel`**: Connects to a local Ollama server running `nomic-embed-text` to generate vector representations for both text chunks and search queries.
  - **`OllamaInferenceClient`**: Connects to local LLMs (e.g., `qwen2.5:7b`) for localized context-augmented generation.
  - **`GeminiInferenceClient`**: Connects to Google's remote GenAI API (`gemini-3.6-flash`) using standard credentials for fast and highly capable remote inference.
* **`transformer_clients.py`**:
  - **`QueryTransformer`**: Uses an underlying inference client (such as Gemini) to dynamically translate, rephrase, and enrich natural language queries into legally dense keywords. This ensures optimal relevance when retrieving matches from ChromaDB.

---

## 🛡️ Architectural Rules

1. **Implement Core Contracts**: Every client must subclass and fully implement the corresponding interface defined in `rag_engine.core.interfaces` (e.g., `BaseEmbeddingModel`, `BaseInferenceClient`, `QueryTransformerInterface`).
2. **Adhere to the Dependency Inversion Principle (DIP)**: Downstream code in the `services` layer must interact with these clients via their core interfaces, never through concrete client implementations.
3. **Encapsulate Vendor SDKs**: Direct imports and usage of third-party SDKs (such as `ollama` or `google-genai`) must be entirely contained within this module to prevent infrastructure leaks.
4. **No Business or Orchestration Logic**: Clients are pure adapters. They should focus exclusively on translating inputs, executing HTTP/network calls, parsing response payloads, and handling network-level exceptions.

---

## 📁 Expected File Structure

```text
src/rag_engine/clients/
├── readme.md                <-- Module overview (this file)
├── __init__.py              <-- Package exports for concrete clients
├── ollama_client.py         <-- Integrations for local models and Gemini API
└── transformer_clients.py   <-- LLM-based query transformation & enrichment
```
