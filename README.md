# Custom Local RAG Pipeline

A highly modular, domain-driven, and local **Retrieval-Augmented Generation (RAG)** pipeline designed for legal document analysis. This system allows you to ingest documents locally, perform legally enriched semantic search queries, and engage in continuous chat-based conversations using state-of-the-art Large Language Models (LLMs), with support for voice inputs (Speech-to-Text) and audio responses (Text-to-Speech).

---

## 🏗️ Architecture Design & Design Patterns

The codebase is built adhering strictly to the **Domain-Driven Design (DDD)** and the **Dependency Inversion Principle (DIP)**.

```
                  ┌─────────────────────────────────────┐
                  │          Presentation Layer         │
                  │   (app.py - Streamlit Interface)    │
                  └──────────────────┬──────────────────┘
                                     │
                                     ▼
                  ┌─────────────────────────────────────┐
                  │          Application Layer          │
                  │       (rag_engine.services)         │
                  └──────────────────┬──────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────┴────────────────────────────────────┐
│                               Core Layer                                │
│                           (rag_engine.core)                             │
│  - Interfaces (BaseEmbeddingModel, BaseInferenceClient, etc.)           │
│  - Shared Domain Models (Document, TextChunk, etc.)                     │
│  - Custom Domain Exceptions                                             │
└────────────────────────────────────┬────────────────────────────────────┘
                                     ▲
                                     │
                  ┌──────────────────┴──────────────────┐
                  │         Infrastructure/Clients      │
                  │        (rag_engine.clients)         │
                  └─────────────────────────────────────┘
```

### Key Modules:
1. **Domain Layer (`src/rag_engine/core`)**: Serves as the central repository of truth. It contains standard entities and definitions, with **zero external dependencies** on vendor SDKs (ChromaDB, Ollama, etc.).
2. **Application Layer (`src/rag_engine/services`)**: Contains the primary business orchestrators and services:
   - `document_processor.py`: Loads, processes, and chunks documents.
   - `vector_repository.py`: Manages storage and similarity search using ChromaDB.
   - `retriever.py`: Orchestrates retrieve-side queries.
   - `prompt_builder.py`: Synthesizes retrieved context and questions into optimized instruction prompts.
   - `session_memory.py`: Maintains session state across conversations via Redis.
   - `audio_service.py`: Converts text to speech using `gTTS` and speech to text using Gemini.
   - `pipeline.py`: Coordinates the end-to-end ingestion and query pipelines.
3. **Infrastructure Layer (`src/rag_engine/clients`)**: External client implementations that fulfill domain contracts defined in `core`. These can be swapped out transparently without affecting Core or Application layers.
   - `ollama_client.py`: Integrates local embedding models (like `nomic-embed-text`) and inference models (like `qwen2.5:7b`).
   - `transformer_clients.py`: Implements a Gemini-backed `QueryTransformer` to expand standard questions into legal terminology.

---

## ⚡ Features

- **Document Ingestion CLI**: Ingests legal documents (e.g., `data_privacy_act.txt`), chunks them using an overlapping strategy, and indexes them in ChromaDB.
- **Multimodal Streamlit Interface**: Beautiful UI supporting text queries as well as live microphone recording (transcribed via Gemini multimodal capabilities).
- **Legally Enriched Retrieval**: Uses Gemini LLM to expand query terms into deep, domain-specific legal vocabulary for superior retrieval match rates.
- **Text-to-Speech Responses**: Synthesizes assistant responses using Google Text-to-Speech (`gTTS`) with autoplay support.
- **Distributed Session History**: Persists active conversations in Redis to support robust multi-user scaling.
- **Extensive Test Coverage**: Fully unit-tested and integration-tested to ensure reliable operations.

---

## 🛠️ Prerequisites

To run this pipeline locally, ensure you have the following installed and running:

1. **Python 3.10+**
2. **Ollama**:
   - Download and install [Ollama](https://ollama.com).
   - Pull the default embedding model:
     ```bash
     ollama pull nomic-embed-text
     ```
3. **Redis**:
   - Install and run a Redis server (locally on port `6379`).
4. **Google Gemini API Key**:
   - Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/).

---

## 🚀 Getting Started

### 1. Installation

Clone this repository and set up a virtual environment:

```bash
# Set up a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# Install the dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Environment Configuration

Create a `.env` file in the root directory and configure the environment variables:

```ini
# Gemini API Configuration
API_KEY=your_gemini_api_key_here

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434

# Vector DB Configuration
CHROMA_PERSIST_DIR=./chroma_db

# Redis Memory Configuration
MEMORY_TYPE=redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Target Document Setup (for ingestion)
DOCUMENT_PATH=data_privacy_act.txt
```

### 3. Step 1: Ingesting Documents

Before running the chat interface, run the CLI ingestion script to parse and index your target document into ChromaDB:

```bash
python main.py
```

### 4. Step 2: Running the Web App

Launch the interactive Streamlit chat interface:

```bash
streamlit run app.py
```

Once loaded, you can ask questions about the ingested document, enable text-to-speech feedback, or record audio queries directly from your browser!

---

## 🧪 Testing

To run the complete test suite (unit and integration tests) and verify the components:

```bash
pytest
```

---

## 🛡️ Architecture & Submodule Deep Dives

For further module-specific details, architecture limits, and dependency rules, refer to:
* [Core Module Details (`src/rag_engine/core/README.md`)](./src/rag_engine/core/readme.md)
* [Services Module Details (`src/rag_engine/services/README.md`)](./src/rag_engine/services/readme.md)
