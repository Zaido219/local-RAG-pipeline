import os
import streamlit as st
from rag_engine.services.document_processor import DocumentProcessor
from rag_engine.services.vector_repository import ChromaVectorRepository
from rag_engine.clients.ollama_client import OllamaEmbeddingModel, OllamaInferenceClient,GeminiInferenceClient
from rag_engine.services.retriever import RetrieverService
from rag_engine.services.prompt_builder import PromptBuilderService
from rag_engine.services.pipeline import RAGPipeline
from rag_engine.services.audio_service import TextToSpeechService
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Local RAG Pipeline", page_icon="🤖", layout="wide")
st.title("Local RAG Pipeline")
st.caption("Powered by qwen2.5:7b, ChromaDB, and Sentence-Transformers")

@st.cache_resource
def get_rag_service() -> RAGPipeline:
    chroma_path = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    processor = DocumentProcessor()
    vector_repo = ChromaVectorRepository(
        persist_directory=chroma_path, 
        collection_name="constitution_docs"
    )

    
    embedder = OllamaEmbeddingModel(model_name="nomic-embed-text", host=ollama_url)
    retriever = RetrieverService(vector_repo=vector_repo, embedding_model=embedder)
    prompt_builder = PromptBuilderService()
    qwen_inference_client = OllamaInferenceClient(model_name="qwen2.5:7b", host=ollama_url)
    inference_client = GeminiInferenceClient()

    pipeline = RAGPipeline(
        document_processor=processor,
        vector_repo=vector_repo,
        embedding_model=embedder,
        retriever=retriever,
        prompt_builder=prompt_builder,
        inference_client=inference_client
    )
    
    return pipeline

pipeline = get_rag_service()

DEFAULT_TOP_K = 2


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything about your ingested documents!"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if prompt := st.chat_input("What do you want to know?..."):
    tts_service = TextToSpeechService()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and running inference..."):
            answer = pipeline.query(prompt, top_k=DEFAULT_TOP_K)
            # tts
            audio_bytes = tts_service.synthesize(answer)
            st.audio(audio_bytes, format="audio/mp3", autoplay=True)

    st.session_state.messages.append({"role": "assistant", "content": answer})