import os
import io
import streamlit as st
from rag_engine.services.document_processor import DocumentProcessor
from rag_engine.services.vector_repository import ChromaVectorRepository
from rag_engine.clients.ollama_client import OllamaEmbeddingModel, OllamaInferenceClient, GeminiInferenceClient
from rag_engine.clients.transformer_clients import QueryTransformer
from rag_engine.services.retriever import RetrieverService
from rag_engine.services.prompt_builder import PromptBuilderService
from rag_engine.services.pipeline import RAGPipeline
from rag_engine.services.audio_service import TextToSpeechService, GeminiTTSService
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Local RAG Pipeline", page_icon="🤖", layout="wide")
st.title("Local RAG Pipeline")
st.caption("Powered by Google's Gemini models, ChromaDB, and Sentence-Transformers")

@st.cache_resource
def get_rag_service():
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
    inference_client = GeminiInferenceClient()
    query_transformer = QueryTransformer(inference_client=inference_client)

    pipeline = RAGPipeline(
        document_processor=processor,
        vector_repo=vector_repo,
        embedding_model=embedder,
        retriever=retriever,
        prompt_builder=prompt_builder,
        inference_client=inference_client,
        query_transformer=query_transformer
    )
    
    stt_service = GeminiTTSService()
    
    return pipeline, stt_service

pipeline, stt_service = get_rag_service()

DEFAULT_TOP_K = 8

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything about the Data Privacy Act of 2012"}
    ]

if "audio_key" not in st.session_state:
    st.session_state.audio_key = 0

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

voice_prompt = None

with st.sidebar:
    st.header("Settings")
    enable_tts = st.toggle("Enable Text-to-Speech", value=False)
    autoplay_tts = st.checkbox("Autoplay Audio", value=True, disabled=not enable_tts)
    
    st.divider()
    st.header("Voice Query")
    
    recorded_audio = st.audio_input(
        "Record your question", 
        key=f"voice_input_{st.session_state.audio_key}"
    )
    
    # Process audio input if provided
    if recorded_audio:
        with st.spinner("Transcribing audio via Gemini..."):
            audio_buffer = io.BytesIO(recorded_audio.read())
            voice_prompt = stt_service.transcribe(audio_buffer)
            
            if voice_prompt:
                st.success(f"**Transcribed Query:** \"{voice_prompt}\"")

text_prompt = st.chat_input("What do you want to know?...")

active_prompt = voice_prompt or text_prompt

if active_prompt:
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    
    with st.chat_message("user"):
        st.markdown(active_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and running inference..."):
            answer = pipeline.query(active_prompt, top_k=DEFAULT_TOP_K)
            st.markdown(answer)
            
            # Conditionally generate TTS audio
            if enable_tts:
                tts_service = TextToSpeechService()
                audio_bytes = tts_service.synthesize(answer)
                st.audio(audio_bytes, format="audio/mp3", autoplay=autoplay_tts)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    if voice_prompt:
        st.session_state.audio_key += 1
        st.rerun()